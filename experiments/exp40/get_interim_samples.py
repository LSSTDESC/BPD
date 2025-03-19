#!/usr/bin/env python3
import os
from functools import partial

os.environ["JAX_ENABLE_X64"] = "True"

import jax.numpy as jnp
import typer
from jax import jit, random, vmap
from jax._src.prng import PRNGKeyArray

from bpd import DATA_DIR
from bpd.draw import draw_gaussian
from bpd.initialization import init_with_truth
from bpd.io import save_dataset
from bpd.likelihood import gaussian_image_loglikelihood
from bpd.pipelines import pipeline_interim_samples_one_galaxy
from bpd.prior import interim_gprops_logprior
from bpd.sample import (
    get_target_images,
    get_true_params_from_galaxy_params,
    sample_target_galaxy_params_simple,
)


def sample_prior(
    rng_key: PRNGKeyArray,
    *,
    mean_logflux: float,
    sigma_logflux: float,
    shape_noise: float,
    g1: float = 0.02,
    g2: float = 0.0,
) -> dict[str, float]:
    k1, k2 = random.split(rng_key)

    lf = random.normal(k1) * sigma_logflux + mean_logflux
    other_params = sample_target_galaxy_params_simple(
        k2, shape_noise=shape_noise, g1=g1, g2=g2
    )

    return {"lf": lf, "hlr": 0.8, **other_params}


def main(
    seed: int,
    tag: str,
    mode: str = "",
    n_gals: int = 2500,
    n_samples_per_gal: int = 300,
    mean_logflux: float = 2.6,
    sigma_logflux: float = 0.4,
    g1: float = 0.02,
    g2: float = 0.0,
    shape_noise: float = 0.1,
    sigma_e_int: float = 0.15,
    slen: int = 63,
    fft_size: int = 256,
    background: float = 1.0,
    initial_step_size: float = 1e-3,
):
    assert (g1 > 0 and mode == "plus") or (g1 < 0 and mode == "minus") or (not mode)

    rng_key = random.key(seed)
    pkey, nkey, gkey = random.split(rng_key, 3)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)

    # galaxy parameters from prior
    pkeys = random.split(pkey, n_gals)
    _get_galaxy_params = partial(
        sample_prior,
        shape_noise=shape_noise,
        mean_logflux=mean_logflux,
        sigma_logflux=sigma_logflux,
        g1=g1,
        g2=g2,
    )
    galaxy_params = vmap(_get_galaxy_params)(pkeys)
    assert galaxy_params["x"].shape == (n_gals,)
    assert galaxy_params["e1"].shape == (n_gals,)

    # now get corresponding target images
    draw_params = {**galaxy_params}
    draw_params["f"] = 10 ** draw_params.pop("lf")
    target_images = get_target_images(
        nkey, draw_params, background=background, slen=slen, draw_type="gaussian"
    )
    assert target_images.shape == (n_gals, slen, slen)

    # interim samples are on 'sheared ellipticity'
    true_params = vmap(get_true_params_from_galaxy_params)(galaxy_params)
    fixed_params = {
        "x": true_params.pop("x"),
        "y": true_params.pop("y"),
        "f": 10 ** true_params.pop("lf"),
        "hlr": true_params.pop("hlr"),
    }

    # setup prior and likelihood
    _logprior = partial(
        interim_gprops_logprior,
        sigma_e=sigma_e_int,
        free_flux_hlr=False,
        free_dxdy=False,
    )
    _draw_fnc = partial(draw_gaussian, slen=slen, fft_size=fft_size)
    _loglikelihood = partial(
        gaussian_image_loglikelihood,
        draw_fnc=_draw_fnc,
        background=background,
        free_flux_hlr=False,
        free_dxdy=False,
    )

    # prepare pipelines
    gkeys = random.split(gkey, n_gals)
    pipe = partial(
        pipeline_interim_samples_one_galaxy,
        initialization_fnc=init_with_truth,
        logprior=_logprior,
        loglikelihood=_loglikelihood,
        n_samples=n_samples_per_gal,
        initial_step_size=initial_step_size,
    )
    vpipe = vmap(jit(pipe))

    # compilation on single target image
    _ = vpipe(
        gkeys[0, None],
        target_images[0, None],
        {k: v[0, None] for k, v in fixed_params.items()},
        {k: v[0, None] for k, v in true_params.items()},
    )

    samples = vpipe(gkeys, target_images, fixed_params, true_params)
    e_post = jnp.stack([samples["e1"], samples["e2"]], axis=-1)

    extra_tag = f"_{mode}" if mode else ""
    fpath = dirpath / f"interim_samples_{seed}{extra_tag}.npz"

    save_dataset(
        {
            "e_post": e_post,
            "e1_true": draw_params["e1"],
            "e2_true": draw_params["e2"],
            "f": draw_params["f"],
            "true_g": jnp.array([g1, g2]),
            "sigma_e": shape_noise,
            "sigma_e_int": sigma_e_int,
            "mean_logflux": mean_logflux,
            "sigma_logflux": sigma_logflux,
        },
        fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
