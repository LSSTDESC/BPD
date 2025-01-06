#!/usr/bin/env python3
from functools import partial

import jax.numpy as jnp
import typer
from jax import jit, random, vmap

from bpd import DATA_DIR
from bpd.draw import draw_gaussian
from bpd.initialization import init_with_truth
from bpd.io import save_dataset
from bpd.pipelines.image_samples import (
    get_target_images,
    get_true_params_from_galaxy_params,
    loglikelihood,
    logprior,
    pipeline_interim_samples_one_galaxy,
    sample_target_galaxy_params_simple,
)


def main(
    seed: int,
    n_gals: int = 1000,  # technically, in this file it means 'noise realizations'
    n_samples_per_gal: int = 100,
    g1: float = 0.02,
    g2: float = 0.0,
    lf: float = 6.0,  # ~ SNR = 1000
    hlr: float = 0.8,
    shape_noise: float = 1e-4,
    sigma_e_int: float = 4e-2,
    slen: int = 63,
    fft_size: int = 256,
    background: float = 1.0,
    initial_step_size: float = 1e-3,
):
    rng_key = random.key(seed)
    pkey, nkey, gkey = random.split(rng_key, 3)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / f"test_fixed_shear_inference_images_{seed}"
    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)

    # galaxy parameters from prior
    pkeys = random.split(pkey, n_gals)
    _get_galaxy_params = partial(
        sample_target_galaxy_params_simple, g1=g1, g2=g2, shape_noise=shape_noise
    )
    galaxy_params = vmap(_get_galaxy_params)(pkeys)
    assert galaxy_params["x"].shape == (n_gals,)
    assert galaxy_params["e1"].shape == (n_gals,)
    assert "lf" not in galaxy_params
    fixed_params = {"f": 10 ** jnp.full((n_gals,), lf), "hlr": jnp.full((n_gals,), hlr)}

    # now get corresponding target images
    # we use the same flux and hlr for every galaxy in this experiment (and fix them in sampling)
    draw_params = {**galaxy_params, **fixed_params}
    target_images = get_target_images(
        nkey, draw_params, background=background, slen=slen
    )
    assert target_images.shape == (n_gals, slen, slen)

    # interim samples are on 'sheared ellipticity'
    true_params = vmap(get_true_params_from_galaxy_params)(galaxy_params)

    # x and y are dithered in the true images but fixed
    x = true_params.pop("x")
    y = true_params.pop("y")
    fixed_params = {"x": x, "y": y, **fixed_params}

    # setup prior and likelihood
    _logprior = partial(
        logprior, sigma_e=sigma_e_int, free_flux_hlr=False, free_dxdy=False
    )

    _draw_fnc = partial(draw_gaussian, slen=slen, fft_size=fft_size)
    _loglikelihood = partial(
        loglikelihood,
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
        {k: v[0, None] for k, v in true_params.items()},
        target_images[0, None],
        {k: v[0, None] for k, v in fixed_params.items()},
    )

    samples = vpipe(gkeys, true_params, target_images, fixed_params)
    e_post = jnp.stack([samples["e1"], samples["e2"]], axis=-1)
    fpath = dirpath / f"e_post_{seed}.npz"

    save_dataset(
        {
            "e_post": e_post,
            "true_g": jnp.array([g1, g2]),
            "sigma_e": shape_noise,
            "sigma_e_int": sigma_e_int,
            "e1": draw_params["e1"],
            "e2": draw_params["e2"],
        },
        fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
