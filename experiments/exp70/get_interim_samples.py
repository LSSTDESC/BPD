#!/usr/bin/env python3
import math
import os
from functools import partial

os.environ["JAX_ENABLE_X64"] = "True"

import jax.numpy as jnp
import typer
from jax import Array, jit, random, vmap
from jax._src.prng import PRNGKeyArray

from bpd import DATA_DIR
from bpd.draw import draw_exponential
from bpd.io import save_dataset
from bpd.likelihood import gaussian_image_loglikelihood
from bpd.pipelines import pipeline_interim_samples_one_galaxy
from bpd.prior import interim_gprops_logprior
from bpd.sample import (
    get_target_images,
    get_true_params_from_galaxy_params,
    sample_galaxy_params_skew,
)
from bpd.utils import DEFAULT_HYPERPARAMS, MAX_N_GALS_PER_GPU


def _init_function(key: PRNGKeyArray, *, data: Array, true_params: dict):
    image = data
    assert image.ndim == 2
    assert image.shape[0] == image.shape[1]
    k1, k2 = random.split(key)

    te1 = true_params["e1"]
    e1 = random.uniform(k1, shape=(), minval=te1 - 0.1, maxval=te1 + 0.1)

    te2 = true_params["e2"]
    e2 = random.uniform(k2, shape=(), minval=te2 - 0.1, maxval=te2 + 0.1)
    return {
        "e1": e1,
        "e2": e2,
    }


def main(
    seed: int,
    tag: str = typer.Option(),
    mode: str = "",
    n_gals: int = 2000,
    n_samples_per_gal: int = 300,
    sigma_e_int: float = 0.3,
    g1: float = 0.02,
    g2: float = 0.0,
    slen: int = 63,
    fft_size: int = 256,
    background: float = 1.0,
    initial_step_size: float = 0.01,
):
    assert (g1 > 0 and mode == "plus") or (g1 < 0 and mode == "minus") or (not mode)

    rng_key = random.key(seed)
    pkey, nkey, gkey = random.split(rng_key, 3)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)
    extra_tag = f"_{mode}" if mode else ""
    out_fpath = dirpath / f"interim_samples_{seed}{extra_tag}.npz"

    # galaxy parameters from prior
    galaxy_params = sample_galaxy_params_skew(
        pkey, n=n_gals, g1=g1, g2=g2, **DEFAULT_HYPERPARAMS
    )
    assert galaxy_params["x"].shape == (n_gals,)
    assert galaxy_params["e1"].shape == (n_gals,)

    # now get corresponding target images
    draw_params = {**galaxy_params}
    draw_params["f"] = 10 ** draw_params.pop("lf")
    draw_params["hlr"] = 10 ** draw_params.pop("lhlr")
    target_images = get_target_images(
        nkey, draw_params, background=background, slen=slen, draw_type="exponential"
    )
    assert target_images.shape == (n_gals, slen, slen)

    # interim samples are on 'sheared ellipticity'
    true_params = vmap(get_true_params_from_galaxy_params)(galaxy_params)
    fixed_params = {
        "x": true_params.pop("x"),
        "y": true_params.pop("y"),
        "f": 10 ** true_params.pop("lf"),
        "hlr": 10 ** true_params.pop("lhlr"),
    }

    # setup prior and likelihood
    _logprior = partial(
        interim_gprops_logprior,
        sigma_e=sigma_e_int,
        free_flux_hlr=False,
        free_dxdy=False,
    )
    _draw_fnc = partial(draw_exponential, slen=slen, fft_size=fft_size)
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
        initialization_fnc=_init_function,
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

    print("Pipeline compiled, now running on all galaxies...")

    # divide in batches if not all galaxies fit on GPU
    n_batches = math.ceil(n_gals / MAX_N_GALS_PER_GPU)
    samples_list = []
    for ii in range(n_batches):
        start, end = ii * MAX_N_GALS_PER_GPU, (ii + 1) * MAX_N_GALS_PER_GPU
        print(f"Batch {ii}, {start}:{end}")
        _gkeys = gkeys[start:end]
        _target_images = target_images[start:end]
        _fixed_params = {k: v[start:end] for k, v in fixed_params.items()}
        _true_params = {k: v[start:end] for k, v in true_params.items()}
        _samples = vpipe(_gkeys, _target_images, _fixed_params, _true_params)
        samples_list.append(_samples)

    # concatenate all samples
    samples = {}
    for k in samples_list[0]:
        samples[k] = jnp.concatenate(
            [samples_list[ii][k] for ii in range(n_batches)], axis=0
        )
    assert samples["e1"].shape == (n_gals, n_samples_per_gal)

    save_dataset(
        {
            "samples": {
                "e1": samples["e1"],
                "e2": samples["e2"],
            },
            "truth": {
                "e1": galaxy_params["e1"],
                "e2": galaxy_params["e2"],
                "lf": galaxy_params["lf"],
                "lhlr": galaxy_params["lhlr"],
                "x": galaxy_params["x"],
                "y": galaxy_params["y"],
            },
            "hyper": {
                "g1": g1,
                "g2": g2,
                "sigma_e_int": sigma_e_int,
                **DEFAULT_HYPERPARAMS,
            },
        },
        out_fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
