#!/usr/bin/env python3
from functools import partial
from math import ceil

import jax.numpy as jnp
import typer
from jax import jit as jjit
from jax import random, vmap

from bpd import DATA_DIR
from bpd.draw import draw_gaussian
from bpd.initialization import init_with_truth
from bpd.io import save_dataset
from bpd.pipelines.image_samples import (
    get_target_images,
    get_true_params_from_galaxy_params,
    pipeline_interim_samples_one_galaxy,
)
from bpd.pipelines.image_samples_fixed_flux import (
    get_target_galaxy_params_simple,
    logprior,
)

INIT_FNC = init_with_truth


def main(
    seed: int,
    tag: str,
    n_gals: int = 100,  # technically, in this file it means 'noise realizations'
    n_samples_per_gal: int = 100,
    n_vec: int = 50,  # how many galaxies to process simultaneously in 1 GPU core
    g1: float = 0.02,
    g2: float = 0.0,
    lf: float = 6.0,
    hlr: float = 1.0,
    shape_noise: float = 1e-3,
    sigma_e_int: float = 3e-2,
    slen: int = 53,
    fft_size: int = 256,
    background: float = 1.0,
    initial_step_size: float = 1e-3,
):
    rng_key = random.key(seed)
    pkey, nkey, gkey = random.split(rng_key, 3)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)

    # galaxy galaxy parameters
    pkeys = random.split(pkey, n_gals)
    _get_galaxy_params = partial(
        get_target_galaxy_params_simple, g1=g1, g2=g2, shape_noise=shape_noise
    )
    galaxy_params = vmap(_get_galaxy_params)(pkeys)
    assert galaxy_params["x"].shape == (n_gals,)
    assert "lf" not in galaxy_params and "hlr" not in galaxy_params  # not inferring

    # now get corresponding target images
    # we use the same flux and hlr for every galaxy in this experiment (and fix them in sampling)
    extra_params = {"f": 10 ** jnp.full((n_gals,), lf), "hlr": jnp.full((n_gals,), hlr)}
    draw_params = {**galaxy_params, **extra_params}
    target_images = get_target_images(
        nkey, draw_params, background=background, slen=slen
    )
    assert target_images.shape == (n_gals, slen, slen)

    # finally, interim samples are on 'sheared ellipticity'
    true_params = vmap(get_true_params_from_galaxy_params)(galaxy_params)

    # prepare pipelines
    gkeys = random.split(gkey, n_gals)
    _draw_fnc = partial(draw_gaussian, f=10**lf, hlr=hlr, slen=slen, fft_size=fft_size)
    pipe = partial(
        pipeline_interim_samples_one_galaxy,
        initialization_fnc=INIT_FNC,
        draw_fnc=_draw_fnc,
        logprior=logprior,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples_per_gal,
        initial_step_size=initial_step_size,
        slen=slen,
        fft_size=fft_size,
        background=background,
    )
    vpipe = vmap(jjit(pipe), (0, 0, 0))

    # compilation on single target image
    _ = vpipe(
        gkeys[0, None],
        {k: v[0, None] for k, v in true_params.items()},
        target_images[0, None],
    )

    # run in batches
    n_batch = ceil(n_gals / n_vec)
    samples = []

    for ii in range(n_batch):
        start, stop = ii * n_vec, (ii + 1) * n_vec

        # slice
        _bkeys = gkeys[start:stop]
        _btparams = {k: v[start:stop] for k, v in true_params.items()}
        _bimages = target_images[start:stop]

        # run
        _samples = vpipe(_bkeys, _btparams, _bimages)
        samples.append(_samples)

    # get e_post
    e_post = jnp.concatenate([jnp.stack([s["e1"], s["e2"]], axis=-1) for s in samples])
    xy = jnp.concatenate([jnp.stack([s["x"], s["y"]], axis=-1) for s in samples])
    fpath = dirpath / f"e_post_{seed}.npz"

    save_dataset(
        {
            "e_post": e_post,
            "xy": xy,
            "true_g": jnp.array([g1, g2]),
            "sigma_e": shape_noise,
            "sigma_e_int": sigma_e_int,
        },
        fpath,
    )


if __name__ == "__main__":
    typer.run(main)