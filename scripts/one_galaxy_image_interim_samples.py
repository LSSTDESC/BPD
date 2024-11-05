#!/usr/bin/env python3
from functools import partial

import jax.numpy as jnp
import typer
from jax import jit as jjit
from jax import random, vmap

from bpd import DATA_DIR
from bpd.initialization import init_with_truth
from bpd.pipelines.image_ellips import (
    get_target_galaxy_params_simple,
    get_target_images_single,
    get_true_params_from_galaxy_params,
    pipeline_image_interim_samples_one_galaxy,
)

init_fnc = init_with_truth


def main(
    tag: str,
    seed: int,
    n_gals: int = 100,  # technically, in this file it means 'noise realizations'
    n_samples_per_gal: int = 100,
    n_vec: int = 50,  # how many galaxies to process simultaneously in 1 GPU core
    g1: float = 0.02,
    g2: float = 0.0,
    lf: float = 6.0,
    shape_noise: float = 1e-3,
    sigma_e_int: float = 3e-2,
    slen: int = 53,
    fft_size: int = 256,
    background: float = 1.0,
    initial_step_size: float = 1e-3,
    trim: int = 1,
):
    rng_key = random.key(seed)
    pkey, nkey, gkey = random.split(rng_key, 3)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag

    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)

    fpath = dirpath / f"e_post_{seed}.npy"

    # get images
    galaxy_params = get_target_galaxy_params_simple(  # default hlr, x, y
        pkey, lf=lf, g1=g1, g2=g2, shape_noise=shape_noise
    )

    draw_params = {**galaxy_params}
    draw_params["f"] = 10 ** draw_params.pop("lf")
    target_images, _ = get_target_images_single(
        nkey,
        n_samples=n_gals,
        single_galaxy_params=draw_params,
        background=background,
        slen=slen,
    )
    assert target_images.shape == (n_gals, slen, slen)

    true_params = get_true_params_from_galaxy_params(galaxy_params)

    # prepare pipelines
    pipe1 = partial(
        pipeline_image_interim_samples_one_galaxy,
        initialization_fnc=init_fnc,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples_per_gal,
        initial_step_size=initial_step_size,
        slen=slen,
        fft_size=fft_size,
        background=background,
    )
    vpipe1 = vmap(jjit(pipe1), (0, None, 0))

    # initialization
    gkey1, gkey2 = random.split(gkey, 2)
    gkeys1 = random.split(gkey1, n_gals)
    gkeys2 = random.split(gkey2, n_gals)
    init_positions = vmap(init_fnc, (0, None))(gkeys1, true_params)

    galaxy_samples = vpipe1(gkeys2, true_params, target_images)

    e_post = jnp.stack([galaxy_samples["e1"], galaxy_samples["e2"]], axis=-1)

    jnp.save()


if __name__ == "__main__":
    typer.run(main)
