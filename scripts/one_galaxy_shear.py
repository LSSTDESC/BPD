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
    pipeline_image_interim_samples,
)
from bpd.pipelines.shear_inference import pipeline_shear_inference

init_fnc = init_with_truth


def main(
    tag: str,
    seed: int,
    g1: float = 0.02,
    g2: float = 0.0,
    lf: float = 6.0,
    shape_noise: float = 1e-3,
    sigma_e_int: float = 3e-2,
    slen: int = 53,
    fft_size: int = 256,
    background: float = 1.0,
    n_gals: int = 1000,  # technically, here it means 'noise realizations'
    n_samples_shear: int = 3000,
    n_samples_per_gal: int = 100,
    trim: int = 1,
):
    rng_key = random.key(seed)
    pkey, nkey, gkey, skey = random.split(rng_key, 3)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag

    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)

    # get images
    galaxy_params = get_target_galaxy_params_simple(
        pkey, lf=lf, g1=g1, g2=g2, hlr=hlr, shape_noise=shape_noise
    )

    target_images = get_target_images_single(
        nkey,
        n_samples=n_gals,
        single_galaxy_params=galaxy_params,
        background=background,
        slen=slen,
    )

    true_params = get_true_params_from_galaxy_params(galaxy_params)

    # prepare pipelines
    pipe1 = partial(
        pipeline_image_interim_samples,
        initialization_fnc=init_fnc,
        n_samples=k,
        max_num_doublings=5,
        initial_step_size=1e-3,
        n_warmup_steps=500,
        is_mass_matrix_diagonal=True,
        background=background,
        slen=slen,
        pixel_scale=pixel_scale,
        fft_size=fft_size,
    )
    vpipe1 = vmap(jjit(pipe1), (0, None, 0))

    pipe2 = partial(
        pipeline_shear_inference,
        true_g=jnp.array([g1, g2]),
        sigma_e=shape_noise,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples_shear,
    )
    vpipe2 = vmap(pipe2, in_axes=(0, 0))

    gkeys = random.split(gkey, n_gals)
    galaxy_samples = vpipe1(gkeys, true_params, target_images)

    e_post = jnp.stack([galaxy_samples["e1"], galaxy_samples["e2"]], axis=-1)

    g_samples = vpipe2(skey, e_post)


def main():
    pipe1 = partial(
        pipeline_toy_ellips_samples,
        g1=g1,
        g2=g2,
        sigma_e=shape_noise,
        sigma_e_int=sigma_e_int,
        sigma_m=obs_noise,
        n_samples=n_samples_gals,
        k=k,
    )

    vpipe1 = vmap(pipe1, in_axes=(0,))

    for ii in range(n_batch):
        print(f"batch: {ii}")
        bkeys = keys[ii * n_vec : (ii + 1) * n_vec]

        ekeys = bkeys[:, 0]
        skeys = bkeys[:, 1]

        e_post, _, _ = vpipe1(ekeys)
        e_post_trimmed = e_post[:, :, ::trim, :]

        fpath_ellip = dirpath / f"e_post_{seed}_{ii}.npy"
        fpath_shear = dirpath / f"g_samples_{seed}_{ii}.npy"

        assert not fpath_shear.exists()
        jnp.save(fpath_ellip, e_post)
        jnp.save(fpath_shear, g_samples)


if __name__ == "__main__":
    typer.run(main)
