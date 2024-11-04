#!/usr/bin/env python3
from functools import partial
from math import ceil

import jax.numpy as jnp
import typer
from jax import random, vmap

from bpd import DATA_DIR
from bpd.pipelines.shear_inference import pipeline_shear_inference
from bpd.pipelines.toy_ellips import pipeline_toy_ellips_samples


def main(
    tag: str,
    seed: int,
    n_exps: int,
    shape_noise: float = 1e-3,
    obs_noise: float = 1e-4,
    sigma_e_int: float = 3e-2,
    initial_shear_step_size: float = 1e-3,
    n_vec: int = 50,
    g1: float = 0.02,
    g2: float = 0.0,
    n_gals: int = 1000,
    n_samples_shear: int = 3000,
    n_samples_per_gal: int = 1000,
    trim: int = 10,
):
    key0 = random.key(seed)
    _keys = random.split(key0, n_exps * 2)  # one for toy ellipticities, one for shear
    keys = _keys.reshape(n_exps, 2)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag

    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)

    n_batch = ceil(len(keys) / n_vec)

    pipe1 = partial(
        pipeline_toy_ellips_samples,
        g1=g1,
        g2=g2,
        sigma_e=shape_noise,
        sigma_e_int=sigma_e_int,
        sigma_m=obs_noise,
        n_gals=n_gals,
        n_samples_per_gal=n_samples_per_gal,
    )
    pipe2 = partial(
        pipeline_shear_inference,
        true_g=jnp.array([g1, g2]),
        sigma_e=shape_noise,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples_shear,
        initial_step_size=initial_shear_step_size,
    )
    vpipe1 = vmap(pipe1, in_axes=(0,))
    vpipe2 = vmap(pipe2, in_axes=(0, 0))

    for ii in range(n_batch):
        print(f"batch: {ii}")
        bkeys = keys[ii * n_vec : (ii + 1) * n_vec]

        ekeys = bkeys[:, 0]
        skeys = bkeys[:, 1]

        e_post, _, _ = vpipe1(ekeys)
        e_post_trimmed = e_post[:, :, ::trim, :]

        g_samples = vpipe2(skeys, e_post_trimmed)

        fpath_ellip = dirpath / f"e_post_{seed}_{ii}.npy"
        fpath_shear = dirpath / f"g_samples_{seed}_{ii}.npy"

        assert not fpath_shear.exists()
        jnp.save(fpath_ellip, e_post)
        jnp.save(fpath_shear, g_samples)


if __name__ == "__main__":
    typer.run(main)
