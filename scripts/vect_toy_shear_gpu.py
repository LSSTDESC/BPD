#!/usr/bin/env python3
from functools import partial
from math import ceil

import click
import jax.numpy as jnp
from jax import random, vmap

from bpd import DATA_DIR
from bpd.pipelines.shear_inference import pipeline_shear_inference
from bpd.pipelines.toy_ellips import pipeline_toy_ellips_samples


@click.command()
@click.option("--n-vec", type=int, default=50, help="# shear chains in parallel")
@click.option("--tag", type=str, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--n-seeds", type=int, required=True)
@click.option("--g1", type=float, default=0.02)
@click.option("--g2", type=float, default=0.0)
@click.option("--n-samples-gals", type=int, default=1000, help="# of gals")
@click.option("--n-samples-shear", type=int, default=3000, help="shear samples")
@click.option("--k", type=int, default=1000, help="# int. post. samples galaxy.")
@click.option("--shape-noise", type=float, default=1e-3)
@click.option("--sigma-e-int", type=float, default=1e-2)
@click.option("--obs-noise", type=float, default=1e-4)
@click.option("--trim", type=int, default=10)
def main(
    n_vec: int,
    tag: str,
    seed: int,
    n_seeds: int,
    g1: float,
    g2: float,
    n_samples_gals: int,
    n_samples_shear: int,
    k: int,
    shape_noise: float,
    sigma_e_int: float,
    obs_noise: float,
    trim: int,
):
    key0 = random.key(seed)
    _keys = random.split(key0, n_seeds * 2)  # one for toy ellipticities, one for shear
    keys = _keys.reshape(n_seeds, 2)

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
        n_samples=n_samples_gals,
        k=k,
    )
    pipe2 = partial(
        pipeline_shear_inference,
        true_g=jnp.array([g1, g2]),
        sigma_e=shape_noise,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples_shear,
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
    main()
