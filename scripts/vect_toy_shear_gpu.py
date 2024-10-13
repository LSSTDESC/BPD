#!/usr/bin/env python3
from functools import partial
from math import ceil

import click
import jax.numpy as jnp
from get_shear_from_post_ellips import pipeline_shear_inference
from get_toy_ellip_samples import pipeline_toy_ellips_samples
from jax import vmap

from bpd import DATA_DIR


@click.command()
@click.option(
    "--n-vec", type=int, default=100, help="# shear chains to run in parallel"
)
@click.option("--tag", type=str, required=True)
@click.option("--start-seed", type=int, required=True)
@click.option("--end-seed", type=int, required=True)
@click.option("--g1", type=float, default=0.02)
@click.option("--g2", type=float, default=0.0)
@click.option("--n-samples-gals", type=int, default=1000, help="# of gals")
@click.option("--n-samples-shear", type=int, default=3000, help="shear samples")
@click.option("--k", type=int, default=100, help="# int. posterior samples per galaxy.")
@click.option("--shape-noise", type=float, default=1e-3)
@click.option("--obs-noise", type=float, default=1e-4)
@click.option("--trim", type=int, default=1)
def main(
    n_vec: int,
    tag: str,
    start_seed: int,
    end_seed: int,
    g1: float,
    g2: float,
    n_samples_gals: int,
    n_samples_shear: int,
    k: int,
    shape_noise: float,
    obs_noise: float,
    trim: int,
):
    seeds = jnp.arange(start_seed, end_seed + 1, 1)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag

    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)

    n_batch = ceil(len(seeds) / n_vec)

    pipe1 = partial(
        pipeline_toy_ellips_samples,
        g1=g1,
        g2=g2,
        sigma_e=shape_noise,
        sigma_m=obs_noise,
        n_samples=n_samples_gals,
        k=k,
    )
    pipe2 = partial(
        pipeline_shear_inference,
        true_g=jnp.array([g1, g2]),
        sigma_e=shape_noise,
        n_samples=n_samples_shear,
    )
    vpipe1 = vmap(pipe1, in_axes=(0,))
    vpipe2 = vmap(pipe2, in_axes=(0, 0))

    for ii in range(n_batch):
        print(f"batch: {ii}")
        b_seeds = seeds[ii * n_vec : (ii + 1) * n_vec]
        e_post, _, _ = vpipe1(b_seeds)
        e_post_trimmed = e_post[:, :, ::trim, :]

        g_samples = vpipe2(b_seeds, e_post_trimmed)

        fpath_ellip = dirpath / f"e_post_{b_seeds[0]}_{b_seeds[-1]}.npy"
        fpath_shear = dirpath / f"g_samples_{b_seeds[0]}_{b_seeds[-1]}.npy"

        assert not fpath_shear.exists()
        jnp.save(fpath_ellip, e_post)
        jnp.save(fpath_shear, g_samples)


if __name__ == "__main__":
    main()
