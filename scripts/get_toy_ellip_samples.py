#!/usr/bin/env python3
"""This file creates artificial samples of ellipticities and saves them to .npz file."""

import click
import jax
import jax.numpy as jnp

from bpd import DATA_DIR
from bpd.io import save_dataset
from bpd.pipelines.toy_ellips import pipeline_toy_ellips_samples


@click.command()
@click.option("--tag", type=str, required=True)
@click.option("--seed", type=int, default=42)
@click.option("--g1", type=float, default=0.02)
@click.option("--g2", type=float, default=0.0)
@click.option("-n", "--n-samples", type=int, default=10_000, help="# of gals")
@click.option("--k", type=int, default=10, help="# int. posterior samples per galaxy.")
@click.option("--shape-noise", type=float, default=1e-3)  # > OK, 1e-4 not OK :(
@click.option("--obs-noise", type=float, default=1e-4)
@click.option("--overwrite", type=bool, default=False)
def main(
    tag: str,
    seed: int,
    g1: float,
    g2: float,
    n_samples: int,
    k: int,
    shape_noise: float,
    obs_noise: float,
    overwrite: bool,
):
    dirpath = DATA_DIR / "cache_chains" / tag
    fpath = DATA_DIR / "cache_chains" / dirpath / f"e_post_{seed}.npz"
    if not dirpath.exists():
        dirpath.mkdir(exist_ok=False)
    if fpath.exists():
        assert overwrite

    key = jax.random.key(seed)
    e_post, e_obs, e_sheared = pipeline_toy_ellips_samples(
        key,
        g1,
        g2,
        sigma_e=shape_noise,
        sigma_m=obs_noise,
        n_samples=n_samples,
        k=k,
        sigma_e_int=2 * shape_noise,
    )

    ds = {
        "e_post": e_post,
        "e_obs": e_obs,
        "e_sheared": e_sheared,
        "true_g": jnp.array([g1, g2]),
        "sigma_e": jnp.array(shape_noise),
        "sigma_m": jnp.array(obs_noise),
        "seed": jnp.array(seed),
    }
    save_dataset(ds, fpath, overwrite=overwrite)


if __name__ == "__main__":
    main()
