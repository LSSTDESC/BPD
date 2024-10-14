#!/usr/bin/env python3

import multiprocessing as mp
from functools import partial

import click
import jax.numpy as jnp
import numpy as np

from bpd import DATA_DIR
from bpd.pipelines.shear_inference import pipeline_shear_inference
from bpd.pipelines.toy_ellips import pipeline_toy_ellips_samples


@click.command()
@click.option("--tag", type=str, required=True)
@click.option("--start-seed", type=int, required=True)
@click.option("--end-seed", type=int, required=True)
@click.option("--g1", type=float, default=0.02)
@click.option("--g2", type=float, default=0)
@click.option("--n-samples-gals", type=int, default=10_000, help="# of gals")
@click.option("--n-samples-shear", type=int, default=3000, help="shear samples")
@click.option("--k", type=int, default=10, help="# int. posterior samples per galaxy.")
@click.option("--shape-noise", type=float, default=1e-3)
@click.option("--obs-noise", type=float, default=1e-4)
@click.option("--overwrite", type=bool, default=False)
def main(
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
    overwrite: bool,
):
    n_seeds = end_seed - start_seed + 1
    seeds = np.arange(start_seed, end_seed + 1, 1)

    n_processes = min(mp.cpu_count() - 3, n_seeds)
    print(f"n_processes: {n_processes}")
    pool = mp.Pool(processes=n_processes)

    task1 = partial(
        pipeline_toy_ellips_samples,
        g1=g1,
        g2=g2,
        sigma_e=shape_noise,
        sigma_m=obs_noise,
        n_samples=n_samples_gals,
        k=k,
        sigma_e_int=shape_noise * 2,
    )
    results1 = pool.map(task1, seeds)
    print("INFO: Ellipticity samples obtained")

    e_post_list = [res[0] for res in results1]  # only want first element of tuple
    del results1  # clear memory

    task2 = partial(
        pipeline_shear_inference,
        true_g=jnp.array([g1, g2]),
        sigma_e=shape_noise,
        n_samples=n_samples_shear,
    )
    g_samples_list = pool.starmap(task2, zip(seeds, e_post_list))
    print("INFO: Shear samples obtained")

    print("INFO: Saving shear samples to disk...")
    g_samples = jnp.stack(g_samples_list)
    fpath = DATA_DIR / "cache_chains" / f"g_samples_{tag}.npy"
    jnp.save(fpath, g_samples)


if __name__ == "__main__":
    main()
