#!/usr/bin/env python3

import multiprocessing as mp
import subprocess
from functools import partial

import click
import numpy as np


def task_interim_posterior(seed: int, tag: str):
    subprocess.call(f"./get_toy_ellip_samples.py --tag {tag} --seed {seed}", shell=True)


def task_shear(seed: int, tag: str):
    e_samples_file = (
        f"/pscratch/sd/i/imendoza/data/cache_chains/e_post_{seed}_{tag}.npz"
    )
    subprocess.call(
        f"./get_shear_from_post_ellips.py --seed {seed} --e-samples-file {e_samples_file}",
        shell=True,
    )


@click.command()
@click.option("--tag", type=str, required=True)
@click.option("--n-seeds", type=int, required=True)
def main(tag: str, n_seeds):
    seeds = np.arange(1, n_seeds + 1, 1)

    n_processes = min(mp.cpu_count() - 3, n_seeds)
    print(f"n_processes: {n_processes}")
    pool = mp.Pool(processes=n_processes)

    task1 = partial(task_interim_posterior, tag=tag)
    task2 = partial(task_shear, tag=tag)

    pool.map(task1, seeds)
    pool.map(task2, seeds)


if __name__ == "__main__":
    main()
