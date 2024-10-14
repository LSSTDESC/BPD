#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from pathlib import Path

import click
import jax
import jax.numpy as jnp

from bpd import DATA_DIR
from bpd.io import load_dataset
from bpd.pipelines.shear_inference import pipeline_shear_inference


def _extract_seed(fpath: str) -> int:
    name = Path(fpath).name
    first = name.find("_")
    second = name.find("_", first + 1)
    third = name.find(".")
    return int(name[second + 1 : third])


@click.command()
@click.option("--tag", type=str, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--e-samples-fname", type=str, required=True)
@click.option("-n", "--n-samples", type=int, default=2000, help="# of shear samples")
@click.option("--trim", type=int, default=1, help="trimming makes like. eval. fast")
@click.option("--overwrite", type=bool, default=False)
def main(
    tag: str,
    seed: int,
    e_samples_fname: str,
    n_samples: int,
    trim: int,
    overwrite: bool,
):

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    assert dirpath.exists()
    e_samples_fpath = DATA_DIR / "cache_chains" / tag / e_samples_fname
    assert e_samples_fpath.exists(), "ellipticity samples file does not exist"
    old_seed = _extract_seed(e_samples_fname)
    fpath = DATA_DIR / "cache_chains" / tag / f"g_samples_{old_seed}_{seed}.npy"

    if fpath.exists():
        if not overwrite:
            raise IOError("overwriting...")

    samples_dataset = load_dataset(e_samples_fpath)
    e_post = samples_dataset["e_post"][:, ::trim, :]
    true_g = samples_dataset["true_g"]
    sigma_e = samples_dataset["sigma_e"]

    rng_key = jax.random.key(seed)
    g_samples = pipeline_shear_inference(
        rng_key, e_post, true_g, sigma_e, sigma_e_int=sigma_e * 2, n_samples=n_samples
    )

    jnp.save(fpath, g_samples)


if __name__ == "__main__":
    main()
