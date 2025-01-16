#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

import jax
import jax.numpy as jnp
import typer

from bpd import DATA_DIR
from bpd.io import load_dataset
from bpd.pipelines import pipeline_shear_inference_simple


def main(
    seed: int,
    old_seed: int = typer.Option(),
    interim_samples_fname: str = typer.Option(),
    tag: str = typer.Option(),
    initial_step_size: float = 1e-3,
    n_samples: int = 3000,
    trim: int = 1,
    overwrite: bool = False,
):
    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    assert dirpath.exists()
    interim_samples_fpath = DATA_DIR / "cache_chains" / tag / interim_samples_fname
    assert interim_samples_fpath.exists(), "ellipticity samples file does not exist"
    fpath = DATA_DIR / "cache_chains" / tag / f"g_samples_{old_seed}_{seed}.npy"

    if fpath.exists() and not overwrite:
        raise IOError("overwriting...")

    samples_dataset = load_dataset(interim_samples_fpath)
    e_post = samples_dataset["e_post"][:, ::trim, :]
    true_g = samples_dataset["true_g"]
    sigma_e = samples_dataset["sigma_e"]
    sigma_e_int = samples_dataset["sigma_e_int"]

    rng_key = jax.random.key(seed)
    g_samples = pipeline_shear_inference_simple(
        rng_key,
        e_post,
        init_g=true_g,
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )

    jnp.save(fpath, g_samples)


if __name__ == "__main__":
    typer.run(main)
