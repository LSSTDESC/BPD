#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from pathlib import Path

import jax
import jax.numpy as jnp
import typer

from bpd import DATA_DIR
from bpd.io import load_dataset
from bpd.pipelines.shear_inference import pipeline_shear_inference_ellipticities


def _extract_seed(fpath: str) -> int:
    name = Path(fpath).name
    first = name.find("_")
    second = name.find("_", first + 1)
    third = name.find(".")
    return int(name[second + 1 : third])


def main(
    seed: int,
    tag: str,
    interim_samples_fname: str,
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
    old_seed = _extract_seed(interim_samples_fpath)
    fpath = DATA_DIR / "cache_chains" / tag / f"g_samples_{old_seed}_{seed}.npy"

    if fpath.exists() and not overwrite:
        raise IOError("overwriting...")

    samples_dataset = load_dataset(interim_samples_fpath)
    e_post = samples_dataset["e_post"][:, ::trim, :]
    true_g = samples_dataset["true_g"]
    sigma_e = samples_dataset["sigma_e"]
    sigma_e_int = samples_dataset["sigma_e_int"]

    rng_key = jax.random.key(seed)
    g_samples = pipeline_shear_inference_ellipticities(
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
