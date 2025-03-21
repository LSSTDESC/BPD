#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

import jax
import jax.numpy as jnp
import numpy as np
import typer

from bpd import DATA_DIR
from bpd.io import load_dataset_jax
from bpd.pipelines import pipeline_shear_inference_simple


def main(
    seed: int,
    interim_samples_fname: str = typer.Option(),
    tag: str = typer.Option(),
    initial_step_size: float = 1e-3,
    n_samples: int = 3000,
    overwrite: bool = False,
    extra_tag: str = "",
):
    extra_txt = f"_{extra_tag}" if extra_tag else ""
    dirpath = DATA_DIR / "cache_chains" / tag
    interim_samples_fpath = DATA_DIR / "cache_chains" / tag / interim_samples_fname
    fpath = DATA_DIR / "cache_chains" / tag / f"g_samples_{seed}{extra_txt}.npy"

    assert dirpath.exists()
    assert interim_samples_fpath.exists(), "ellipticity samples file does not exist"
    if fpath.exists() and not overwrite:
        raise IOError("overwriting...")

    ds = load_dataset_jax(interim_samples_fpath)
    e1 = ds["samples"]["e1"]
    e2 = ds["samples"]["e2"]
    e1e2 = jnp.stack([e1, e2], axis=-1)
    sigma_e = ds["hyper"]["shape_noise"]
    sigma_e_int = ds["hyper"]["sigma_e_int"]

    rng_key = jax.random.key(seed)
    g_samples = pipeline_shear_inference_simple(
        rng_key,
        e1e2=e1e2,
        init_g=jnp.array([0.0, 0.0]),
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )

    np.save(fpath, np.asarray(g_samples))


if __name__ == "__main__":
    typer.run(main)
