#!/usr/bin/env python3

import math
from functools import partial

import jax
import jax.numpy as jnp
import typer
from jax import jit, random, vmap

from bpd import DATA_DIR
from bpd.io import load_dataset_jax, save_dataset
from bpd.pipelines import pipeline_shear_inference_simple
from bpd.utils import process_in_batches


def main(
    seed: int,
    tag: str = typer.Option(),
    plus_samples_fname: str = typer.Option(),
    minus_samples_fname: str = typer.Option(),
    initial_step_size: float = 0.01,
    n_gals: int | None = None,
    n_splits: int = 500,
    n_batches: int = 5,
):
    rng_key = jax.random.key(seed)

    dirpath = DATA_DIR / "cache_chains" / tag
    pfpath = DATA_DIR / "cache_chains" / tag / plus_samples_fname
    mfpath = DATA_DIR / "cache_chains" / tag / minus_samples_fname
    fpath = DATA_DIR / "cache_chains" / tag / f"g_samples_{seed}_errs.npz"

    assert dirpath.exists()
    assert pfpath.exists(), "ellipticity samples file does not exist"
    assert mfpath.exists(), "ellipticity samples file does not exist"

    dsp = load_dataset_jax(pfpath)
    if n_gals is None:
        n_gals = dsp["samples"]["e1"].shape[0]
    e1 = dsp["samples"]["e1"][:n_gals]
    e2 = dsp["samples"]["e2"][:n_gals]
    e1e2p = jnp.stack([e1, e2], axis=-1)
    sigma_e = dsp["hyper"]["shape_noise"]
    sigma_e_int = dsp["hyper"]["sigma_e_int"]

    dsm = load_dataset_jax(mfpath)
    e1 = dsm["samples"]["e1"][:n_gals]
    e2 = dsm["samples"]["e2"][:n_gals]
    e1e2m = jnp.stack([e1, e2], axis=-1)
    assert e1e2p.shape == e1e2m.shape, "ellipticity samples do not match"
    assert sigma_e == dsm["hyper"]["shape_noise"], "shape noise does not match"
    assert sigma_e_int == dsm["hyper"]["sigma_e_int"], "shape noise does not match"

    split_size = e1e2p.shape[0] // n_splits
    assert split_size * n_splits == e1e2p.shape[0], "dimensions do not match"
    # Reshape ellipticity samples
    e1e2ps = jnp.reshape(e1e2p, (n_splits, split_size, -1, 2))
    e1e2ms = jnp.reshape(e1e2m, (n_splits, split_size, -1, 2))

    # get shear inference pipeline
    _pipe = partial(
        pipeline_shear_inference_simple,
        init_g=jnp.array([0.0, 0.0]),
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        n_samples=1000,
        initial_step_size=initial_step_size,
    )
    _pipe = jit(_pipe)
    pipe = vmap(_pipe, in_axes=(0, 0))

    # run shear inference pipeline in batches
    keys = random.split(rng_key, n_splits)
    g_plus = process_in_batches(
        pipe, keys, e1e2ps, n_points=n_splits, n_batches=n_batches
    )
    g_minus = process_in_batches(
        pipe, keys, e1e2ms, n_points=n_splits, n_batches=n_batches
    )
    assert g_plus.shape == g_minus.shape, "shear samples do not match"
    assert g_plus.shape == (n_splits, 1000, 2), "shear samples do not match"

    save_dataset(
        {
            "g_plus": g_plus,
            "g_minus": g_minus,
            "sigma_e": sigma_e,
            "sigma_e_int": sigma_e_int,
        },
        fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
