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


def main(
    seed: int,
    tag: str = typer.Option(),
    plus_samples_fname: str = typer.Option(),
    minus_samples_fname: str = typer.Option(),
    initial_step_size: float = 0.01,
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
    e1 = dsp["samples"]["e1"]
    e2 = dsp["samples"]["e2"]
    e1e2p = jnp.stack([e1, e2], axis=-1)
    sigma_e = dsp["hyper"]["shape_noise"]
    sigma_e_int = dsp["hyper"]["sigma_e_int"]

    dsm = load_dataset_jax(mfpath)
    e1 = dsm["samples"]["e1"]
    e2 = dsm["samples"]["e2"]
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

    # run shear inference pipeline
    keys = random.split(rng_key, n_splits)
    batch_size = math.ceil(n_splits / n_batches)
    for ii in range(n_batches):
        start = ii * batch_size
        end = (ii + 1) * batch_size
        print(start, end)
        _keys = keys[start:end]
        _e1e2ps = e1e2ps[start:end]
        _e1e2ms = e1e2ms[start:end]
        print(f"Running shear inference pipeline (plus) batch {ii + 1}/{n_batches}...")
        gp = pipe(_keys, _e1e2ps)
        print(f"Running shear inference pipeline (minus) batch {ii + 1}/{n_batches}...")
        gm = pipe(_keys, _e1e2ms)

        print(gp.shape)
        assert gp.shape == gm.shape, "shear samples do not match"
        assert gp.shape[1:] == (1000, 2), "shear samples do not match"
        if ii == 0:
            g_plus = gp
            g_minus = gm
        else:
            g_plus = jnp.concatenate((g_plus, gp), axis=0)
            g_minus = jnp.concatenate((g_minus, gm), axis=0)
        print(g_plus.shape)

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
