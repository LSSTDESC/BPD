#!/usr/bin/env python3

import math
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import typer
from jax import jit, random, vmap
from jax.tree_util import tree_map

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import load_dataset_jax, save_dataset
from bpd.pipelines import logtarget_shear_and_sn


def main(
    seed: int,
    tag: str = typer.Option(),
    plus_samples_fpath: str = typer.Option(),
    minus_samples_fpath: str = typer.Option(),
    initial_step_size: float = 0.01,
    n_splits: int = 500,
    n_batches: int = 5,
):
    rng_key = jax.random.key(seed)

    dirpath = DATA_DIR / "cache_chains" / tag
    pfpath = Path(plus_samples_fpath)
    mfpath = Path(minus_samples_fpath)
    fpath = dirpath / f"g_samples_{seed}_errs.npz"

    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)
    assert pfpath.exists(), "ellipticity samples file does not exist"
    assert mfpath.exists(), "ellipticity samples file does not exist"

    dsp = load_dataset_jax(pfpath)
    dsm = load_dataset_jax(mfpath)

    e1p = dsp["samples"]["e1"]
    e2p = dsp["samples"]["e2"]
    e1e2p = jnp.stack([e1p, e2p], axis=-1)

    e1m = dsm["samples"]["e1"]
    e2m = dsm["samples"]["e2"]
    e1e2m = jnp.stack([e1m, e2m], axis=-1)

    sigma_e_int = dsp["hyper"]["sigma_e_int"]
    assert dsp["hyper"]["shape_noise"] == dsm["hyper"]["shape_noise"]
    assert sigma_e_int == dsm["hyper"]["sigma_e_int"]
    assert jnp.all(dsp["truth"]["e1"] == dsm["truth"]["e1"])
    assert jnp.all(dsp["truth"]["lf"] == dsm["truth"]["lf"])

    split_size = e1e2p.shape[0] // n_splits
    assert split_size * n_splits == e1e2p.shape[0], "dimensions do not match"
    # Reshape ellipticity samples
    e1e2ps = jnp.reshape(e1e2p, (n_splits, split_size, 300, 2))
    e1e2ms = jnp.reshape(e1e2m, (n_splits, split_size, 300, 2))

    k1, k2 = random.split(rng_key)

    _logtarget = jit(partial(logtarget_shear_and_sn, sigma_e_int=sigma_e_int))
    raw_pipeline = partial(
        run_inference_nuts,
        logtarget=_logtarget,
        n_samples=1000,
        initial_step_size=initial_step_size,
        max_num_doublings=3,
    )
    pipe = jit(raw_pipeline)

    # initializiations for each bootstrap
    _sigma_e_true = dsp["hyper"]["shape_noise"]
    sigma_e_inits = random.uniform(
        k1, shape=(n_splits,), minval=_sigma_e_true - 0.02, maxval=_sigma_e_true + 0.02
    )
    init_positions = {"g": jnp.zeros((n_splits, 2)), "sigma_e": sigma_e_inits}

    # run shear inference pipeline
    # split into batches to avoid memory issues
    keys = jax.random.split(k2, n_splits)
    batch_size = math.ceil(n_splits / n_batches)

    samples_plus = []
    samples_minus = []
    for ii in range(n_batches):
        start = ii * batch_size
        end = min((ii + 1) * batch_size, n_splits)
        _keys = keys[start:end]
        _e1e2ps = e1e2ps[start:end]
        _e1e2ms = e1e2ms[start:end]
        _init_positions = {k: v[start:end] for k, v in init_positions.items()}
        print(f"Running shear inference pipeline (plus) batch {ii + 1}/{n_batches}...")
        sp = vmap(pipe)(_keys, _e1e2ps, _init_positions)
        print(f"Running shear inference pipeline (minus) batch {ii + 1}/{n_batches}...")
        sm = vmap(pipe)(_keys, _e1e2ms, _init_positions)

        samples_plus.append(sp)
        samples_minus.append(sm)

    samples_plus = tree_map(lambda *x: jnp.concatenate(x, axis=0), *samples_plus)
    samples_minus = tree_map(lambda *x: jnp.concatenate(x, axis=0), *samples_minus)
    print(samples_plus["g"].shape)
    assert samples_plus["g"].shape == (n_splits, 1000, 2), "shear samples do not match"
    assert samples_minus["g"].shape == (n_splits, 1000, 2), "shear samples do not match"
    assert samples_plus["sigma_e"].shape == (n_splits, 1000), (
        "sigma_e samples do not match"
    )

    save_dataset(
        {
            "plus": {"g": samples_plus["g"], "sigma_e": samples_plus["sigma_e"]},
            "minus": {"g": samples_minus["g"], "sigma_e": samples_minus["sigma_e"]},
        },
        fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
