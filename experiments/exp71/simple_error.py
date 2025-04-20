#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import typer
from jax import jit, random, vmap

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import load_dataset_jax, save_dataset
from bpd.pipelines import logtarget_shear_and_sn


def main(
    seed: int,
    tag: str = typer.Option(),
    plus_samples_fpath: str = typer.Option(),
    minus_samples_fpath: str = typer.Option(),
    initial_step_size: float = 1e-3,
    n_splits: int = 500,
):
    rng_key = jax.random.key(seed)

    dirpath = DATA_DIR / "cache_chains" / tag
    pfpath = Path(plus_samples_fpath)
    mfpath = Path(minus_samples_fpath)
    fpath = dirpath / f"g_samples_{seed}_errs.npz"

    assert dirpath.exists()
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
    e1e2p = jnp.reshape(e1e2p, (n_splits, split_size, 300, 2))
    e1e2m = jnp.reshape(e1e2m, (n_splits, split_size, 300, 2))

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
    keys = jax.random.split(k2, n_splits)
    print("Running shear inference pipeline (plus)...")
    plus_samples = vmap(pipe)(keys, e1e2p, init_positions)
    print("Running shear inference pipeline (minus)...")
    minus_samples = vmap(pipe)(keys, e1e2m, init_positions)
    assert plus_samples["g"].shape == (n_splits, 1000, 2), "shear samples do not match"
    save_dataset(
        {
            "plus": {"g": plus_samples["g"], "sigma_e": plus_samples["sigma_e"]},
            "minus": {"g": minus_samples["g"], "sigma_e": minus_samples["sigma_e"]},
        },
        fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
