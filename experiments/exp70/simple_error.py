#!/usr/bin/env python3

from functools import partial

import jax
import jax.numpy as jnp
import typer

from bpd import DATA_DIR
from bpd.io import load_dataset_jax, save_dataset
from bpd.pipelines import pipeline_shear_inference_simple


def main(
    seed: int,
    tag: str = typer.Option(),
    plus_samples_fname: str = typer.Option(),
    minus_samples_fname: str = typer.Option(),
    initial_step_size: float = 1e-3,
    n_splits: int = 500,
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
    e1e2p = jnp.reshape(e1e2p, (n_splits, split_size, 300, 2))
    e1e2m = jnp.reshape(e1e2m, (n_splits, split_size, 300, 2))

    # get shear inference pipeline
    _pipe = partial(
        pipeline_shear_inference_simple,
        init_g=jnp.array([0.0, 0.0]),
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        n_samples=1000,
        initial_step_size=initial_step_size,
    )
    _pipe = jax.jit(_pipe)
    pipe = jax.vmap(_pipe, in_axes=(0, 0))

    # run shear inference pipeline
    keys = jax.random.split(rng_key, n_splits)
    print("Running shear inference pipeline (plus)...")
    gp = pipe(keys, e1e2p)
    print("Running shear inference pipeline (minus)...")
    gm = pipe(keys, e1e2m)

    assert gp.shape == gm.shape, "shear samples do not match"
    assert gp.shape == (n_splits, 1000, 2), "shear samples do not match"

    save_dataset(
        {
            "g_plus": gp,
            "g_minus": gm,
            "sigma_e": sigma_e,
            "sigma_e_int": sigma_e_int,
        },
        fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
