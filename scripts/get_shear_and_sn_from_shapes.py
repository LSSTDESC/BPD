#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from functools import partial

import jax
import jax.numpy as jnp
import typer
from jax import jit

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import load_dataset_jax, save_dataset
from bpd.pipelines import logtarget_shear_and_sn


def main(
    seed: int,
    samples_fname: str = typer.Option(),
    tag: str = typer.Option(),
    initial_step_size: float = 0.01,
    n_samples: int = 3000,
    overwrite: bool = False,
    extra_tag: str = "",
):
    extra_txt = f"_{extra_tag}" if extra_tag else ""
    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)
    interim_samples_fpath = DATA_DIR / "cache_chains" / tag / samples_fname
    assert interim_samples_fpath.exists(), "ellipticity samples file does not exist"
    fpath = DATA_DIR / "cache_chains" / tag / f"shear_samples_{seed}{extra_txt}.npz"

    ds = load_dataset_jax(interim_samples_fpath)
    e1 = ds["samples"]["e1"]
    e2 = ds["samples"]["e2"]
    e1e2 = jnp.stack([e1, e2], axis=-1)
    sigma_e_int = ds["hyper"]["sigma_e_int"]
    _logtarget = jit(partial(logtarget_shear_and_sn, sigma_e_int=sigma_e_int))
    rng_key = jax.random.key(seed)
    samples = run_inference_nuts(
        rng_key,
        data=e1e2,
        init_positions={"g": jnp.array([0.0, 0.0]), "sigma_e": 0.2},
        logtarget=_logtarget,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
        max_num_doublings=3,
        n_warmup_steps=500,
    )

    assert samples["g"].shape == (n_samples, 2)

    out = {
        "samples": {
            "g1": samples["g"][:, 0],
            "g2": samples["g"][:, 1],
            "sigma_e": samples["sigma_e"],
        },
        "truth": {
            "g1": ds["hyper"]["g1"],
            "g2": ds["hyper"]["g2"],
            "sigma_e": ds["hyper"]["shape_noise"],
        },
    }
    save_dataset(out, fpath, overwrite=overwrite)


if __name__ == "__main__":
    typer.run(main)
