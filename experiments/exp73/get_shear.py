#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import typer
from jax import jit, random

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import load_dataset_jax, save_dataset
from bpd.pipelines import init_all_params, logtarget_all_free


def main(
    seed: int,
    samples_fpath: str = typer.Option(),
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
    interim_samples_fpath = Path(samples_fpath)
    assert interim_samples_fpath.exists(), "ellipticity samples file does not exist"
    fpath = DATA_DIR / "cache_chains" / tag / f"shear_samples_{seed}{extra_txt}.npz"

    ds = load_dataset_jax(interim_samples_fpath)

    post_params = {
        "lf": ds["samples"]["lf"],
        "lhlr": ds["samples"]["lhlr"],
        "e1": ds["samples"]["e1"],
        "e2": ds["samples"]["e2"],
    }

    true_sigma_e = ds["hyper"]["shape_noise"]
    true_mean_logflux = ds["hyper"]["mean_logflux"]
    true_sigma_logflux = ds["hyper"]["sigma_logflux"]
    true_mean_loghlr = ds["hyper"]["mean_loghlr"]
    true_sigma_loghlr = ds["hyper"]["sigma_loghlr"]
    true_a_logflux = ds["hyper"]["a_logflux"]

    sigma_e_int = ds["hyper"]["sigma_e_int"]

    _logtarget = jit(partial(logtarget_all_free, sigma_e_int=sigma_e_int))

    rng_key = jax.random.key(seed)
    k1, k2 = random.split(rng_key)

    # initialize positions randomly uniform from true parameters
    init_positions = init_all_params(
        k1,
        true_params={
            "sigma_e": true_sigma_e,
            "mean_logflux": true_mean_logflux,
            "sigma_logflux": true_sigma_logflux,
            "a_logflux": true_a_logflux,
            "mean_loghlr": true_mean_loghlr,
            "sigma_loghlr": true_sigma_loghlr,
        },
        p=0.1,
    )
    init_positions["g"] = jnp.array([0.0, 0.0])

    _pipe = partial(
        run_inference_nuts,
        init_positions=init_positions,
        logtarget=_logtarget,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
        max_num_doublings=7,
        n_warmup_steps=1000,
    )
    pipe = jit(_pipe)

    # jit function quickly
    print("JITting function...")
    _ = pipe(k2, data={k: v[0, None] for k, v in post_params.items()})

    # then run on full data
    print("Running inference...")
    samples = pipe(k2, data=post_params)

    assert samples["g"].shape == (n_samples, 2)
    g = samples.pop("g")
    g1 = g[:, 0]
    g2 = g[:, 1]

    out = {
        "samples": {
            "g1": g1,
            "g2": g2,
            **samples,
        },
        "truth": {
            "g1": ds["hyper"]["g1"],
            "g2": ds["hyper"]["g2"],
            "sigma_e": true_sigma_e,
            "sigma_e_int": sigma_e_int,
            "a_logflux": true_a_logflux,
            "mean_logflux": true_mean_logflux,
            "sigma_logflux": true_sigma_logflux,
            "mean_loghlr": true_mean_loghlr,
            "sigma_loghlr": true_sigma_loghlr,
        },
    }
    save_dataset(out, fpath, overwrite=overwrite)


if __name__ == "__main__":
    typer.run(main)
