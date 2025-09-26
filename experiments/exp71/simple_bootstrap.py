#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import typer
from jax import jit, random

from bpd import DATA_DIR
from bpd.bootstrap import run_bootstrap
from bpd.chains import run_inference_nuts
from bpd.io import load_dataset_jax, save_dataset
from bpd.pipelines import logtarget_shear_and_sn


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_plus_fpath: str = typer.Option(),
    samples_minus_fpath: str = typer.Option(),
    initial_step_size: float = 0.001,
    n_samples: int = 1000,
    n_boots: int = 30,
    no_bar: bool = False,
    n_gals: int | None = None,
):
    rng_key = jax.random.key(seed)
    k1, k2, k3 = random.split(rng_key, 3)

    dirpath = DATA_DIR / "cache_chains" / tag
    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)

    samples_plus_fpath = Path(samples_plus_fpath)
    samples_minus_fpath = Path(samples_minus_fpath)
    assert samples_plus_fpath.exists() and samples_minus_fpath.exists()
    fpath = dirpath / f"g_samples_boots_{seed}.npz"

    dsp = load_dataset_jax(samples_plus_fpath)
    dsm = load_dataset_jax(samples_minus_fpath)

    total_n_gals = dsp["samples"]["e1"].shape[0]
    if n_gals is None:
        n_gals = total_n_gals
    assert n_gals <= total_n_gals
    subset = random.choice(k1, jnp.arange(total_n_gals), shape=(n_gals,), replace=False)

    e1p = dsp["samples"]["e1"][subset]
    e2p = dsp["samples"]["e2"][subset]
    e1e2p = jnp.stack([e1p, e2p], axis=-1)

    e1m = dsm["samples"]["e1"][subset]
    e2m = dsm["samples"]["e2"][subset]
    e1e2m = jnp.stack([e1m, e2m], axis=-1)

    sigma_e_int = dsp["hyper"]["sigma_e_int"]
    assert dsp["hyper"]["shape_noise"] == dsm["hyper"]["shape_noise"]
    assert sigma_e_int == dsm["hyper"]["sigma_e_int"]
    assert jnp.all(dsp["truth"]["e1"] == dsm["truth"]["e1"])
    assert jnp.all(dsp["truth"]["lf"] == dsm["truth"]["lf"])

    _logtarget = jit(partial(logtarget_shear_and_sn, sigma_e_int=sigma_e_int))
    raw_pipeline = partial(
        run_inference_nuts,
        logtarget=_logtarget,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
        max_num_doublings=3,
    )
    _pipe = jit(raw_pipeline)
    pipe = lambda k, d, ip: _pipe(k, d["e1e2"], ip)

    # initializiations for each bootstrap
    _sigma_e_true = dsp["hyper"]["shape_noise"]
    init_positions = []
    k2s = random.split(k2, n_boots)
    for ii in range(n_boots):
        g = jnp.array([0.0, 0.0])
        sigma_e = random.uniform(
            k2s[ii], shape=(), minval=_sigma_e_true - 0.02, maxval=_sigma_e_true + 0.02
        )
        init_positions.append({"g": g, "sigma_e": sigma_e})

    samples_plus = run_bootstrap(
        k3,
        init_positions,
        post_params={"e1e2": e1e2p},
        pipeline=pipe,
        n_gals=e1e2p.shape[0],
        n_boots=n_boots,
        no_bar=no_bar,
    )

    samples_minus = run_bootstrap(
        k3,
        init_positions,
        post_params={"e1e2": e1e2m},
        pipeline=pipe,
        n_gals=e1e2p.shape[0],
        n_boots=n_boots,
        no_bar=no_bar,
    )

    assert samples_plus["g"].shape == (n_boots, n_samples, 2)
    assert samples_minus["g"].shape == (n_boots, n_samples, 2)

    save_dataset(
        {
            "plus": {
                "g1": samples_plus["g"][..., 0],
                "g2": samples_plus["g"][..., 1],
                "sigma_e": samples_plus["sigma_e"],
            },
            "minus": {
                "g1": samples_minus["g"][..., 0],
                "g2": samples_minus["g"][..., 1],
                "sigma_e": samples_minus["sigma_e"],
            },
        },
        fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
