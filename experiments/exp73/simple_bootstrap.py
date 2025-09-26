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
from bpd.io import load_dataset, save_dataset
from bpd.pipelines import init_all_params, logtarget_all_free

CPU = jax.devices("cpu")[0]
GPU = jax.devices("gpu")[0]


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_plus_fpath: str = typer.Option(),
    samples_minus_fpath: str = typer.Option(),
    initial_step_size: float = 0.01,
    n_samples: int = 1000,
    n_boots=100,
    overwrite: bool = False,
    no_bar: bool = False,
    n_gals: int | None = None,
):
    k1, k2, k3 = random.split(seed, 3)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    fpp = Path(samples_plus_fpath)
    fpm = Path(samples_minus_fpath)
    assert dirpath.exists() and fpp.exists() and fpm.exists()
    fpath = dirpath / f"g_samples_boots_{seed}.npz"

    dsp = load_dataset(fpp)  # carefuly with memory from the start
    dsm = load_dataset(fpm)

    total_n_gals = dsp["samples"]["e1"].shape[0]
    if n_gals is None:
        n_gals = total_n_gals
    assert n_gals <= total_n_gals
    subset = random.choice(k1, jnp.arange(total_n_gals), shape=(n_gals,), replace=False)

    ppp = {
        "lf": dsp["samples"]["lf"][subset],
        "lhlr": dsp["samples"]["lhlr"][subset],
        "e1": dsp["samples"]["e1"][subset],
        "e2": dsp["samples"]["e2"][subset],
    }

    ppm = {
        "lf": dsm["samples"]["lf"][subset],
        "lhlr": dsm["samples"]["lhlr"][subset],
        "e1": dsm["samples"]["e1"][subset],
        "e2": dsm["samples"]["e2"][subset],
    }

    true_sigma_e = dsp["hyper"]["shape_noise"]
    true_mean_logflux = dsp["hyper"]["mean_logflux"]
    true_sigma_logflux = dsp["hyper"]["sigma_logflux"]
    true_mean_loghlr = dsp["hyper"]["mean_loghlr"]
    true_sigma_loghlr = dsp["hyper"]["sigma_loghlr"]
    true_a_logflux = dsp["hyper"]["a_logflux"]

    sigma_e_int = dsp["hyper"]["sigma_e_int"]

    assert true_sigma_e == dsm["hyper"]["shape_noise"]
    assert true_mean_logflux == dsm["hyper"]["mean_logflux"]
    assert true_sigma_logflux == dsm["hyper"]["sigma_logflux"]
    assert true_mean_loghlr == dsm["hyper"]["mean_loghlr"]
    assert true_sigma_loghlr == dsm["hyper"]["sigma_loghlr"]
    assert true_a_logflux == dsm["hyper"]["a_logflux"]

    _logtarget = jit(partial(logtarget_all_free, sigma_e_int=sigma_e_int))

    # initialize positions randomly uniform from true parameters
    init_positions = []
    k2s = random.split(k2, n_boots)
    for ii in range(n_boots):
        init_params = init_all_params(
            k2s[ii],
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
        init_params["g"] = jnp.array([0.0, 0.0])
        init_positions.append(init_params)

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
    _ = pipe(k3, data={k: v[0, None] for k, v in ppp.items()})

    samples_plus = run_bootstrap(
        k3,
        init_positions,
        post_params=ppp,
        n_gals=ppp["e1"].shape[0],
        n_boots=n_boots,
        no_bar=no_bar,
        cpu=CPU,
        gpu=GPU,
    )

    samples_minus = run_bootstrap(
        k3,
        init_positions,
        post_params=ppm,
        n_gals=ppm["e1"].shape[0],
        n_boots=n_boots,
        no_bar=no_bar,
        cpu=CPU,
        gpu=GPU,
    )

    assert samples_plus["g"] == (n_samples, 2)
    assert samples_minus["g"] == (n_samples, 2)

    gp = samples_plus.pop("g")
    gm = samples_minus.pop("g")
    g1p = gp[:, 0]
    g2p = gp[:, 1]
    g1m = gm[:, 0]
    g2m = gm[:, 1]

    out = {
        "plus": {"g1": g1p, "g2": g2p, **samples_plus},
        "minus": {"g1": g1m, "g2": g2m, **samples_minus},
    }
    save_dataset(out, fpath, overwrite=overwrite)


if __name__ == "__main__":
    typer.run(main)
