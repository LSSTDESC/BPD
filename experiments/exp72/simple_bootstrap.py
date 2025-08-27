#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import typer
from jax import random

from bpd import DATA_DIR
from bpd.io import load_dataset_jax, save_dataset
from bpd.jackknife import run_bootstrap_shear_pipeline
from bpd.pipelines import pipeline_shear_inference
from bpd.prior import interim_gprops_logprior, true_all_params_skew_logprior


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_plus_fpath: str = typer.Option(),
    samples_minus_fpath: str = typer.Option(),
    initial_step_size: float = 0.001,
    n_samples: int = 1000,
    n_boots: int = 25,
    no_bar: bool = False,
    n_gals: int | None = None,
):
    rng_key = random.key(seed)
    k1, k2 = random.split(rng_key)

    dirpath = DATA_DIR / "cache_chains" / tag
    pfpath = Path(samples_plus_fpath)
    mfpath = Path(samples_minus_fpath)
    assert dirpath.exists() and pfpath.exists() and mfpath.exists()
    fpath = dirpath / f"g_samples_boots_{seed}.npz"

    dsp = load_dataset_jax(samples_plus_fpath)
    dsm = load_dataset_jax(samples_minus_fpath)

    total_n_gals = dsp["samples"]["e1"].shape[0]
    if n_gals is not None:
        subset = random.choice(
            k1, jnp.arange(total_n_gals), shape=(n_gals,), replace=False
        )
    else:
        subset = jnp.arange(total_n_gals)

    # positions are not used as we assume true and interim prior cancels
    post_params_plus = {
        "lf": dsp["samples"]["lf"][subset],
        "lhlr": dsp["samples"]["lhlr"][subset],
        "e1": dsp["samples"]["e1"][subset],
        "e2": dsp["samples"]["e2"][subset],
    }
    post_params_minus = {
        "lf": dsm["samples"]["lf"][subset],
        "lhlr": dsm["samples"]["lhlr"][subset],
        "e1": dsm["samples"]["e1"][subset],
        "e2": dsm["samples"]["e2"][subset],
    }

    sigma_e = dsp["hyper"]["shape_noise"]
    sigma_e_int = dsp["hyper"]["sigma_e_int"]
    mean_logflux = dsp["hyper"]["mean_logflux"]
    sigma_logflux = dsp["hyper"]["sigma_logflux"]
    a_logflux = dsp["hyper"]["a_logflux"]
    mean_loghlr = dsp["hyper"]["mean_loghlr"]
    sigma_loghlr = dsp["hyper"]["sigma_loghlr"]

    assert dsp["hyper"]["g1"] == -dsm["hyper"]["g1"]
    assert dsp["hyper"]["g2"] == -dsm["hyper"]["g2"]
    assert sigma_e == dsm["hyper"]["shape_noise"]
    assert sigma_e_int == dsm["hyper"]["sigma_e_int"]
    assert mean_logflux == dsm["hyper"]["mean_logflux"]
    assert mean_loghlr == dsm["hyper"]["mean_loghlr"]
    assert jnp.all(dsp["truth"]["e1"] == dsm["truth"]["e1"])
    assert jnp.all(dsp["truth"]["lf"] == dsm["truth"]["lf"])

    logprior_fnc = partial(
        true_all_params_skew_logprior,
        sigma_e=sigma_e,
        mean_logflux=mean_logflux,
        sigma_logflux=sigma_logflux,
        mean_loghlr=mean_loghlr,
        sigma_loghlr=sigma_loghlr,
        a_logflux=a_logflux,
    )
    interim_logprior_fnc = partial(
        interim_gprops_logprior,
        sigma_e=sigma_e_int,
        free_flux_hlr=True,
        free_dxdy=False,
    )

    raw_pipeline = partial(
        pipeline_shear_inference,
        init_g=jnp.array([0.0, 0.0]),
        logprior=logprior_fnc,
        interim_logprior=interim_logprior_fnc,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )

    @jax.jit
    def pipe(k, d):
        out = raw_pipeline(k, d)
        return {"g1": out[..., 0], "g2": out[..., 1]}

    # jit pipe function (to avoid memory error)
    print("Jitting pipeline...")
    _ = pipe(k2, {k: v[0, None] for k, v in post_params_plus.items()})
    print("Pipeline jitted.")

    samples_plus, samples_minus = run_bootstrap_shear_pipeline(
        k2,
        post_params_plus=post_params_plus,
        post_params_minus=post_params_minus,
        shear_pipeline=pipe,
        n_gals=post_params_plus["e1"].shape[0],
        n_boots=n_boots,
        no_bar=no_bar,
    )

    gp = jnp.stack([samples_plus["g1"], samples_plus["g2"]], axis=-1)
    gm = jnp.stack([samples_minus["g1"], samples_minus["g2"]], axis=-1)

    assert gp.shape == (n_boots, n_samples, 2)
    assert gm.shape == (n_boots, n_samples, 2)

    save_dataset({"gp": gp, "gm": gm}, fpath, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
