#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from functools import partial

import jax
import jax.numpy as jnp
import typer
from jax import Array, jit

from bpd import DATA_DIR
from bpd.io import load_dataset, save_dataset
from bpd.jackknife import run_jackknife_vectorized
from bpd.pipelines import pipeline_shear_inference
from bpd.prior import interim_gprops_logprior, true_all_params_logprior


def _interim_logprior(post_params: dict[str, Array], sigma_e_int: float):
    # we do not evaluate dxdy as we assume it's the same as the true prior and they cancel
    return interim_gprops_logprior(
        post_params, sigma_e=sigma_e_int, free_flux_hlr=True, free_dxdy=False
    )


def main(
    seed: int,
    old_seed: int = typer.Option(),
    samples_plus_fname: str = typer.Option(),
    samples_minus_fname: str = typer.Option(),
    tag: str = typer.Option(),
    initial_step_size: float = 1e-3,
    n_samples: int = 1000,
    n_jacks: int = 100,
    n_splits: int = 10,
    no_bar: bool = False,
    overwrite: bool = False,
):
    rng_key = jax.random.key(seed)

    dirpath = DATA_DIR / "cache_chains" / tag
    samples_plus_fpath = dirpath / samples_plus_fname
    samples_minus_fpath = dirpath / samples_minus_fname
    assert samples_plus_fpath.exists() and samples_minus_fpath.exists()
    out_fpath = dirpath / f"g_samples_jack_{old_seed}_{seed}.npz"

    if out_fpath.exists() and not overwrite:
        raise IOError("overwriting...")

    ds_plus = load_dataset(samples_plus_fpath)
    ds_minus = load_dataset(samples_minus_fpath)

    samples_plus = ds_plus["samples"]
    post_params_plus = {
        "lf": samples_plus["lf"],
        "lhlr": samples_plus["lhlr"],
        "e1": samples_plus["e1"],
        "e2": samples_plus["e2"],
    }

    samples_minus = ds_minus["samples"]
    post_params_minus = {
        "lf": samples_minus["lf"],
        "lhlr": samples_minus["lhlr"],
        "e1": samples_minus["e1"],
        "e2": samples_minus["e2"],
    }

    g1 = ds_plus["hyper"]["g1"]
    g2 = ds_plus["hyper"]["g2"]
    true_g = jnp.array([g1, g2])
    sigma_e = ds_plus["hyper"]["sigma_e"]
    sigma_e_int = ds_plus["hyper"]["sigma_e_int"]
    mean_logflux = ds_plus["hyper"]["mean_logflux"]
    sigma_logflux = ds_plus["hyper"]["sigma_logflux"]
    mean_loghlr = ds_plus["hyper"]["mean_loghlr"]
    sigma_loghlr = ds_plus["hyper"]["sigma_loghlr"]

    g1m = ds_minus["hyper"]["g1"]
    g2m = ds_minus["hyper"]["g2"]
    true_gm = jnp.array([g1m, g2m])

    assert jnp.all(true_g == -true_gm)
    assert sigma_e == ds_minus["hyper"]["sigma_e"]
    assert sigma_e_int == ds_minus["hyper"]["sigma_e_int"]
    assert mean_logflux == ds_minus["hyper"]["mean_logflux"]
    assert mean_loghlr == ds_minus["hyper"]["mean_loghlr"]

    assert jnp.all(ds_plus["truth"]["e1"] == ds_minus["truth"]["e1"])
    assert jnp.all(ds_plus["truth"]["f"] == ds_minus["truth"]["f"])

    logprior_fnc = partial(
        true_all_params_logprior,
        sigma_e=sigma_e,
        mean_logflux=mean_logflux,
        sigma_logflux=sigma_logflux,
        mean_loghlr=mean_loghlr,
        sigma_loghlr=sigma_loghlr,
    )
    interim_logprior_fnc = partial(_interim_logprior, sigma_e_int=sigma_e_int)

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

    g_plus, g_minus = run_jackknife_vectorized(
        rng_key,
        post_params_plus=post_params_plus,
        post_params_minus=post_params_minus,
        shear_pipeline=pipe,
        n_gals=samples_plus["e1"].shape[0],
        n_jacks=n_jacks,
        n_splits=n_splits,
        no_bar=no_bar,
    )

    assert g_plus.shape == (n_jacks, n_samples, 2)
    assert g_minus.shape == (n_jacks, n_samples, 2)

    save_dataset({"g_plus": g_plus, "g_minus": g_minus}, out_fpath, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
