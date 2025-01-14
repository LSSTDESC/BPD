#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from functools import partial

import jax
import jax.numpy as jnp
import typer
from jax import Array
from jax.scipy import stats

from bpd import DATA_DIR
from bpd.io import load_dataset
from bpd.pipelines import pipeline_shear_inference
from bpd.prior import interim_gprops_logprior, true_ellip_logprior


def _logprior(
    post_params: dict[str, Array],
    g: Array,
    *,
    sigma_e: float,
    mean_logflux: float,
    sigma_logflux: float,
    mean_loghlr: float,
    sigma_loghlr: float,
):
    lf = post_params["lf"]
    lhlr = post_params["lhlr"]
    e_post = jnp.stack((post_params["e1"], post_params["e2"]), axis=-1)

    prior = jnp.array(0.0)

    # we also pickup a jacobian in the corresponding probability densities
    prior += stats.norm.logpdf(lf, loc=mean_logflux, scale=sigma_logflux)
    prior += stats.norm.logpdf(lhlr, loc=mean_loghlr, scale=sigma_loghlr)

    # elliptcity
    prior += true_ellip_logprior(e_post, g, sigma_e=sigma_e)

    return prior


def _interim_logprior(post_params: dict[str, Array], sigma_e_int: float):
    # we do not evaluate dxdy as we assume it's the same as the true prior and they cancel
    return interim_gprops_logprior(
        post_params, sigma_e=sigma_e_int, free_flux_hlr=True, free_dxdy=False
    )


def main(
    seed: int,
    initial_step_size: float = 1e-3,
    n_samples: int = 3000,
    overwrite: bool = False,
):
    # directory structure
    dirpath = DATA_DIR / "cache_chains" / f"exp32_{seed}"
    interim_samples_fpath = dirpath / f"interim_samples_{seed}.npz"
    assert interim_samples_fpath.exists(), "ellipticity samples file does not exist"
    assert dirpath.exists()
    fpath = dirpath / f"g_samples_{seed}.npy"

    if fpath.exists() and not overwrite:
        raise IOError("overwriting...")

    samples_dataset = load_dataset(interim_samples_fpath)
    true_g = samples_dataset["true_g"]

    # prior parameters
    sigma_e = samples_dataset["sigma_e"]
    sigma_e_int = samples_dataset["sigma_e_int"]
    mean_logflux = samples_dataset["mean_logflux"]
    sigma_logflux = samples_dataset["sigma_logflux"]
    mean_loghlr = samples_dataset["mean_loghlr"]
    sigma_loghlr = samples_dataset["sigma_loghlr"]

    # data
    post_params = {
        "lf": samples_dataset["lf"],
        "lhlr": samples_dataset["lhlr"],
        "e1": samples_dataset["e_post"][..., 0],
        "e2": samples_dataset["e_post"][..., 1],
    }

    # setup priors
    logprior_fnc = partial(
        _logprior,
        sigma_e=sigma_e,
        mean_logflux=mean_logflux,
        sigma_logflux=sigma_logflux,
        mean_loghlr=mean_loghlr,
        sigma_loghlr=sigma_loghlr,
    )
    interim_logprior_fnc = partial(_interim_logprior, sigma_e_int=sigma_e_int)

    rng_key = jax.random.key(seed)
    g_samples = pipeline_shear_inference(
        rng_key,
        post_params,
        init_g=true_g,
        logprior=logprior_fnc,
        interim_logprior=interim_logprior_fnc,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )

    jnp.save(fpath, g_samples)


if __name__ == "__main__":
    typer.run(main)
