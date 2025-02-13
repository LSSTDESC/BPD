#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import typer
from jax import Array
from jax.scipy import stats

from bpd import DATA_DIR
from bpd.io import load_dataset_jax
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
    e1e2 = jnp.stack((post_params["e1"], post_params["e2"]), axis=-1)

    prior = jnp.array(0.0)

    # we also pickup a jacobian in the corresponding probability densities
    prior += stats.norm.logpdf(lf, loc=mean_logflux, scale=sigma_logflux)
    prior += stats.norm.logpdf(lhlr, loc=mean_loghlr, scale=sigma_loghlr)

    # elliptcity
    prior += true_ellip_logprior(e1e2, g, sigma_e=sigma_e)

    return prior


def _interim_logprior(post_params: dict[str, Array], sigma_e_int: float):
    # we do not evaluate dxdy as we assume it's the same as the true prior and they cancel
    return interim_gprops_logprior(
        post_params, sigma_e=sigma_e_int, free_flux_hlr=True, free_dxdy=False
    )


def main(
    seed: int,
    tag: str,
    samples_fname: str,
    initial_step_size: float = 1e-3,
    n_samples: int = 3000,
    overwrite: bool = False,
):
    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    interim_samples_fpath = dirpath / samples_fname
    assert dirpath.exists()
    assert interim_samples_fpath.exists(), "ellipticity samples file does not exist"
    out_fpath = dirpath / f"g_samples_{seed}.npy"

    if out_fpath.exists() and not overwrite:
        raise IOError("overwriting...")

    ds = load_dataset_jax(interim_samples_fpath)

    # data
    samples = ds["samples"]
    post_params = {
        "lf": samples["lf"],
        "lhlr": samples["lhlr"],
        "e1": samples["e_post"][..., 0],
        "e2": samples["e_post"][..., 1],
    }

    # prior parameters
    hyper = ds["truth"]
    true_g = hyper["g"]
    sigma_e_int = hyper["sigma_e_int"]
    sigma_e = hyper["sigma_e"]
    mean_logflux = hyper["mean_logflux"]
    sigma_logflux = hyper["sigma_logflux"]
    mean_loghlr = hyper["mean_loghlr"]
    sigma_loghlr = hyper["sigma_loghlr"]

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

    np.save(out_fpath, np.asarray(g_samples))


if __name__ == "__main__":
    typer.run(main)
