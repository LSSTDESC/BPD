#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import typer
from jax import Array, jit
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import load_dataset
from bpd.likelihood import shear_loglikelihood, true_ellip_logprior
from bpd.pipelines.image_samples import logprior


def logtarget_density(
    g: Array, *, data: Array, loglikelihood: Callable, sigma_g: float = 0.01
):
    loglike = loglikelihood(g, post_params=data)
    logprior = stats.norm.logpdf(g, loc=0.0, scale=sigma_g).sum()
    return logprior + loglike


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
    return logprior(
        post_params, sigma_e=sigma_e_int, free_flux_hlr=True, free_dxdy=False
    )


def pipeline_shear_inference(
    rng_key: PRNGKeyArray,
    post_params: Array,
    init_g: Array,
    *,
    logprior: Callable,
    interim_logprior: Callable,
    n_samples: int,
    initial_step_size: float,
    sigma_g: float = 0.01,
    n_warmup_steps: int = 500,
    max_num_doublings: int = 2,
):
    # NOTE: jit must be applied without `e_post` in partial!
    _loglikelihood = jit(
        partial(
            shear_loglikelihood, logprior=logprior, interim_logprior=interim_logprior
        )
    )
    _logtarget = partial(
        logtarget_density, loglikelihood=_loglikelihood, sigma_g=sigma_g
    )

    _do_inference = partial(
        run_inference_nuts,
        data=post_params,
        logtarget=_logtarget,
        n_samples=n_samples,
        n_warmup_steps=n_warmup_steps,
        max_num_doublings=max_num_doublings,
        initial_step_size=initial_step_size,
    )

    g_samples = _do_inference(rng_key, init_g)

    return g_samples


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
        logprior=logprior_fnc,
        interim_logprior=interim_logprior_fnc,
        init_g=true_g,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )

    jnp.save(fpath, g_samples)


if __name__ == "__main__":
    typer.run(main)
