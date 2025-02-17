#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from functools import partial

import jax
import jax.numpy as jnp
import typer
from jax import Array, jit
from jax.scipy import stats

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import load_dataset_jax, save_dataset
from bpd.likelihood import shear_loglikelihood
from bpd.prior import interim_gprops_logprior, true_all_params_logprior
from bpd.utils import uniform_logpdf


def logtarget(
    params,
    *,
    data: Array | dict[str, Array],
    sigma_e_int: float,
    sigma_g: float = 0.01,
):
    g = params["g"]
    sigma_e = params["sigma_e"]
    mean_logflux = params["mean_logflux"]
    sigma_logflux = params["sigma_logflux"]
    mean_loghlr = params["mean_loghlr"]
    sigma_loghlr = params["sigma_loghlr"]

    # ignores dx,dy
    _logprior = partial(
        true_all_params_logprior,
        sigma_e=sigma_e,
        mean_logflux=mean_logflux,
        sigma_logflux=sigma_logflux,
        mean_loghlr=mean_loghlr,
        sigma_loghlr=sigma_loghlr,
    )
    _interim_logprior = partial(
        interim_gprops_logprior,
        sigma_e=sigma_e_int,
        free_flux_hlr=True,
        free_dxdy=False,
    )
    loglike = shear_loglikelihood(
        g, post_params=data, logprior=_logprior, interim_logprior=_interim_logprior
    )
    logprior_g = stats.norm.logpdf(g, loc=0.0, scale=sigma_g).sum()

    # uninformative
    logprior1 = uniform_logpdf(sigma_e, 1e-4, 0.4)
    logprior2 = uniform_logpdf(mean_logflux, 0.0, 6.0)
    logprior3 = uniform_logpdf(sigma_logflux, 0.0, 1.0)
    logprior4 = uniform_logpdf(mean_loghlr, -1.0, 1.0)
    logprior5 = uniform_logpdf(sigma_loghlr, 0.0, 0.25)
    logprior = logprior1 + logprior2 + logprior3 + logprior4 + logprior5 + logprior_g

    return logprior + loglike


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

    samples_dataset = load_dataset_jax(interim_samples_fpath)

    sigma_e_int = samples_dataset["hyper"]["sigma_e_int"]

    g1 = samples_dataset["hyper"]["g1"]
    g2 = samples_dataset["hyper"]["g2"]
    true_g = jnp.array([g1, g2])
    true_sigma_e = samples_dataset["hyper"]["sigma_e"]
    true_mean_logflux = samples_dataset["hyper"]["mean_logflux"]
    true_sigma_logflux = samples_dataset["hyper"]["sigma_logflux"]
    true_mean_loghlr = samples_dataset["hyper"]["mean_loghlr"]
    true_sigma_loghlr = samples_dataset["hyper"]["sigma_loghlr"]

    _logtarget = jit(partial(logtarget, sigma_e_int=sigma_e_int))

    rng_key = jax.random.key(seed)
    samples = run_inference_nuts(
        rng_key,
        {
            "g": true_g,
            "sigma_e": true_sigma_e,
            "mean_logflux": true_mean_logflux,
            "sigma_logflux": true_sigma_logflux,
            "mean_loghlr": true_mean_loghlr,
            "sigma_loghlr": true_sigma_loghlr,
        },
        samples_dataset["samples"],
        logtarget=_logtarget,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
        max_num_doublings=5,
        n_warmup_steps=1000,
    )

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
            "g1": true_g[0].item(),
            "g2": true_g[1].item(),
            "sigma_e": true_sigma_e,
            "mean_logflux": true_mean_logflux,
            "sigma_logflux": true_sigma_logflux,
            "mean_loghlr": true_mean_loghlr,
            "sigma_loghlr": true_sigma_loghlr,
        },
    }
    save_dataset(out, fpath, overwrite=overwrite)


if __name__ == "__main__":
    typer.run(main)
