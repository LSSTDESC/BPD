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
from bpd.prior import ellip_prior_e1e2, true_ellip_logprior


def logtarget(
    params,
    *,
    data: Array | dict[str, Array],
    sigma_e_int: float,
    sigma_g: float = 0.01,
):
    g = params["g"]
    sigma_e = params["sigma_e"]
    _logprior = lambda e, g: true_ellip_logprior(e, g, sigma_e=sigma_e)
    _interim_logprior = lambda e: jnp.log(ellip_prior_e1e2(e, sigma=sigma_e_int))
    loglike = shear_loglikelihood(
        g, post_params=data, logprior=_logprior, interim_logprior=_interim_logprior
    )
    logprior1 = stats.norm.logpdf(g, loc=0.0, scale=sigma_g).sum()
    logprior2 = stats.uniform.logpdf(sigma_e, 1e-4, 0.4 - 1e-4)  # uninformative
    return logprior1 + logprior2 + loglike


def main(
    seed: int,
    samples_fname: str = typer.Option(),
    tag: str = typer.Option(),
    initial_step_size: float = 0.01,
    n_samples: int = 3000,
    trim: int = 1,
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
    e_post = samples_dataset["e_post"][:, ::trim, :]
    true_g = samples_dataset["true_g"]
    true_sigma_e = samples_dataset["sigma_e"]
    sigma_e_int = samples_dataset["sigma_e_int"]

    _logtarget = jit(partial(logtarget, sigma_e_int=sigma_e_int))

    rng_key = jax.random.key(seed)
    samples = run_inference_nuts(
        rng_key,
        {"g": true_g, "sigma_e": true_sigma_e},
        e_post,
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
            "g1": true_g[0].item(),
            "g2": true_g[1].item(),
            "sigma_e": true_sigma_e,
        },
    }
    save_dataset(out, fpath, overwrite=overwrite)


if __name__ == "__main__":
    typer.run(main)
