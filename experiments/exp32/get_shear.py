#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from functools import partial
from pathlib import Path
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
from bpd.prior import ellip_prior_e1e2


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
    e_post = post_params["e1e2"]

    prior = jnp.array(0.0)
    prior += stats.norm.logpdf(lf, loc=mean_logflux, scale=sigma_logflux)
    prior += stats.norm.logpdf(lhlr, loc=mean_loghlr, scale=sigma_loghlr)
    prior += true_ellip_logprior(e_post, g, sigma_e=sigma_e)

    return prior


def _interim_logprior(post_params: dict[str, Array], sigma_e_int: float):
    return logprior(post_params, sigma_e=sigma_e_int, free_dxdy=False)


def pipeline_shear_inference(
    rng_key: PRNGKeyArray,
    e_post: Array,
    init_g: Array,
    *,
    sigma_e: float,
    sigma_e_int: float,
    n_samples: int,
    initial_step_size: float,
    sigma_g: float = 0.01,
    n_warmup_steps: int = 500,
    max_num_doublings: int = 2,
):
    # NOTE: jit must be applied without `e_post` in partial!
    _loglikelihood = jit(
        partial(
            shear_loglikelihood,
            logprior=partial(_logprior, sigma_e=sigma_e),
            interim_logprior=partial(_interim_logprior, sigma_e_int=sigma_e_int),
        )
    )
    _logtarget = partial(
        logtarget_density, loglikelihood=_loglikelihood, sigma_g=sigma_g
    )

    _do_inference = partial(
        run_inference_nuts,
        data={"e1e2": e_post},
        logtarget=_logtarget,
        n_samples=n_samples,
        n_warmup_steps=n_warmup_steps,
        max_num_doublings=max_num_doublings,
        initial_step_size=initial_step_size,
    )

    g_samples = _do_inference(rng_key, init_g)

    return g_samples


def _extract_seed(fpath: str) -> int:
    name = Path(fpath).name
    first = name.find("_")
    second = name.find("_", first + 1)
    third = name.find(".")
    return int(name[second + 1 : third])


def main(
    seed: int,
    tag: str,
    interim_samples_fname: str,
    initial_step_size: float = 1e-3,
    n_samples: int = 3000,
    trim: int = 1,
    overwrite: bool = False,
):
    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    assert dirpath.exists()
    interim_samples_fpath = DATA_DIR / "cache_chains" / tag / interim_samples_fname
    assert interim_samples_fpath.exists(), "ellipticity samples file does not exist"
    old_seed = _extract_seed(interim_samples_fpath)
    fpath = DATA_DIR / "cache_chains" / tag / f"g_samples_{old_seed}_{seed}.npy"

    if fpath.exists() and not overwrite:
        raise IOError("overwriting...")

    samples_dataset = load_dataset(interim_samples_fpath)
    e_post = samples_dataset["e_post"][:, ::trim, :]
    true_g = samples_dataset["true_g"]
    sigma_e = samples_dataset["sigma_e"]
    sigma_e_int = samples_dataset["sigma_e_int"]

    rng_key = jax.random.key(seed)
    g_samples = pipeline_shear_inference_ellipticities(
        rng_key,
        e_post,
        init_g=true_g,
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )

    jnp.save(fpath, g_samples)


if __name__ == "__main__":
    typer.run(main)
