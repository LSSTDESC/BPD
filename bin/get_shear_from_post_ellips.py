#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from functools import partial
from pathlib import Path
from typing import Callable

import blackjax
import click
import jax
import jax.numpy as jnp
from jax import jit as jjit
from jax import random
from jax.scipy import stats
from jax.typing import ArrayLike

from bpd import DATA_DIR
from bpd.chains import inference_loop
from bpd.io import load_dataset
from bpd.likelihood import shear_loglikelihood
from bpd.prior import ellip_mag_prior


def _extract_seed(fpath: str) -> int:
    name = Path(fpath).name
    first = name.find("_")
    second = name.find("_", first + 1)
    third = name.find(".")
    return int(name[second + 1 : third])


def logtarget_density(g: ArrayLike, e_post: ArrayLike, loglikelihood: Callable):
    loglike = loglikelihood(g, e_post)
    logprior = stats.uniform.logpdf(g, -0.1, 0.2).sum()
    return logprior + loglike


def do_inference(rng_key, init_g: ArrayLike, logtarget: Callable, n_samples: int):
    key1, key2 = random.split(rng_key)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logtarget,
        progress_bar=False,
        is_mass_matrix_diagonal=True,
        max_num_doublings=2,
        initial_step_size=1e-2,
        target_acceptance_rate=0.80,
    )

    (init_states, tuned_params), _ = warmup.run(key1, init_g, 500)
    kernel = blackjax.nuts(logtarget, **tuned_params).step
    states, _ = inference_loop(key2, init_states, kernel=kernel, n_samples=n_samples)
    return states.position


def pipeline_shear_inference(
    seed: int, e_post: ArrayLike, true_g: ArrayLike, sigma_e: float, n_samples: int
):
    rng_key = random.key(seed)
    prior = partial(ellip_mag_prior, sigma=sigma_e)
    interim_prior = partial(ellip_mag_prior, sigma=sigma_e * 2)

    # NOTE: jit must be applied without `e_post` in partial!
    _loglikelihood = jjit(
        partial(shear_loglikelihood, prior=prior, interim_prior=interim_prior)
    )

    _logtarget = partial(logtarget_density, loglikelihood=_loglikelihood, e_post=e_post)
    _do_inference = partial(do_inference, logtarget=_logtarget, n_samples=n_samples)

    g_samples = _do_inference(rng_key, true_g)

    return g_samples


@click.command()
@click.option("--tag", type=str, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--e-samples-fname", type=str, required=True)
@click.option("-n", "--n-samples", type=int, default=3000, help="# of shear samples")
@click.option("--trim", type=int, default=1, help="trimming makes like. eval. fast")
@click.option("--overwrite", type=bool, default=False)
def main(
    tag: str,
    seed: int,
    e_samples_fname: str,
    n_samples: int,
    trim: int,
    overwrite: bool,
):

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    assert dirpath.exists()
    e_samples_fpath = DATA_DIR / "cache_chains" / tag / e_samples_fname
    assert e_samples_fpath.exists(), "ellipticity samples file does not exist"
    old_seed = _extract_seed(e_samples_fname)
    fpath = DATA_DIR / "cache_chains" / tag / f"g_samples_{old_seed}_{seed}.npy"

    if fpath.exists():
        if not overwrite:
            raise IOError("overwriting...")

    samples_dataset = load_dataset(e_samples_fpath)
    e_post = samples_dataset["e_post"][:, ::trim, :]
    true_g = samples_dataset["true_g"]
    sigma_e = samples_dataset["sigma_e"]

    g_samples = pipeline_shear_inference(seed, e_post, true_g, sigma_e, n_samples)

    jnp.save(fpath, g_samples)


if __name__ == "__main__":
    main()
