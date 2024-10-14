#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from functools import partial
from typing import Callable

import blackjax
import click
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit as jjit
from jax import random, vmap
from jax.typing import ArrayLike

from bpd import DATA_DIR
from bpd.chains import inference_loop
from bpd.io import save_dataset
from bpd.prior import ellip_mag_prior, sample_synthetic_sheared_ellips_unclipped


def log_target(
    e_sheared: ArrayLike,
    e_obs: ArrayLike,
    sigma_m: float,
    interim_prior: Callable = None,
):
    assert e_sheared.shape == (2,) and e_obs.shape == (2,)

    # ignore angle because flat
    # prior enforces magnitude < 1.0 for posterior samples
    e_sheared_mag = jnp.sqrt(e_sheared[0] ** 2 + e_sheared[1] ** 2)
    prior = jnp.log(interim_prior(e_sheared_mag))

    likelihood = jnp.sum(jsp.stats.norm.logpdf(e_obs, loc=e_sheared, scale=sigma_m))
    return prior + likelihood


def do_inference(
    rng_key,
    init_positions: ArrayLike,
    e_obs: ArrayLike,
    sigma_m: float,
    sigma_e: float,
    interim_prior: Callable,
    k: int,
):
    _logtarget = partial(
        log_target, e_obs=e_obs, sigma_m=sigma_m, interim_prior=interim_prior
    )

    key1, key2 = random.split(rng_key)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        _logtarget,
        progress_bar=False,
        is_mass_matrix_diagonal=True,
        max_num_doublings=2,
        initial_step_size=sigma_e,
        target_acceptance_rate=0.80,
    )

    (init_states, tuned_params), _ = warmup.run(key1, init_positions, 500)
    kernel = blackjax.nuts(_logtarget, **tuned_params).step
    states, _ = inference_loop(key2, init_states, kernel=kernel, n_samples=k)
    return states.position


def pipeline_toy_ellips_samples(
    seed: int,
    g1: float,
    g2: float,
    sigma_e: float,
    sigma_m: float,
    n_samples: int,
    k: int,
    sigma_e_int: float | None = None,  # interim posterior sigma, default 2 * sigma_e
):
    rng_key = random.key(seed)

    k1, k2 = random.split(rng_key)

    true_g = jnp.array([g1, g2])

    e_obs, e_sheared, _ = sample_synthetic_sheared_ellips_unclipped(
        k1, true_g, n=n_samples, sigma_m=sigma_m, sigma_e=sigma_e
    )

    _sigma_e_int = sigma_e_int if sigma_e_int is not None else 2 * sigma_e
    interim_prior = partial(ellip_mag_prior, sigma=_sigma_e_int)

    keys2 = random.split(k2, n_samples)
    _do_inference_jitted = jjit(
        partial(
            do_inference,
            sigma_e=sigma_e,
            sigma_m=sigma_m,
            interim_posterior=interim_prior,
            k=k,
        )
    )
    _do_inference = vmap(_do_inference_jitted, in_axes=(0, 0, 0))

    # compile
    _ = _do_inference(keys2[:2], e_sheared[:2], e_obs[:2])

    e_post = _do_inference(keys2, e_sheared, e_obs)

    return e_post, e_obs, e_sheared


@click.command()
@click.option("--tag", type=str, required=True)
@click.option("--seed", type=int, default=42)
@click.option("--g1", type=float, default=0.02)
@click.option("--g2", type=float, default=0.0)
@click.option("-n", "--n-samples", type=int, default=10_000, help="# of gals")
@click.option("--k", type=int, default=10, help="# int. posterior samples per galaxy.")
@click.option("--shape-noise", type=float, default=1e-3)  # > OK, 1e-4 not OK :(
@click.option("--obs-noise", type=float, default=1e-4)
@click.option("--overwrite", type=bool, default=False)
def main(
    tag: str,
    seed: int,
    g1: float,
    g2: float,
    n_samples: int,
    k: int,
    shape_noise: float,
    obs_noise: float,
    overwrite: bool,
):
    dirpath = DATA_DIR / "cache_chains" / tag
    fpath = DATA_DIR / "cache_chains" / dirpath / f"e_post_{seed}.npz"
    if not dirpath.exists():
        dirpath.mkdir(exist_ok=False)
    if fpath.exists():
        assert overwrite

    e_post, e_obs, e_sheared = pipeline_toy_ellips_samples(
        seed, g1, g2, sigma_e=shape_noise, sigma_m=obs_noise, n_samples=n_samples, k=k
    )

    ds = {
        "e_post": e_post,
        "e_obs": e_obs,
        "e_sheared": e_sheared,
        "true_g": jnp.array([g1, g2]),
        "sigma_e": jnp.array(shape_noise),
        "sigma_m": jnp.array(obs_noise),
        "seed": jnp.array(seed),
    }
    save_dataset(ds, fpath, overwrite=overwrite)


if __name__ == "__main__":
    main()
