from functools import partial
from typing import Callable

import blackjax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax import jit as jjit
from jax import random, vmap
from jax._src.prng import PRNGKeyArray

from bpd.chains import inference_loop
from bpd.prior import ellip_mag_prior, sample_synthetic_sheared_ellips_unclipped


def log_target(
    e_sheared: Array,
    e_obs: Array,
    sigma_m: float,
    interim_prior: Callable,
):
    assert e_sheared.shape == (2,) and e_obs.shape == (2,)

    # ignore angle prior assumed uniform
    # prior enforces magnitude < 1.0 for posterior samples
    e_sheared_mag = jnp.sqrt(e_sheared[0] ** 2 + e_sheared[1] ** 2)
    prior = jnp.log(interim_prior(e_sheared_mag))

    likelihood = jnp.sum(jsp.stats.norm.logpdf(e_obs, loc=e_sheared, scale=sigma_m))
    return prior + likelihood


def do_inference(
    rng_key: PRNGKeyArray,
    init_positions: Array,
    e_obs: Array,
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
    key: PRNGKeyArray,
    g1: float,
    g2: float,
    sigma_e: float,
    sigma_e_int: float,
    sigma_m: float,
    n_samples: int,
    k: int,
):

    k1, k2 = random.split(key)

    true_g = jnp.array([g1, g2])

    e_obs, e_sheared, _ = sample_synthetic_sheared_ellips_unclipped(
        k1, true_g, n=n_samples, sigma_m=sigma_m, sigma_e=sigma_e
    )

    interim_prior = partial(ellip_mag_prior, sigma=sigma_e_int)

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
