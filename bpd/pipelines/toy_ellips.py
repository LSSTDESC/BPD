from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, random, vmap
from jax import jit as jjit
from jax._src.prng import PRNGKeyArray

from bpd.chains import run_inference_nuts
from bpd.prior import ellip_mag_prior, sample_synthetic_sheared_ellips_unclipped


def logtarget(
    e_sheared: Array,
    *,
    data: Array,  # renamed from `e_obs` for comptability with `do_inference_nuts`
    sigma_m: float,
    interim_prior: Callable,
):
    e_obs = data
    assert e_sheared.shape == (2,) and e_obs.shape == (2,)

    # ignore angle prior assumed uniform
    # prior enforces magnitude < 1.0 for posterior samples
    e_sheared_mag = jnp.sqrt(e_sheared[0] ** 2 + e_sheared[1] ** 2)
    prior = jnp.log(interim_prior(e_sheared_mag))

    likelihood = jnp.sum(jsp.stats.norm.logpdf(e_obs, loc=e_sheared, scale=sigma_m))
    return prior + likelihood


def pipeline_toy_ellips_samples(
    key: PRNGKeyArray,
    *,
    g1: float,
    g2: float,
    sigma_e: float,
    sigma_e_int: float,
    sigma_m: float,
    n_gals: int,
    n_samples_per_gal: int,
    n_warmup_steps: int = 500,
    max_num_doublings: int = 2,
):
    k1, k2 = random.split(key)

    true_g = jnp.array([g1, g2])

    e_obs, e_sheared, _ = sample_synthetic_sheared_ellips_unclipped(
        k1, true_g, n=n_gals, sigma_m=sigma_m, sigma_e=sigma_e
    )

    interim_prior = partial(ellip_mag_prior, sigma=sigma_e_int)

    _logtarget = partial(logtarget, sigma_m=sigma_m, interim_prior=interim_prior)

    keys2 = random.split(k2, n_gals)
    _do_inference_jitted = jjit(
        partial(
            run_inference_nuts,
            logtarget=_logtarget,
            n_samples=n_samples_per_gal,
            initial_step_size=sigma_e,
            interim_prior=interim_prior,
            n_warmup_steps=n_warmup_steps,
            max_num_doublins=max_num_doublings,
        )
    )
    _do_inference = vmap(_do_inference_jitted, in_axes=(0, 0, 0))

    # compile
    _ = _do_inference(keys2[:2], e_sheared[:2], e_obs[:2])

    e_post = _do_inference(keys2, e_sheared, e_obs)

    return e_post, e_obs, e_sheared
