from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, random, vmap
from jax import jit as jjit
from jax._src.prng import PRNGKeyArray

from bpd.chains import run_inference_nuts
from bpd.prior import (
    ellip_prior_e1e2,
    sample_noisy_ellipticities_unclipped,
)


def logtarget(
    e_sheared: Array,
    *,
    data: Array,  # renamed from `e_obs` for comptability with `do_inference_nuts`
    sigma_m: float,
    sigma_e_int: float,
):
    e_obs = data
    assert e_sheared.shape == (2,) and e_obs.shape == (2,)

    prior = jnp.log(ellip_prior_e1e2(e_sheared, sigma=sigma_e_int))
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

    e_obs, e_sheared, _ = sample_noisy_ellipticities_unclipped(
        k1, g=true_g, sigma_m=sigma_m, sigma_e=sigma_e, n=n_gals
    )

    _logtarget = partial(logtarget, sigma_m=sigma_m, sigma_e_int=sigma_e_int)

    keys2 = random.split(k2, n_gals)
    _do_inference_jitted = jjit(
        partial(
            run_inference_nuts,
            logtarget=_logtarget,
            n_samples=n_samples_per_gal,
            initial_step_size=max(sigma_e, sigma_m),
            max_num_doublings=max_num_doublings,
            n_warmup_steps=n_warmup_steps,
        )
    )
    _do_inference = vmap(_do_inference_jitted, in_axes=(0, 0, 0))

    # compile
    _ = _do_inference(keys2[:2], e_sheared[:2], e_obs[:2])

    e_post = _do_inference(keys2, e_sheared, e_obs)

    return e_post, e_obs, e_sheared
