from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import Array, jit
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats

from bpd.chains import run_inference_nuts
from bpd.likelihood import shear_loglikelihood, true_ellip_logprior
from bpd.prior import ellip_prior_e1e2


def logtarget_density(
    g: Array, *, data: Array, loglikelihood: Callable, sigma_g: float = 0.01
):
    loglike = loglikelihood(g, post_params=data)
    logprior = stats.norm.logpdf(g, loc=0.0, scale=sigma_g).sum()
    return logprior + loglike


def _logprior(post_params: dict[str, Array], g: Array, *, sigma_e: float):
    e_post = post_params["e1e2"]
    return true_ellip_logprior(e_post, g, sigma_e=sigma_e)


def _interim_logprior(post_params: dict[str, Array], sigma_e_int: float):
    e_post = post_params["e1e2"]
    return jnp.log(ellip_prior_e1e2(e_post, sigma=sigma_e_int))


def pipeline_shear_inference_ellipticities(
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
