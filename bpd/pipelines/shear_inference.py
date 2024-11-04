from functools import partial
from typing import Callable

from jax import Array
from jax import jit as jjit
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats

from bpd.chains import run_inference_nuts
from bpd.likelihood import shear_loglikelihood
from bpd.prior import ellip_mag_prior


def logtarget_density(g: Array, *, data: Array, loglikelihood: Callable):
    e_post = data  # comptability with `do_inference_nuts`
    loglike = loglikelihood(g, e_post)
    logprior = stats.uniform.logpdf(g, -0.1, 0.2).sum()
    return logprior + loglike


def pipeline_shear_inference(
    rng_key: PRNGKeyArray,
    e_post: Array,
    *,
    true_g: Array,
    sigma_e: float,
    sigma_e_int: float,
    n_samples: int,
    initial_step_size: float,
    n_warmup_steps: int = 500,
    max_num_doublings: int = 2,
):
    prior = partial(ellip_mag_prior, sigma=sigma_e)
    interim_prior = partial(ellip_mag_prior, sigma=sigma_e_int)

    # NOTE: jit must be applied without `e_post` in partial!
    _loglikelihood = jjit(
        partial(shear_loglikelihood, prior=prior, interim_prior=interim_prior)
    )
    _logtarget = partial(logtarget_density, loglikelihood=_loglikelihood)

    _do_inference = partial(
        run_inference_nuts,
        data=e_post,
        logtarget=_logtarget,
        n_samples=n_samples,
        n_warmup_steps=n_warmup_steps,
        max_num_doublings=max_num_doublings,
        intial_step_size=initial_step_size,
    )

    g_samples = _do_inference(rng_key, true_g)

    return g_samples
