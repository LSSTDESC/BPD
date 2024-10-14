from functools import partial
from typing import Callable

import blackjax
from jax import Array
from jax import jit as jjit
from jax import random
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats

from bpd.chains import inference_loop
from bpd.likelihood import shear_loglikelihood
from bpd.prior import ellip_mag_prior


def logtarget_density(g: Array, e_post: Array, loglikelihood: Callable):
    loglike = loglikelihood(g, e_post)
    logprior = stats.uniform.logpdf(g, -0.1, 0.2).sum()
    return logprior + loglike


def do_inference(
    rng_key: PRNGKeyArray, init_g: Array, logtarget: Callable, n_samples: int
):
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
    rng_key: PRNGKeyArray,
    e_post: Array,
    true_g: Array,
    sigma_e: float,
    sigma_e_int: float,
    n_samples: int,
):
    prior = partial(ellip_mag_prior, sigma=sigma_e)
    interim_prior = partial(ellip_mag_prior, sigma=sigma_e_int)

    # NOTE: jit must be applied without `e_post` in partial!
    _loglikelihood = jjit(
        partial(shear_loglikelihood, prior=prior, interim_prior=interim_prior)
    )
    _logtarget = partial(logtarget_density, loglikelihood=_loglikelihood, e_post=e_post)

    _do_inference = partial(do_inference, logtarget=_logtarget, n_samples=n_samples)
    g_samples = _do_inference(rng_key, true_g)

    return g_samples
