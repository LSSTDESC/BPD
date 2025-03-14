from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, jit, random, vmap
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats

from bpd.chains import run_inference_nuts
from bpd.likelihood import shear_loglikelihood
from bpd.prior import ellip_prior_e1e2, true_ellip_logprior
from bpd.sample import sample_noisy_ellipticities_unclipped


def logtarget_shear(
    g: Array, *, data: Array | dict[str, Array], loglikelihood: Callable, sigma_g: float
):
    loglike = loglikelihood(g, post_params=data)
    logprior = stats.norm.logpdf(g, loc=0.0, scale=sigma_g).sum()
    return logprior + loglike


def pipeline_shear_inference(
    rng_key: PRNGKeyArray,
    post_params: Array,
    init_g: Array,
    *,
    logprior: Callable,
    interim_logprior: Callable,
    n_samples: int,
    initial_step_size: float,
    sigma_g: float = 0.01,
    n_warmup_steps: int = 500,
    max_num_doublings: int = 2,
):
    _loglikelihood = partial(
        shear_loglikelihood, logprior=logprior, interim_logprior=interim_logprior
    )
    _loglikelihood_jitted = jit(_loglikelihood)

    _logtarget = partial(
        logtarget_shear, loglikelihood=_loglikelihood_jitted, sigma_g=sigma_g
    )

    _do_inference = partial(
        run_inference_nuts,
        data=post_params,
        logtarget=_logtarget,
        n_samples=n_samples,
        n_warmup_steps=n_warmup_steps,
        max_num_doublings=max_num_doublings,
        initial_step_size=initial_step_size,
    )

    g_samples = _do_inference(rng_key, init_g)
    return g_samples


def pipeline_shear_inference_simple(
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
    _logprior = lambda e, g: true_ellip_logprior(e, g, sigma_e=sigma_e)
    _interim_logprior = lambda e: jnp.log(ellip_prior_e1e2(e, sigma=sigma_e_int))

    _loglikelihood = partial(
        shear_loglikelihood, logprior=_logprior, interim_logprior=_interim_logprior
    )
    _loglikelihood_jitted = jit(_loglikelihood)

    _logtarget = partial(
        logtarget_shear, loglikelihood=_loglikelihood_jitted, sigma_g=sigma_g
    )

    _do_inference = partial(
        run_inference_nuts,
        data=e_post,
        logtarget=_logtarget,
        n_samples=n_samples,
        n_warmup_steps=n_warmup_steps,
        max_num_doublings=max_num_doublings,
        initial_step_size=initial_step_size,
    )

    return _do_inference(rng_key, init_g)


def logtarget_images(
    params: dict[str, Array],
    data: Array,
    *,
    fixed_params: dict[str, Array],
    logprior_fnc: Callable,
    loglikelihood_fnc: Callable,
):
    return logprior_fnc(params) + loglikelihood_fnc(params, data, fixed_params)


def pipeline_interim_samples_one_galaxy(
    rng_key: PRNGKeyArray,
    true_params: dict[str, float],
    target_image: Array,
    fixed_params: dict[str, float],
    *,
    initialization_fnc: Callable,
    logprior: Callable,
    loglikelihood: Callable,
    n_samples: int = 300,
    max_num_doublings: int = 5,
    initial_step_size: float = 1e-3,
    n_warmup_steps: int = 500,
    is_mass_matrix_diagonal: bool = True,
):
    # Flux and HLR are fixed to truth and not inferred in this function.
    k1, k2 = random.split(rng_key)

    init_position = initialization_fnc(k1, true_params=true_params, data=target_image)

    _logtarget = partial(
        logtarget_images,
        logprior_fnc=logprior,
        loglikelihood_fnc=loglikelihood,
        fixed_params=fixed_params,
    )

    _inference_fnc = partial(
        run_inference_nuts,
        logtarget=_logtarget,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        n_warmup_steps=n_warmup_steps,
        max_num_doublings=max_num_doublings,
        initial_step_size=initial_step_size,
        n_samples=n_samples,
    )
    _run_inference = jit(_inference_fnc)

    interim_samples = _run_inference(k2, init_position, target_image)
    return interim_samples


def logtarget_toy_ellips(
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


def pipeline_toy_ellips(
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

    _logtarget = partial(logtarget_toy_ellips, sigma_m=sigma_m, sigma_e_int=sigma_e_int)

    keys2 = random.split(k2, n_gals)
    _do_inference_jitted = jit(
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
