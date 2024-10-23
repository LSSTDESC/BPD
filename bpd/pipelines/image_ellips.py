from functools import partial
from typing import Callable

import blackjax
import jax.numpy as jnp
from jax import Array
from jax import jit as jjit
from jax import random
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats

from bpd.chains import inference_loop
from bpd.draw import draw_gaussian, draw_gaussian_galsim
from bpd.noise import add_noise
from bpd.pipelines.toy_ellips import do_inference
from bpd.prior import ellip_mag_prior, sample_ellip_prior


def get_target_galaxy_params_simple(
    rng_key: PRNGKeyArray,
    sigma_e: float = 1e-3,
    lf: float = 3.0,
    hlr: float = 1.0,
    x: float = 0.0,  # pixels
    y: float = 0.0,
    g1: float = 0.02,
    g2: float = 0.0,
):
    """Fix all parameters except ellipticity, which come from prior."""
    e = sample_ellip_prior(rng_key, sigma=sigma_e, n=1)
    return {
        "lf": lf,
        "hlr": hlr,
        "e1": e[0, 0],
        "e2": e[0, 1],
        "x": x,
        "y": y,
        "g1": g1,
        "g2": g2,
    }


def get_target_images_single(
    rng_key: PRNGKeyArray,
    n_samples: int,
    single_galaxy_params: dict[str, float],
    psf_hlr: float = 0.7,
    background: float = 1.0,
    slen: int = 53,
    pixel_scale: float = 0.2,
):
    """In this case, we sample multiple noise realizations of the same galaxy."""
    assert "f" in single_galaxy_params and "lf" not in single_galaxy_params

    noiseless = draw_gaussian_galsim(
        **single_galaxy_params,
        pixel_scale=pixel_scale,
        psf_hlr=psf_hlr,
        slen=slen,
    )
    return add_noise(rng_key, noiseless, bg=background, n=n_samples), noiseless


# interim prior
def logprior(
    params: dict[str, Array],
    flux_bds: tuple = (-1.0, 9.0),
    hlr_bds: tuple = (0.01, 5.0),
    sigma_e: float = 1e-2,
    sigma_x: float = 1.0,  # pixels
) -> Array:
    prior = jnp.array(0.0)

    f1, f2 = flux_bds
    prior += stats.uniform.logpdf(params["lf"], f1, f2 - f1)

    h1, h2 = hlr_bds
    prior += stats.uniform.logpdf(params["hlr"], h1, h2 - h1)

    e_mag = jnp.sqrt(params["e1"] ** 2 + params["e2"] ** 2)
    prior += jnp.log(ellip_mag_prior(e_mag, sigma=sigma_e))

    # NOTE: hard-coded assumption that galaxy is always centered in odd-size image.
    prior += stats.norm.logpdf(params["x"], loc=0.0, scale=sigma_x)
    prior += stats.norm.logpdf(params["y"], loc=0.0, scale=sigma_x)

    return prior


def loglikelihood(
    params: dict[str, Array], data: Array, *, draw_fnc: Callable, background: float
):
    _draw_params = {**{"g1": 0.0, "g2": 0.0}, **params}  # function is more general
    lf = _draw_params.pop("lf")
    _draw_params["f"] = 10**lf

    model = draw_fnc(**_draw_params)
    likelihood_pp = stats.norm.logpdf(data, loc=model, scale=jnp.sqrt(background))
    likelihood = jnp.sum(likelihood_pp)
    return likelihood


def logtarget(
    params: dict[str, Array],
    data: Array,
    logprior_fnc: Callable,
    loglikelihood_fnc: Callable,
):
    return logprior_fnc(params) + loglikelihood_fnc(params, data)


def do_inference(
    rng_key: PRNGKeyArray,
    init_positions: dict[str, Array],
    data: Array,
    *,
    logtarget_fnc: Callable,
    is_mass_matrix_diagonal: bool = False,
    n_warmup_steps: int = 500,
    max_num_doublings: int = 5,
    initial_step_size: float = 1e-3,
    target_acceptance_rate: float = 0.80,
    n_samples: int = 100,
):

    key1, key2 = random.split(rng_key)

    _logdensity = partial(logtarget_fnc, data=data)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        _logdensity,
        progress_bar=False,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        max_num_doublings=max_num_doublings,
        initial_step_size=initial_step_size,
        target_acceptance_rate=target_acceptance_rate,
    )

    (init_states, tuned_params), _ = warmup.run(key1, init_positions, n_warmup_steps)

    kernel = blackjax.nuts(_logdensity, **tuned_params).step
    states, _ = inference_loop(key2, init_states, kernel=kernel, n_samples=n_samples)

    return states.position


def pipeline_image_interim_samples(
    rng_key: PRNGKeyArray,
    true_params: dict[str, float],
    target_image: Array,
    *,
    initialization_fnc: Callable,
    sigma_e_int: float = 1e-2,
    n_samples: int = 100,
    max_num_doublings: int = 5,
    initial_step_size: float = 1e-3,
    target_acceptance_rate: float = 0.80,
    n_warmup_steps: int = 500,
    is_mass_matrix_diagonal: bool = False,
    slen: int = 53,
    pixel_scale: float = 0.2,
    psf_hlr: float = 0.7,
    background: float = 1.0,
):

    k1, k2 = random.split(rng_key)

    init_position = initialization_fnc(k1, true_params=true_params, data=target_image)

    _draw_fnc = partial(
        draw_gaussian, pixel_scale=pixel_scale, slen=slen, psf_hlr=psf_hlr
    )
    _loglikelihood = partial(loglikelihood, draw_fnc=_draw_fnc, background=background)
    _logprior = partial(logprior, sigma_e=sigma_e_int)

    _logtarget = partial(
        logtarget, logprior_fnc=_logprior, loglikelihood_fnc=_loglikelihood
    )

    _inference_fnc = partial(
        do_inference,
        logtarget_fnc=_logtarget,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        n_warmup_steps=n_warmup_steps,
        max_num_doublings=max_num_doublings,
        initial_step_size=initial_step_size,
        target_acceptance_rate=target_acceptance_rate,
        n_samples=n_samples,
    )
    _run_inference = jjit(_inference_fnc)

    interim_samples = _run_inference(k2, init_position, target_image)
    return interim_samples
