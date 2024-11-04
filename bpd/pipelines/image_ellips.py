from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import Array, random
from jax import jit as jjit
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats

from bpd.chains import run_inference_nuts
from bpd.draw import draw_gaussian, draw_gaussian_galsim
from bpd.noise import add_noise
from bpd.prior import ellip_mag_prior, sample_ellip_prior


def get_target_galaxy_params_simple(
    rng_key: PRNGKeyArray,
    shape_noise: float = 1e-3,
    lf: float = 6.0,
    hlr: float = 1.0,
    x: float = 0.0,  # pixels
    y: float = 0.0,
    g1: float = 0.02,
    g2: float = 0.0,
):
    """Fix all parameters except ellipticity, which come from prior."""
    e = sample_ellip_prior(rng_key, sigma=shape_noise, n=1)
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
    *,
    background: float,
    slen: int,
):
    """In this case, we sample multiple noise realizations of the same galaxy."""
    assert "f" in single_galaxy_params and "lf" not in single_galaxy_params

    noiseless = draw_gaussian_galsim(**single_galaxy_params, slen=slen)
    return add_noise(rng_key, noiseless, bg=background, n=n_samples), noiseless


# interim prior
def logprior(
    params: dict[str, Array],
    *,
    sigma_e: float,
    flux_bds: tuple = (-1.0, 9.0),
    hlr_bds: tuple = (0.01, 5.0),
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


def pipeline_image_interim_samples_one_galaxy(
    rng_key: PRNGKeyArray,
    true_params: dict[str, float],
    target_image: Array,
    *,
    initialization_fnc: Callable,
    sigma_e_int: float,
    n_samples: int = 100,
    max_num_doublings: int = 5,
    initial_step_size: float = 1e-3,
    n_warmup_steps: int = 500,
    is_mass_matrix_diagonal: bool = False,
    slen: int = 53,
    fft_size: int = 256,
    background: float = 1.0,
):
    k1, k2 = random.split(rng_key)

    init_position = initialization_fnc(k1, true_params=true_params, data=target_image)

    _draw_fnc = partial(draw_gaussian, slen=slen, fft_size=fft_size)
    _loglikelihood = partial(loglikelihood, draw_fnc=_draw_fnc, background=background)
    _logprior = partial(logprior, sigma_e=sigma_e_int)

    _logtarget = partial(
        logtarget, logprior_fnc=_logprior, loglikelihood_fnc=_loglikelihood
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
    _run_inference = jjit(_inference_fnc)

    interim_samples = _run_inference(k2, init_position, target_image)
    return interim_samples
