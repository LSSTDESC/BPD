from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import Array, random
from jax import jit as jjit
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats

from bpd.chains import run_inference_nuts
from bpd.draw import draw_gaussian_galsim
from bpd.noise import add_noise
from bpd.prior import ellip_mag_prior, scalar_shear_transformation


# interim prior
def logprior(
    params: dict[str, Array],
    *,
    sigma_e: float,
    sigma_x: float = 0.5,  # pixels
    flux_bds: tuple = (-1.0, 9.0),
    hlr_bds: tuple = (0.01, 5.0),
) -> Array:
    prior = jnp.array(0.0)

    f1, f2 = flux_bds
    prior += stats.uniform.logpdf(params["lf"], f1, f2 - f1)

    h1, h2 = hlr_bds
    prior += stats.uniform.logpdf(params["hlr"], h1, h2 - h1)

    e_mag = jnp.sqrt(params["e1"] ** 2 + params["e2"] ** 2)
    prior += jnp.log(ellip_mag_prior(e_mag, sigma=sigma_e))

    # NOTE: hard-coded assumption that galaxy is in center-pixel within odd-size image.
    # sigma_x in units of pixels.
    prior += stats.norm.logpdf(params["x"], loc=0.0, scale=sigma_x)
    prior += stats.norm.logpdf(params["y"], loc=0.0, scale=sigma_x)

    return prior


def loglikelihood(
    params: dict[str, Array], data: Array, *, draw_fnc: Callable, background: float
):
    # NOTE: draw_fnc should already contain `f` and `hlr` as constant arguments.
    _draw_params = {**{"g1": 0.0, "g2": 0.0}, **params}  # function is more general
    model = draw_fnc(**_draw_params)

    likelihood_pp = stats.norm.logpdf(data, loc=model, scale=jnp.sqrt(background))
    likelihood = jnp.sum(likelihood_pp)
    return likelihood


def logtarget(
    params: dict[str, Array],
    data: Array,
    *,
    logprior_fnc: Callable,
    loglikelihood_fnc: Callable,
):
    return logprior_fnc(params) + loglikelihood_fnc(params, data)


def get_true_params_from_galaxy_params(galaxy_params: dict[str, Array]):
    true_params = {**galaxy_params}
    e1, e2 = true_params.pop("e1"), true_params.pop("e2")
    g1, g2 = true_params.pop("g1"), true_params.pop("g2")

    e1_prime, e2_prime = scalar_shear_transformation((e1, e2), (g1, g2))
    true_params["e1"] = e1_prime
    true_params["e2"] = e2_prime

    return true_params  # don't add g1,g2 back as we are not inferring those in interim posterior


def get_target_images_single(
    rng_key: PRNGKeyArray,
    *,
    single_galaxy_params: dict[str, float],
    background: float,
    slen: int,
    n_samples: int = 1,  # single noise realization
):
    """Multiple noise realizations of single galaxy (GalSim)."""
    noiseless = draw_gaussian_galsim(**single_galaxy_params, slen=slen)
    return add_noise(rng_key, noiseless, bg=background, n=n_samples)


def get_target_images(
    rng_key: PRNGKeyArray,
    galaxy_params: dict[str, Array],
    *,
    background: float,
    slen: int,
):
    """Single noise realization of multiple galaxies (GalSim)."""
    n_gals = galaxy_params["f"].shape[0]
    nkeys = random.split(rng_key, n_gals)

    target_images = []
    for ii in range(n_gals):
        _params = {k: v[ii].item() for k, v in galaxy_params.items()}
        noiseless = draw_gaussian_galsim(**_params, slen=slen)
        target_image = add_noise(nkeys[ii], noiseless, bg=background, n=1)
        assert target_image.shape == (1, slen, slen)
        target_images.append(target_image)

    return jnp.concatenate(target_images, axis=0)


def pipeline_interim_samples_one_galaxy(
    rng_key: PRNGKeyArray,
    true_params: dict[str, float],
    target_image: Array,
    *,
    initialization_fnc: Callable,
    draw_fnc: Callable,
    logprior: Callable,
    sigma_e_int: float,
    n_samples: int = 100,
    max_num_doublings: int = 5,
    initial_step_size: float = 1e-3,
    n_warmup_steps: int = 500,
    is_mass_matrix_diagonal: bool = True,
    background: float = 1.0,
):
    # Flux and HLR are fixed to truth and not inferred in this function.
    k1, k2 = random.split(rng_key)

    init_position = initialization_fnc(k1, true_params=true_params, data=target_image)
    _loglikelihood = partial(loglikelihood, draw_fnc=draw_fnc, background=background)
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