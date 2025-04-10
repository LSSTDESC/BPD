from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, random, vmap
from jax._src.prng import PRNGKeyArray
from scipy.stats import skewnorm, truncnorm

from bpd.draw import add_noise, draw_exponential_galsim, draw_gaussian_galsim
from bpd.prior import ellip_mag_prior
from bpd.shear import scalar_shear_transformation, shear_transformation


def sample_mag_ellip_prior(
    rng_key: PRNGKeyArray, sigma: float, n: int = 1, n_bins: int = 1_000_000
):
    """Sample n points from GB's ellipticity magnitude prior."""
    e_mag_array = jnp.linspace(0, 1, n_bins)
    p_array = ellip_mag_prior(e_mag_array, sigma=sigma)
    p_array /= p_array.sum()
    return random.choice(rng_key, e_mag_array, shape=(n,), p=p_array)


def sample_ellip_prior(rng_key: PRNGKeyArray, sigma: float, n: int = 1):
    """Sample n ellipticities isotropic components with Gary's prior for magnitude."""
    key1, key2 = random.split(rng_key, 2)
    e_mag = sample_mag_ellip_prior(key1, sigma=sigma, n=n)
    e_phi = random.uniform(key2, shape=(n,), minval=0, maxval=jnp.pi)
    e1 = e_mag * jnp.cos(2 * e_phi)
    e2 = e_mag * jnp.sin(2 * e_phi)
    return jnp.stack((e1, e2), axis=1)


def sample_noisy_ellipticities_unclipped(
    rng_key: PRNGKeyArray,
    *,
    g: Array,
    sigma_m: float,
    sigma_e: float,
    n: int = 1,
):
    """We sample noisy sheared ellipticities from N(e_int + g, sigma_m^2)"""
    key1, key2 = random.split(rng_key, 2)

    e_int = sample_ellip_prior(key1, sigma=sigma_e, n=n)
    e_sheared = shear_transformation(e_int, g)
    e_obs = random.normal(key2, shape=(n, 2)) * sigma_m + e_sheared.reshape(n, 2)
    return e_obs, e_sheared, e_int


def sample_shapes_and_centroids(
    rng_key: PRNGKeyArray,
    *,
    shape_noise: float,
    g1: float = 0.02,
    g2: float = 0.0,
):
    """Fix parameters except position and ellipticity, which come from a prior.

    * The position is drawn uniformly within a pixel (dither).
    * The ellipticity is drawn from Gary's prior given the shape noise.

    """
    dkey, ekey = random.split(rng_key, 2)

    x, y = random.uniform(dkey, shape=(2,), minval=-0.5, maxval=0.5)
    e = sample_ellip_prior(ekey, sigma=shape_noise, n=1)
    return {
        "e1": e[0, 0],
        "e2": e[0, 1],
        "x": x,
        "y": y,
        "g1": g1,
        "g2": g2,
    }


def sample_galaxy_params_simple(
    rng_key: PRNGKeyArray,
    *,
    shape_noise: float,
    mean_logflux: float,
    sigma_logflux: float,
    mean_loghlr: float,
    sigma_loghlr: float,
    g1: float = 0.02,
    g2: float = 0.0,
) -> dict[str, float]:
    k1, k2, k3 = random.split(rng_key, 3)

    lf = random.normal(k1) * sigma_logflux + mean_logflux
    lhlr = random.normal(k2) * sigma_loghlr + mean_loghlr

    other_params = sample_shapes_and_centroids(
        k3, shape_noise=shape_noise, g1=g1, g2=g2
    )

    return {"lf": lf, "lhlr": lhlr, **other_params}


def sample_galaxy_params_trunc(
    rng_key: PRNGKeyArray,
    *,
    n: int,
    shape_noise: float,
    mean_logflux: float,
    sigma_logflux: float,
    min_logflux: float,
    mean_loghlr: float,
    sigma_loghlr: float,
    g1: float = 0.02,
    g2: float = 0.0,
) -> Array:
    """Same as the above but it generates fluxes a truncted log normal distribution."""
    k1, k2, k3 = random.split(rng_key, 3)

    a = (min_logflux - mean_logflux) / sigma_logflux
    b = np.inf
    rv = truncnorm(a, b, loc=mean_logflux, scale=sigma_logflux)
    rng = np.random.default_rng(np.array(random.key_data(k1)))
    lf = rv.rvs(size=n, random_state=rng)
    lf = jnp.array(lf)

    k2s = random.split(k2, n)
    lhlr = vmap(random.normal)(k2s) * sigma_loghlr + mean_loghlr

    k3s = random.split(k3, n)
    _samples_fnc = partial(
        sample_shapes_and_centroids, shape_noise=shape_noise, g1=g1, g2=g2
    )
    other_params = vmap(_samples_fnc)(k3s)

    return {"lf": lf, "lhlr": lhlr, **other_params}


def sample_galaxy_params_skew(
    rng_key: PRNGKeyArray,
    *,
    n: int,
    shape_noise: float,
    a_logflux: float,
    mean_logflux: float,
    sigma_logflux: float,
    mean_loghlr: float,
    sigma_loghlr: float,
    g1: float = 0.02,
    g2: float = 0.0,
) -> Array:
    """Same as the above but it generates fluxes a truncted log normal distribution."""
    k1, k2, k3 = random.split(rng_key, 3)

    rng = np.random.default_rng(np.array(random.key_data(k1)))
    lf = skewnorm.rvs(
        a=a_logflux, loc=mean_logflux, scale=sigma_logflux, size=n, random_state=rng
    )
    lf = jnp.array(lf)

    k2s = random.split(k2, n)
    lhlr = vmap(random.normal)(k2s) * sigma_loghlr + mean_loghlr

    k3s = random.split(k3, n)
    _samples_fnc = partial(
        sample_shapes_and_centroids, shape_noise=shape_noise, g1=g1, g2=g2
    )
    other_params = vmap(_samples_fnc)(k3s)

    return {"lf": lf, "lhlr": lhlr, **other_params}


def get_true_params_from_galaxy_params(galaxy_params: dict[str, Array]):
    """Utility function to get sheared ellipticities for initializing chains."""
    true_params = {**galaxy_params}

    # ellipticities
    e1, e2 = true_params.pop("e1"), true_params.pop("e2")
    g1, g2 = true_params.pop("g1"), true_params.pop("g2")

    e1_prime, e2_prime = scalar_shear_transformation(
        jnp.array([e1, e2]), jnp.array([g1, g2])
    )
    true_params["e1"] = e1_prime
    true_params["e2"] = e2_prime

    return true_params


def get_target_image_single(
    rng_key: PRNGKeyArray,
    *,
    single_galaxy_params: dict[str, float],
    background: float,
    slen: int,
    draw_type: str,
    n_samples: int = 1,  # single noise realization,
) -> Array:
    """Multiple noise realizations of single galaxy (GalSim)."""
    assert draw_type in ("gaussian", "exponential")

    if draw_type == "gaussian":
        noiseless = draw_gaussian_galsim(**single_galaxy_params, slen=slen)
    elif draw_type == "exponential":
        noiseless = draw_exponential_galsim(**single_galaxy_params, slen=slen)
    else:
        raise NotImplementedError("The galaxy type selected has not been implemented.")

    return add_noise(rng_key, noiseless, bg=background, n=n_samples)


def get_target_images(
    rng_key: PRNGKeyArray,
    galaxy_params: dict[str, Array],
    *,
    background: float,
    slen: int,
    draw_type: str,
) -> Array:
    """Single noise realization of multiple galaxies (GalSim)."""
    n_gals = galaxy_params["f"].shape[0]
    nkeys = random.split(rng_key, n_gals)

    target_images = []
    for ii in range(n_gals):
        _params = {k: v[ii].item() for k, v in galaxy_params.items()}
        one_image = get_target_image_single(
            nkeys[ii],
            single_galaxy_params=_params,
            background=background,
            slen=slen,
            n_samples=1,
            draw_type=draw_type,
        )
        assert one_image.shape == (1, slen, slen)
        target_images.append(jnp.asarray(one_image))

    return jnp.concatenate(target_images, axis=0)
