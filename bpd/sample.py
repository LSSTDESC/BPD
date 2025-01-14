import jax.numpy as jnp
from jax import Array, random
from jax._src.prng import PRNGKeyArray

from bpd.draw import add_noise, draw_gaussian_galsim
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


def sample_target_galaxy_params_simple(
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


def get_true_params_from_galaxy_params(galaxy_params: dict[str, Array]):
    true_params = {**galaxy_params}
    e1, e2 = true_params.pop("e1"), true_params.pop("e2")
    g1, g2 = true_params.pop("g1"), true_params.pop("g2")

    e1_prime, e2_prime = scalar_shear_transformation(
        jnp.array([e1, e2]), jnp.array([g1, g2])
    )
    true_params["e1"] = e1_prime
    true_params["e2"] = e2_prime

    return true_params  # don't add back g1,g2 as we are not inferring those in interim posterior


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
