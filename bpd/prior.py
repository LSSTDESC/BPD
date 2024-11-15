import jax.numpy as jnp
from jax import Array, random
from jax.numpy.linalg import norm


def ellip_mag_prior(e, sigma: float):
    """Unnormalized Prior for the magnitude of the ellipticity, domain is (0, 1)

    This distribution is taken from Gary's 2013 paper on Bayesian shear inference.
    The additional factor on the truncated Gaussian guarantees differentiability
    at e = 0 and e = 1.

    Gary uses 0.3 as a default level of shape noise.
    """
    return (1 - e**2) ** 2 * jnp.exp(-(e**2) / (2 * sigma**2))


def sample_mag_ellip_prior(rng_key, sigma: float, n: int = 1, n_bins: int = 1_000_000):
    """Sample n points from Gary's ellipticity magnitude prior."""
    # this part could be cached
    e_array = jnp.linspace(0, 1, n_bins)
    p_array = ellip_mag_prior(e_array, sigma=sigma)
    p_array /= jnp.sum(p_array)

    return random.choice(rng_key, e_array, shape=(n,), p=p_array)


def sample_ellip_prior(rng_key, sigma: float, n: int = 1):
    """Sample n ellipticities isotropic components with Gary's prior from magnitude."""
    key1, key2 = random.split(rng_key, 2)
    e_mag = sample_mag_ellip_prior(key1, sigma=sigma, n=n)
    e_phi = random.uniform(key2, shape=(n,), minval=0, maxval=2 * jnp.pi)
    e1 = e_mag * jnp.cos(2 * e_phi)
    e2 = e_mag * jnp.sin(2 * e_phi)
    return jnp.stack((e1, e2), axis=1)


def scalar_shear_transformation(e: tuple[float, float], g: tuple[float, float]):
    """Transform elliptiticies by a fixed shear (scalar version).

    The transformation we used is equation 3.4b in Seitz & Schneider (1997).

    NOTE: This function is meant to be vmapped later.
    """
    e1, e2 = e
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp + g_comp) / (1 + g_comp.conjugate() * e_comp)
    return e_prime.real, e_prime.imag


def scalar_inv_shear_transformation(e: tuple[float, float], g: tuple[float, float]):
    """Same as above but the inverse."""
    e1, e2 = e
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp - g_comp) / (1 - g_comp.conjugate() * e_comp)
    return e_prime.real, e_prime.imag


# useful for jacobian later, only need 2 grads really
inv_shear_func1 = lambda e, g: scalar_inv_shear_transformation(e, g)[0]
inv_shear_func2 = lambda e, g: scalar_inv_shear_transformation(e, g)[1]


def shear_transformation(e: Array, g: tuple[float, float]):
    """Transform elliptiticies by a fixed shear.

    The transformation we used is equation 3.4b in Seitz & Schneider (1997).
    """
    e1, e2 = e[..., 0], e[..., 1]
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp + g_comp) / (1 + g_comp.conjugate() * e_comp)
    return jnp.stack([e_prime.real, e_prime.imag], axis=-1)


def inv_shear_transformation(e: Array, g: tuple[float, float]):
    """Same as above but the inverse."""
    e1, e2 = e[..., 0], e[..., 1]
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp - g_comp) / (1 - g_comp.conjugate() * e_comp)
    return jnp.stack([e_prime.real, e_prime.imag], axis=-1)


# get synthetic measured sheared ellipticities
def sample_synthetic_sheared_ellips_unclipped(
    rng_key,
    g: tuple[float, float],
    n: int,
    sigma_m: float,
    sigma_e: float,
):
    """We sample sheared ellipticities from N(e_int + g, sigma_m^2)"""
    key1, key2 = random.split(rng_key, 2)

    e_int = sample_ellip_prior(key1, sigma=sigma_e, n=n)
    e_sheared = shear_transformation(e_int, g)
    e_obs = random.normal(key2, shape=(n, 2)) * sigma_m + e_sheared.reshape(n, 2)
    return e_obs, e_sheared, e_int


def sample_synthetic_sheared_ellips_clipped(
    rng_key,
    g: tuple[float, float],
    sigma_m: float,
    sigma_e: float,
    n: int = 1,
    m: int = 10,
    e_tol: float = 0.99999,
):
    """We sample sheared ellipticities from N(e_int + g, sigma_m^2)

    The prior for galaxies is Gary's model for the ellipticity magnitude.

    We generate `m` samples per intrinsic ellipticity.
    """
    key1, key2 = random.split(rng_key, 2)

    e_int = sample_ellip_prior(key1, sigma=sigma_e, n=n)
    e_sheared = shear_transformation(e_int, g)
    e_obs = random.normal(key2, shape=(n, m, 2)) * sigma_m + e_sheared.reshape(n, 1, 2)

    # clip magnitude to < 1
    # preserve angle after noise added when clipping
    beta = jnp.arctan2(e_obs[:, :, 1], e_obs[:, :, 0]) / 2
    e_obs_mag = norm(e_obs, axis=-1)
    e_obs_mag = jnp.clip(e_obs_mag, 0, e_tol)  # otherwise likelihood explodes

    final_eobs1 = e_obs_mag * jnp.cos(2 * beta)
    final_eobs2 = e_obs_mag * jnp.sin(2 * beta)
    final_eobs = jnp.stack([final_eobs1, final_eobs2], axis=2)

    return final_eobs, e_int, e_sheared
