import jax.numpy as jnp
from jax import Array, random, vmap
from jax._src.prng import PRNGKeyArray
from jax.numpy.linalg import norm
from jax.typing import ArrayLike


def ellip_mag_prior(e_mag: ArrayLike, sigma: float) -> ArrayLike:
    """Prior for the magnitude of the ellipticity with domain (0, 1).

    This distribution is slightly modified from Gary's 2013 paper on Bayesian shear inference.

    Importantly, the paper did not include an additional factor of |e| that is needed
    make this the correct expression for a Gaussian in `polar` coordinates. We include
    this factor in this equation. This blog post is helpful: https://andrewcharlesjones.github.io/journal/rayleigh.html

    The additional factor of (1-e^2)^2 introduced by Gary guarantees differentiability
    at e = 0 and e = 1.
    """

    # norm from Mathematica
    _norm = -4 * sigma**4 + sigma**2 + 8 * sigma**6 * (1 - jnp.exp(-1 / (2 * sigma**2)))
    return (1 - e_mag**2) ** 2 * e_mag * jnp.exp(-(e_mag**2) / (2 * sigma**2)) / _norm


def ellip_prior_e1e2(e: Array, sigma: float) -> ArrayLike:
    """Prior on e1, e2 using Gary's prior for magnitude. Includes Jacobian factor: `|e|`"""
    e_mag = norm(e, axis=-1)

    _norm1 = (
        -4 * sigma**4 + sigma**2 + 8 * sigma**6 * (1 - jnp.exp(-1 / (2 * sigma**2)))
    )
    _norm2 = 2 * jnp.pi  # from f(\beta) and Jacobian
    _norm = _norm1 * _norm2

    # jacobian factor also cancels `e_mag` term below
    return (1 - e_mag**2) ** 2 * jnp.exp(-(e_mag**2) / (2 * sigma**2)) / _norm


def sample_mag_ellip_prior(
    rng_key: PRNGKeyArray, sigma: float, n: int = 1, n_bins: int = 1_000_000
):
    """Sample n points from Gary's ellipticity magnitude prior."""
    # this part could be cached
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


def scalar_shear_transformation(e: Array, g: Array):
    """Transform elliptiticies by a fixed shear (scalar version).

    The transformation we used is equation 3.4b in Seitz & Schneider (1997).

    NOTE: This function is meant to be vmapped later.
    """
    assert e.shape == (2,) and g.shape == (2,)

    e1, e2 = e
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp + g_comp) / (1 + g_comp.conjugate() * e_comp)
    return jnp.array([e_prime.real, e_prime.imag])


def scalar_inv_shear_transformation(e: Array, g: Array):
    """Same as above but the inverse."""
    assert e.shape == (2,) and g.shape == (2,)
    e1, e2 = e
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp - g_comp) / (1 - g_comp.conjugate() * e_comp)
    return jnp.array([e_prime.real, e_prime.imag])


# batched
shear_transformation = vmap(scalar_shear_transformation, in_axes=(0, None))
inv_shear_transformation = vmap(scalar_inv_shear_transformation, in_axes=(0, None))

# useful for jacobian later
inv_shear_func1 = lambda e, g: scalar_inv_shear_transformation(e, g)[0]
inv_shear_func2 = lambda e, g: scalar_inv_shear_transformation(e, g)[1]


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
