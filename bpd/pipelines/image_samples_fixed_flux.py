import jax.numpy as jnp
from jax import Array, random
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats

from bpd.prior import ellip_mag_prior, sample_ellip_prior


def get_target_galaxy_params_simple(
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


# interim prior
def logprior(
    params: dict[str, Array], *, sigma_e: float, sigma_x: float = 0.5
) -> Array:
    prior = jnp.array(0.0)

    e_mag = jnp.sqrt(params["e1"] ** 2 + params["e2"] ** 2)
    prior += jnp.log(ellip_mag_prior(e_mag, sigma=sigma_e))

    # NOTE: hard-coded assumption that galaxy is in center-pixel within odd-size image.
    # sigma_x in units of pixels.
    prior += stats.norm.logpdf(params["x"], loc=0.0, scale=sigma_x)
    prior += stats.norm.logpdf(params["y"], loc=0.0, scale=sigma_x)

    return prior
