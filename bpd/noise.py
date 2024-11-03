import jax.numpy as jnp
from jax import random
from jax._src.prng import PRNGKeyArray
from jax.typing import ArrayLike


def add_noise(
    rng_key: PRNGKeyArray,
    x: ArrayLike,
    bg: float,
    n: int = 1,
):
    """Produce `n` independent Gaussian noise realizations of a given image `x`.

    NOTE: This function assumes image is background-subtracted and dominated.
    """
    assert isinstance(bg, float) or bg.shape == ()
    x = x.reshape(1, *x.shape)
    x = x.repeat(n, axis=0)
    noise = random.normal(rng_key, shape=x.shape) * jnp.sqrt(bg)
    return x + noise
