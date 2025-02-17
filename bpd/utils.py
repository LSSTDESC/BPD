import jax.numpy as jnp
from jax.scipy.stats import uniform
from jax.typing import ArrayLike


def get_snr(im: ArrayLike, background: float) -> float:
    """Calculate the signal-to-noise ratio of an image.

    Args:
        im: 2D image array with no background.
        background: Background level.

    Returns:
        float: The signal-to-noise ratio.
    """
    assert im.ndim == 2
    assert isinstance(background, float) or background.shape == ()
    return jnp.sqrt(jnp.sum(im * im / (background + im)))


def uniform_logpdf(x: ArrayLike, a: float, b: float):
    return uniform.logpdf(x, loc=a, scale=b - a)
