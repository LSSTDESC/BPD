import jax.numpy as jnp
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
