import numpy as np


def get_snr(im, background: float) -> float:
    """Calculate the signal-to-noise ratio of an image.

    Args:
        im: Image array with no background.
        background: Background level.
    """
    assert im.shape == 2
    assert isinstance(background, float) or background.shape == ()
    return np.sqrt(np.sum(im * im / (background + im)))
