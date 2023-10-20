import numpy as np


def get_snr(im, background):
    """Calculate the signal-to-noise ratio of an image.

    Args:
        im: Image array with no background.
        background: Background level.
    """

    return np.sqrt(np.sum(im * im / (background + im)))
