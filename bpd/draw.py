import numpy as np


def add_noise(
    x: np.ndarray,
    bg: float,
    rng: np.random.Generator = np.random.default_rng(42),
    n=1,
    noise_factor=1,
):
    """Produce `n` independent Gaussian noise realizations of a given image `x`.

    NOTE: This function assumes image is background-subtracted and dominated.
    """
    assert isinstance(bg, float) or bg.shape == ()
    x = x.reshape(1, *x.shape)
    x = x.repeat(n, axis=0)
    noise = rng.normal(loc=0, scale=np.sqrt(bg), size=x.shape) * noise_factor
    return x + noise
