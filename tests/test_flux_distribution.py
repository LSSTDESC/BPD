import jax.numpy as jnp
import numpy as np
from scipy.stats import skewnorm

from bpd.prior import skewnorm_logpdf


def test_skewnorm_logpdf():
    # Test the skewnorm_logpdf function with some known values
    a = 4.0
    x = jnp.linspace(-5, 5, 100)
    loc = 2.0
    scale = 1.0

    # Calculate the expected value using scipy's skewnorm.pdf
    tvalues = skewnorm.logpdf(x, a, loc=loc, scale=scale)

    # Calculate the actual value using the _skewnorm_logpdf function
    pvalues = skewnorm_logpdf(x, a, loc=loc, scale=scale)
    pvalues = np.asarray(pvalues)

    # Check if the values are close
    assert np.allclose(pvalues, tvalues)
