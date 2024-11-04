from functools import partial

import numpy as np
import pytest
from jax import jit as jjit
from jax import random

from bpd.draw import draw_gaussian
from bpd.prior import (
    inv_shear_transformation,
    sample_ellip_prior,
    scalar_inv_shear_transformation,
    scalar_shear_transformation,
    shear_transformation,
)


def test_scalar_inverse():
    # scalar version
    ellips = (0.0, 0.1, 0.2, -0.1, -0.2)
    shears = (0.0, -0.01, 0.01, -0.02, 0.02)
    for e1 in ellips:
        for e2 in ellips:
            for g1 in shears:
                for g2 in shears:
                    e_trans = scalar_shear_transformation((e1, e2), (g1, g2))
                    e1_new, e2_new = scalar_inv_shear_transformation(e_trans, (g1, g2))

                    e_array = np.array([e1, e2])
                    e_new_array = np.array([e1_new, e2_new])
                    np.testing.assert_allclose(e_new_array, e_array, atol=1e-15)


@pytest.mark.parametrize("seed", [1234, 4567])
def test_transformation(seed):
    shears = (0.0, -0.01, 0.01, -0.02, 0.02)

    k = random.key(seed)
    e_samples = sample_ellip_prior(k, sigma=0.3, n=100)
    assert e_samples.shape == (100, 2)

    for g1 in shears:
        for g2 in shears:
            e_trans_samples = shear_transformation(e_samples, (g1, g2))
            e_new = inv_shear_transformation(e_trans_samples, (g1, g2))
            assert e_new.shape == (100, 2)
            np.testing.assert_allclose(e_new, e_samples)


def test_image_shear_commute():
    """Test that the shear operation on galsim commutes with the analytical shear transformation."""
    ellips = (0.0, 0.1, 0.2, -0.1, -0.2)
    shears = (0.0, -0.01, 0.01, -0.02, 0.02, 0.05, 0.1, -0.05, -0.1)
    f = 1e3
    hlr = 0.9
    x, y = (1, 1)

    draw_jitted = jjit(partial(draw_gaussian), slen=53, fft_size=256)
    for e1 in ellips:
        for e2 in ellips:
            for g1 in shears:
                for g2 in shears:
                    (e1_p, e2_p) = scalar_shear_transformation((e1, e2), (g1, g2))
                    im1 = draw_jitted(
                        f=f, hlr=hlr, e1=e1, e2=e2, g1=g1, g2=g2, x=x, y=y
                    )
                    im2 = draw_jitted(
                        f=f, hlr=hlr, e1=e1_p, e2=e2_p, g1=0.0, g2=0.0, x=x, y=y
                    )
                    im3 = draw_jitted(
                        f=f, hlr=hlr, e1=e1, e2=e2, g1=0.0, g2=0.0, x=x, y=y
                    )

                    np.testing.assert_allclose(im1, im2, rtol=1e-6, atol=1e-10)

                    if not (g1 == 0 and g2 == 0):
                        assert not np.allclose(im3, im1, rtol=1e-6, atol=1e-10)
