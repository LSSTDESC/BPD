from functools import partial

import jax.numpy as jnp
import jax_galsim as xgalsim
import numpy as np
import pytest
from jax import jit, random
from jax_galsim import GSParams

from bpd.sample import sample_ellip_prior
from bpd.shear import (
    inv_shear_transformation,
    scalar_inv_shear_transformation,
    scalar_shear_transformation,
    shear_transformation,
)


# different from our `draw.py` functions as those are for forward model
# but will still demonstrate the test
def _draw_gaussian(
    *,
    f: float,
    hlr: float,
    e1: float,
    e2: float,
    x: float,
    y: float,
    g1: float,
    g2: float,
    slen: int,
    fft_size: int,
    psf_hlr: float = 0.7,
    pixel_scale: float = 0.2,
):
    gsparams = GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)

    gal = xgalsim.Gaussian(flux=f, half_light_radius=hlr)
    gal = gal.shear(g1=e1, g2=e2)
    gal = gal.shear(g1=g1, g2=g2)

    psf = xgalsim.Gaussian(flux=1.0, half_light_radius=psf_hlr)
    gal_conv = xgalsim.Convolve([gal, psf]).withGSParams(gsparams)
    image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset=(x, y))
    return image.array


def test_scalar_inverse():
    # scalar version
    ellips = (0.0, 0.1, 0.2, -0.1, -0.2)
    shears = (0.0, -0.01, 0.01, -0.02, 0.02)
    for e1 in ellips:
        for e2 in ellips:
            for g1 in shears:
                for g2 in shears:
                    e = jnp.array([e1, e2])
                    g = jnp.array([g1, g2])
                    e_trans = scalar_shear_transformation(e, g)
                    e_new = scalar_inv_shear_transformation(e_trans, g)
                    np.testing.assert_allclose(e_new, e, atol=1e-15)


@pytest.mark.parametrize("seed", [1234, 4567])
def test_transformation(seed):
    shears = (0.0, -0.01, 0.01, -0.02, 0.02)

    k = random.key(seed)
    e_samples = sample_ellip_prior(k, sigma=0.3, n=100)
    assert e_samples.shape == (100, 2)

    for g1 in shears:
        for g2 in shears:
            g = jnp.array([g1, g2])
            e_trans_samples = shear_transformation(e_samples, g)
            e_new = inv_shear_transformation(e_trans_samples, g)
            assert e_new.shape == (100, 2)
            np.testing.assert_allclose(e_new, e_samples)


def test_image_shear_commute():
    """Test that the shear operation on galsim commutes with the analytical shear transformation."""
    ellips = (0.0, 0.1, 0.2, -0.1, -0.2)
    shears = (0.0, -0.01, 0.01, -0.02, 0.02, 0.05, 0.1, -0.05, -0.1)
    f = 1e3
    hlr = 0.9
    x, y = (1, 1)

    draw_jitted = jit(partial(_draw_gaussian, slen=53, fft_size=256))
    for e1 in ellips:
        for e2 in ellips:
            for g1 in shears:
                for g2 in shears:
                    # shear
                    e = jnp.array([e1, e2])
                    g = jnp.array([g1, g2])
                    (e1_p, e2_p) = scalar_shear_transformation(e, g)

                    im1 = draw_jitted(f=f, hlr=hlr, e1=e1, e2=e2, g1=0, g2=0, x=x, y=y)
                    im2 = draw_jitted(
                        f=f, hlr=hlr, e1=e1, e2=e2, g1=g1, g2=g2, x=x, y=y
                    )
                    im3 = draw_jitted(
                        f=f, hlr=hlr, e1=e1_p, e2=e2_p, g1=0, g2=0, x=x, y=y
                    )

                    # rtol is 0 because image contains lots of 0s
                    np.testing.assert_allclose(im2, im3, rtol=0, atol=1e-10)

                    if not (g1 == 0 and g2 == 0):
                        assert not np.allclose(im1, im2, rtol=0, atol=1e-10)
                        assert not np.allclose(im1, im3, rtol=0, atol=1e-10)
