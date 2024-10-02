"""Minimal amount of code that checks jax-galsim is working correctly."""

import jax
import jax.numpy as jnp
import jax_galsim as galsim

HLR = 1.0
PSF_HLR = 0.7
SLEN = 101
PIXEL_SCALE = 0.2


GSPARAMS = galsim.GSParams(minimum_fft_size=512, maximum_fft_size=512)


@jax.jit
@jax.vmap
def draw_gal(lf):
    gal = galsim.Gaussian(flux=10**lf, half_light_radius=HLR)
    psf = galsim.Gaussian(flux=1.0, half_light_radius=PSF_HLR)
    gal_conv = galsim.Convolve([gal, psf]).withGSParams(GSPARAMS)
    image = gal_conv.drawImage(nx=SLEN, ny=SLEN, scale=PIXEL_SCALE)
    return image.array


a = draw_gal(jnp.array([1e4, 1e5]))
assert a.ndim == 3
assert a.shape == (2, 101, 101)
