import galsim
import jax.numpy as jnp
import jax_galsim as xgalsim
from jax import random
from jax._src.prng import PRNGKeyArray
from jax.typing import ArrayLike
from jax_galsim import GSParams


def draw_gaussian(
    *,
    f: float,
    hlr: float,
    e1: float,
    e2: float,
    x: float,  # pixels
    y: float,
    slen: int,
    fft_size: int,  # rule of thumb: at least 4 times `slen`
    psf_fwhm: float = 0.8,
    pixel_scale: float = 0.2,
):
    gsparams = GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)

    gal = xgalsim.Gaussian(flux=f, half_light_radius=hlr)
    gal = gal.shear(g1=e1, g2=e2)

    psf = xgalsim.Gaussian(flux=1.0, fwhm=psf_fwhm)
    gal_conv = xgalsim.Convolve([gal, psf]).withGSParams(gsparams)
    image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset=(x, y))
    return image.array


def draw_gaussian_galsim(
    *,
    f: float,
    hlr: float,
    e1: float,
    e2: float,
    g1: float,
    g2: float,
    x: float,  # pixels
    y: float,
    slen: int,
    psf_fwhm: float = 0.8,
    pixel_scale: float = 0.2,
):
    gal = galsim.Gaussian(flux=f, half_light_radius=hlr)
    gal = gal.shear(g1=e1, g2=e2)  # intrinsic ellipticity
    gal = gal.shear(g1=g1, g2=g2)

    psf = galsim.Gaussian(flux=1.0, fwhm=psf_fwhm)
    gal_conv = galsim.Convolve([gal, psf])
    image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset=(x, y))
    return image.array


def draw_exponential(
    *,
    f: float,
    hlr: float,
    e1: float,
    e2: float,
    x: float,  # pixels
    y: float,
    slen: int,
    fft_size: int,  # rule of thumb: at least 4 times `slen`
    psf_fwhm: float = 0.8,
    pixel_scale: float = 0.2,
):
    gsparams = GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)

    gal = xgalsim.Exponential(flux=f, half_light_radius=hlr)
    gal = gal.shear(g1=e1, g2=e2)

    psf = xgalsim.Gaussian(flux=1.0, fwhm=psf_fwhm)
    gal_conv = xgalsim.Convolve([gal, psf]).withGSParams(gsparams)
    image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset=(x, y))
    return image.array


def draw_exponential_galsim(
    *,
    f: float,
    hlr: float,
    e1: float,
    e2: float,
    g1: float,
    g2: float,
    x: float,  # pixels
    y: float,
    slen: int,
    psf_fwhm: float = 0.8,
    pixel_scale: float = 0.2,
):
    gal = galsim.Exponential(flux=f, half_light_radius=hlr)
    gal = gal.shear(g1=e1, g2=e2)  # intrinsic ellipticity
    gal = gal.shear(g1=g1, g2=g2)

    psf = galsim.Gaussian(flux=1.0, fwhm=psf_fwhm)
    gal_conv = galsim.Convolve([gal, psf])
    image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset=(x, y))
    return image.array


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
