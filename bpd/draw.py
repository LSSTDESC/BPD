import galsim
import jax_galsim as xgalsim
from jax_galsim import GSParams


def draw_gaussian(
    f: float,
    hlr: float,
    e1: float,
    e2: float,
    g1: float,
    g2: float,
    x: float,
    y: float,
    pixel_scale: float = 0.2,
    slen: int = 53,
    psf_hlr: float = 0.7,
    fft_size: int = 256,
):
    gsparams = GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)

    # x, y arguments in pixels
    gal = xgalsim.Gaussian(flux=f, half_light_radius=hlr)
    gal = gal.shear(g1=e1, g2=e2)
    gal = gal.shear(g1=g1, g2=g2)

    psf = xgalsim.Gaussian(flux=1, half_light_radius=psf_hlr)
    gal_conv = xgalsim.Convolve([gal, psf]).withGSParams(gsparams)
    image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset=(x, y))
    return image.array


def draw_gaussian_galsim(
    f: float,
    hlr: float,
    e1: float,
    e2: float,
    g1: float,
    g2: float,
    x: float,  # pixels
    y: float,
    pixel_scale: float = 0.2,
    slen: int = 53,
    psf_hlr: float = 0.7,
):
    gal = galsim.Gaussian(flux=f, half_light_radius=hlr)
    gal = gal.shear(g1=e1, g2=e2)
    gal = gal.shear(g1=g1, g2=g2)

    psf = galsim.Gaussian(flux=1.0, half_light_radius=psf_hlr)
    gal_conv = galsim.Convolve([gal, psf])
    image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset=(x, y))
    return image.array
