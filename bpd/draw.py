import galsim
import jax_galsim as xgalsim
from jax_galsim import GSParams


# forward model
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
    psf_hlr: float = 0.7,
    pixel_scale: float = 0.2,
):
    gsparams = GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)

    gal = xgalsim.Gaussian(flux=f, half_light_radius=hlr)
    gal = gal.shear(g1=e1, g2=e2)

    psf = xgalsim.Gaussian(flux=1.0, half_light_radius=psf_hlr)
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
    psf_hlr: float = 0.7,
    pixel_scale: float = 0.2,
):
    gal = galsim.Gaussian(flux=f, half_light_radius=hlr)
    gal = gal.shear(g1=e1, g2=e2)  # intrinsic ellipticity

    # the correct weak lensing effect includes magnification
    # see: https://galsim-developers.github.io/GalSim/_build/html/shear.html
    mu = (1 - g1**2 - g2**2) ** -1  # convergence kappa = 0
    gal = gal.lens(g1=g1, g2=g2, mu=mu)

    psf = galsim.Gaussian(flux=1.0, half_light_radius=psf_hlr)
    gal_conv = galsim.Convolve([gal, psf])
    image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset=(x, y))
    return image.array
