import jax_galsim as xgalsim

GSPARAMS = xgalsim.GSParams(minimum_fft_size=256, maximum_fft_size=256)


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
):
    # x, y arguments in pixels
    gal = xgalsim.Gaussian(flux=f, half_light_radius=hlr)
    gal = gal.shear(g1=e1, g2=e2)
    gal = gal.shear(g1=g1, g2=g2)

    psf = xgalsim.Gaussian(flux=1, half_light_radius=psf_hlr)
    gal_conv = xgalsim.Convolve([gal, psf]).withGSParams(GSPARAMS)
    image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset=(x, y))
    return image.array
