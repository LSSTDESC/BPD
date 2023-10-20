from functools import partial

import galsim
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax_galsim as galaxim
import numpy as np
import numpyro
import numpyro.distributions as dist
from descwl_shear_sims.constants import FIXED_PSF_FWHM, SCALE
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.objlists import get_objlist
from descwl_shear_sims.psfs import make_fixed_psf
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.surveys import get_survey
from numpyro.infer import MCMC, NUTS

SEED = 8317
RNG = np.random.RandomState(SEED)

COADD_DIM = 251
N_OBJECTS = 25

PSF_DIM = 51
bands = ["i"]
NOISE_FACTOR = 1


def get_coadd_center_gs_pos(coadd_wcs, coadd_bbox):
    """
    get the sky position of the center of the coadd within the
    bbox as a galsim CelestialCoord

    Parameters
    ----------
    coadd_wcs: DM wcs
        The wcs for the coadd
    coadd_bbox: geom.Box2I
        The bounding box for the coadd within larger wcs system

    Returns
    -------
    galsim CelestialCoord
    """

    # world origin is at center of the coadd, which itself
    # is in a bbox shifted from the overall WORLD_ORIGIN

    bbox_cen_skypos = coadd_wcs.pixelToSky(coadd_bbox.getCenter())

    return galsim.CelestialCoord(
        ra=float(bbox_cen_skypos.getRa()) * galsim.radians,
        dec=float(bbox_cen_skypos.getDec()) * galsim.radians,
    )


def _coadd_sim_data(rng, sim_data, nowarp, remove_poisson):
    """
    copied from mdet-lsst-sim
    """
    from descwl_coadd.coadd import make_coadd
    from descwl_coadd.coadd_nowarp import make_coadd_nowarp
    from metadetect.lsst.util import extract_multiband_coadd_data

    bands = list(sim_data["band_data"].keys())

    if nowarp:
        exps = sim_data["band_data"][bands[0]]

        if len(exps) > 1:
            raise ValueError("only one epoch for nowarp")

        coadd_data_list = [
            make_coadd_nowarp(
                exp=exps[0],
                psf_dims=sim_data["psf_dims"],
                rng=rng,
                remove_poisson=remove_poisson,
            )
            for band in bands
        ]
    else:
        coadd_data_list = [
            make_coadd(
                exps=sim_data["band_data"][band],
                psf_dims=sim_data["psf_dims"],
                rng=rng,
                coadd_wcs=sim_data["coadd_wcs"],
                coadd_bbox=sim_data["coadd_bbox"],
                remove_poisson=remove_poisson,
            )
            for band in bands
        ]
    return extract_multiband_coadd_data(coadd_data_list)


GSPARAMS = galaxim.GSParams(minimum_fft_size=512, maximum_fft_size=512)


@partial(jax.jit, static_argnums=(-1,))
def draw_model(xs, ys, hlrs, fluxes, g1, g2, shifts):
    psf_gs_no_pix = galaxim.Gaussian(fwhm=FIXED_PSF_FWHM)
    image = galaxim.Image(COADD_DIM, COADD_DIM, scale=SCALE)
    for ii in range(N_OBJECTS):
        shift = shifts[ii]  # arcsec
        shift = shift.shear(galsim.Shear(g1=0.02, g2=0.0))
        pos = shift / SCALE

        pos = galaxim.PositionD(pos.x, pos.y)
        x = xs[ii]
        y = ys[ii]
        hlr = hlrs[ii]
        flux = fluxes[ii]

        shift = galaxim.PositionD(x=x, y=y)
        image_pos = pos - shift
        gal = galaxim.Exponential(flux=flux, half_light_radius=hlr)
        gal = gal.shear(g1=g1, g2=g2)

        obj = galaxim.Convolve([gal, psf_gs_no_pix]).withGSParams(GSPARAMS)
        stamp = obj.drawImage(nx=COADD_DIM, ny=COADD_DIM, offset=image_pos, scale=SCALE)
        image += stamp

    return image.array


def prob_model(y=None, shifts=None):
    g1 = numpyro.sample("g1", dist.Uniform(-0.1, 0.1))
    g2 = numpyro.sample("g2", dist.Uniform(-0.1, 0.1))
    with numpyro.plate("gals", N_OBJECTS, dim=-1):
        fluxes = jnp.power(10, numpyro.sample("f", dist.Uniform(2.5, 6)))
        hlrs = numpyro.sample("hlr", dist.Uniform(0.2, 1.2))
        xs = numpyro.sample("x", dist.Uniform(-0.5, 0.5))
        ys = numpyro.sample("y", dist.Uniform(-0.5, 0.5))
    im = draw_model(xs, ys, hlrs, fluxes, g1, g2, shifts)
    numpyro.sample("obs", dist.Normal(im, 1.0), obs=y)


def main():
    # this makes a grid of fixed exponential galaxies
    # with default properties. One exposure per band

    galaxy_catalog = make_galaxy_catalog(
        rng=RNG,
        gal_type="fixed",
        coadd_dim=COADD_DIM,
        buff=20,
        layout="grid",
        gal_config={"morph": "exp", "mag": 22},
    )
    survey = get_survey(gal_type=galaxy_catalog.gal_type, band=bands[0])
    noise_for_gsparams = survey.noise * NOISE_FACTOR
    lists = get_objlist(
        galaxy_catalog=galaxy_catalog,
        survey=survey,
        star_catalog=None,
        noise=noise_for_gsparams,
    )
    assert len(lists["shifts"]) == N_OBJECTS
    # gaussian psf
    psf = make_fixed_psf(psf_type="gauss")

    data = make_sim(
        rng=RNG,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=COADD_DIM,
        psf_dim=PSF_DIM,
        bands=bands,
        g1=0.02,
        g2=0.00,
        psf=psf,
        dither=False,
        rotate=False,
        se_dim=COADD_DIM,
        noise_factor=NOISE_FACTOR,
    )

    se_wcs = data["se_wcs"][0]
    coadd_gs_center = get_coadd_center_gs_pos(data["coadd_wcs"], data["coadd_bbox"])
    obj_x = []
    obj_y = []
    for shift in lists["shifts"]:
        shift = shift.shear(galsim.Shear(g1=0.02, g2=0.0))
        w_pos = coadd_gs_center.deproject(
            shift.x * galsim.arcsec, shift.y * galsim.arcsec
        )
        image_pos = se_wcs.toImage(w_pos)
        obj_x.append(image_pos.x)
        obj_y.append(image_pos.y)
    obj_x = np.array(obj_x)
    obj_y = np.array(obj_y)

    mb_coadd = _coadd_sim_data(RNG, data, True, False)
    data = mb_coadd["mbexp"]["i"].image.array

    shifts = jnp.array(lists["shifts"])

    nuts_kernel = NUTS(prob_model, max_tree_depth=10)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000)

    rng_key = jrandom.PRNGKey(SEED)
    mcmc.run(rng_key, y=data, shifts=shifts)


if __name__ == "__main__":
    main()
