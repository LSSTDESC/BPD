#!/usr/bin/env python
import math

import galsim as _galsim
import jax
import jax.numpy as jnp
import jax_galsim as galsim
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import device_get, device_put, random
from numpyro.infer import MCMC, NUTS

from bpd.chains import run_chains
from bpd.draw import add_noise

PSF_HLR = 0.7
PIXEL_SCALE = 0.2
SLEN = 51
BACKGROUND = 100

N_GAL = 2

# galaxy
TRUE_HLR = 0.9
LOG_TRUE_FLUX = 4.5
TRUE_FLUX = 10**LOG_TRUE_FLUX

# positions
POS = jnp.array([[-5.0, 0.0], [7.5, 0.0]])

# shear (per bin)
G = jnp.array([[0.02, 0], [-0.02, 0]])

# which tomographic bin
N_TOMO = 2
TOMO_BINS = jnp.array([0, 1])

LF = jnp.array([LOG_TRUE_FLUX, LOG_TRUE_FLUX])
HLR = jnp.array([TRUE_HLR, TRUE_HLR])


assert len(TOMO_BINS) == POS.shape[0] == len(HLR) == len(LF) == N_GAL
assert len(jnp.unique(TOMO_BINS)) == G.shape[0] == N_TOMO


# params
TRUE_PARAMS = {"lfs": LF, "hlrs": HLR, "gs": G, "poss": POS, "tomo_bins": TOMO_BINS}
TRUE_PARAMS = {k: v[None] for k, v in TRUE_PARAMS.items()}
# TRUE_PARAMS_ARR = {k:jnp.array([v]) for k,v in TRUE_PARAMS.items()}

GSPARAMS = galsim.GSParams(minimum_fft_size=512, maximum_fft_size=512)


# get true image (no noise)
def _draw_gals():
    fim = np.zeros((SLEN, SLEN))
    for ii in range(N_GAL):
        gal = _galsim.Gaussian(flux=TRUE_FLUX, half_light_radius=TRUE_HLR)
        t = TOMO_BINS[ii]
        gal = gal.shear(g1=G[t, 0], g2=G[t, 1])
        dx, dy = POS[ii]
        pos = _galsim.PositionD(x=dx, y=dy)
        psf = _galsim.Gaussian(flux=1.0, half_light_radius=PSF_HLR)
        gal_conv = _galsim.Convolve([gal, psf])
        im = gal_conv.drawImage(nx=SLEN, ny=SLEN, scale=PIXEL_SCALE, offset=pos)
        fim += im.array

    return fim


TRUE_IMAGE = _draw_gals()


@jax.jit
def draw_gals(lfs, hlrs, gs, poss, tomo_bins):
    fim = jnp.zeros((SLEN, SLEN))
    for ii in range(N_GAL):
        lf = lfs[ii]
        hlr = hlrs[ii]
        x, y = poss[ii]
        t = tomo_bins[ii]
        g1, g2 = gs[t]

        gal = galsim.Gaussian(flux=10**lf, half_light_radius=hlr)
        gal = gal.shear(g1=g1, g2=g2)

        pos = galsim.PositionD(x=x, y=y)
        psf = galsim.Gaussian(flux=1.0, half_light_radius=PSF_HLR)
        gal_conv = galsim.Convolve([gal, psf]).withGSParams(GSPARAMS)
        image = gal_conv.drawImage(
            nx=SLEN,
            ny=SLEN,
            scale=PIXEL_SCALE,
            offset=pos,
        )
        fim += image.array
    return fim


def prob_model(data):
    batch_dim, _, _ = data.shape

    # global shears, one per tomo bin.
    g11 = numpyro.sample("g11", dist.Uniform(-0.1, 0.1))
    g12 = numpyro.sample("g12", dist.Uniform(-0.1, 0.1))
    g21 = numpyro.sample("g21", dist.Uniform(-0.1, 0.1))
    g22 = numpyro.sample("g22", dist.Uniform(-0.1, 0.1))
    g = jnp.array([[g11, g12], [g21, g22]]).reshape(batch_dim, 2, 2)

    with numpyro.plate("n", N_GAL, dim=-1):
        lf = numpyro.sample("lf", dist.Uniform(3, 6))
        hlr = numpyro.sample("hlr", dist.Uniform(0.5, 1.5))

    im = draw_gals(lf, hlr, g, POS, TOMO_BINS)
    numpyro.sample("obs", dist.Normal(im, math.sqrt(BACKGROUND)), obs=data)


def main():
    # TODO: Write function to pmap on non vectorized version and use it here.
    data, _ = add_noise(TRUE_IMAGE, BACKGROUND, n=4)
    nuts_kernel = NUTS(
        prob_model,
        max_tree_depth=10,
        # init_strategy=init_to_median,  find_heuristic_step_size=True
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=1000,
        num_samples=4000,
        num_chains=1,
        chain_method="sequential",
    )
    rng_key = random.PRNGKey(6)
    mcmc.run(rng_key, data=data)

    # TODO: save samples


if __name__ == "__main__":
    main()