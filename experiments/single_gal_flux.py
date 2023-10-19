#!/usr/bin/env python
"""Run inference on a single galaxy with flux as a parameter."""
import math

import click
import galsim as _galsim
import jax
import jax_galsim as galsim
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS

from bpd.chains import run_chains
from bpd.draw import add_noise
from bpd.save_load import save_samples

PSF_HLR = 0.7
PIXEL_SCALE = 0.2
SLEN = 101
BACKGROUND = 1e4
NOISE = 1

# galaxy
HLR = 0.9
TRUE_FLUX = 1e5
TRUE_LOG_FLUX = np.log10(TRUE_FLUX)

GSPARAMS = galsim.GSParams(minimum_fft_size=512, maximum_fft_size=512)


# get true image
def _draw_gal():
    gal = _galsim.Gaussian(flux=TRUE_FLUX, half_light_radius=HLR)
    psf = _galsim.Gaussian(flux=1.0, half_light_radius=PSF_HLR)
    gal_conv = _galsim.Convolve([gal, psf])
    gal_conv = gal_conv.drawImage(nx=SLEN, ny=SLEN, scale=PIXEL_SCALE, bandpass=None)
    im = gal_conv.array
    return im


@jax.jit
@jax.vmap
def draw_gal(lf):
    gal = galsim.Gaussian(flux=10**lf, half_light_radius=HLR)
    psf = galsim.Gaussian(flux=1.0, half_light_radius=PSF_HLR)
    gal_conv = galsim.Convolve([gal, psf]).withGSParams(GSPARAMS)
    image = gal_conv.drawImage(nx=SLEN, ny=SLEN, scale=PIXEL_SCALE)
    return image.array


def prob_model(data: np.ndarray):
    batch_dim, _, _ = data.shape
    with numpyro.plate("b", batch_dim, dim=-1):
        lf = numpyro.sample("lf", dist.Uniform(3, 6))
        im = draw_gal(lf)
    numpyro.sample("obs", dist.Normal(im, math.sqrt(BACKGROUND)), obs=data)


@click.command()
@click.option("-g", "--gpu", default=0, type=int)
@click.option("-n", "--n-chain", default=100, type=int)
@click.option("-nv", "--n-vec", default=10, type=int)
@click.option("-nw", "--n_warmup", default=100, type=int)
@click.option("-ns", "--n_samples", default=1000, type=int)
@click.option("-s", "--seed", default=42, type=int)
def main(gpu: int, n_chain: int, n_vec: int, n_warmup: int, n_samples: int, seed: int):
    # setup gpu
    assert gpu in [0, 1, 2, 3]
    print(f"Using GPU {gpu}")
    jax.config.update("jax_default_device", jax.devices()[gpu])

    nuts_kernel = NUTS(prob_model)

    true_image = _draw_gal()
    data, _ = add_noise(true_image, n=n_vec * n_chain)

    samples = run_chains(
        data,
        nuts_kernel,
        n_vec=n_vec,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
    )

    save_samples(samples, "single_gal_flux.hdf5")


if __name__ == "__main__":
    main()
