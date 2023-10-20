#!/usr/bin/env python
"""Run inference on a single galaxy with 6 parameters"""
import math
import os
import sys
from pathlib import Path

import galsim as _galsim
import jax
import jax_galsim as galsim
import numpyro
import numpyro.distributions as dist
import yaml
from numpyro.infer import NUTS

from bpd.chains import run_chains
from bpd.draw import add_noise
from bpd.save_load import save_samples

GSPARAMS = galsim.GSParams(minimum_fft_size=512, maximum_fft_size=512)

# read yaml config
if len(sys.argv) > 1:
    config_filename = sys.argv[1]
else:
    config_filename = "single_gal.yaml"

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
with open(cwd / "configs" / config_filename, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# set up image parameters
SLEN = config["image"]["slen"]
PIXEL_SCALE = config["image"]["pixel_scale"]
BACKGROUND = config["image"]["background"]
NOISE_FACTOR = config["image"]["noise_factor"]
PSF_HLR = config["psf"]["hlr"]

# true galaxy parameters
HLR = config["galaxy"]["hlr"]
FLUX = config["galaxy"]["flux"]
X0 = config["galaxy"]["x0"]
Y0 = config["galaxy"]["y0"]
G1 = config["galaxy"]["g1"]
G2 = config["galaxy"]["g2"]

# chains parameters
N_CHAIN = config["chains"]["n_chains"]
N_VEC = config["chains"]["n_vecs"]
N_WARMUP = config["chains"]["n_warmup"]
N_SAMPLES = config["chains"]["n_samples"]

# gpu and seed
GPU = config["gpu"]
SEED = config["seed"]
ID = config["run_id"]

GSPARAMS = galsim.GSParams(minimum_fft_size=512, maximum_fft_size=512)


# get true
def _draw_gal():
    gal = _galsim.Gaussian(flux=FLUX, half_light_radius=HLR)
    gal = gal.shear(g1=G1, g2=G2)
    pos = _galsim.PositionD(x=X0, y=Y0)
    psf = _galsim.Gaussian(flux=1.0, half_light_radius=PSF_HLR)
    gal_conv = _galsim.Convolve([gal, psf])
    gal_conv = gal_conv.drawImage(
        nx=SLEN, ny=SLEN, scale=PIXEL_SCALE, offset=pos, bandpass=None
    )
    im = gal_conv.array
    return im


TRUE_IMAGE = _draw_gal()


@jax.jit
@jax.vmap
def draw_gal(lf, hlr, x, y, g1, g2):
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
    return image.array


def prob_model(data):
    batch_dim, _, _ = data.shape
    with numpyro.plate("b", batch_dim, dim=-1):
        lf = numpyro.sample("lf", dist.Uniform(3, 6))
        hlr = numpyro.sample("hlr", dist.Uniform(0.5, 1.5))
        x = numpyro.sample("x", dist.Uniform(-0.5, 0.5))
        y = numpyro.sample("y", dist.Uniform(-0.5, 0.5))
        g1 = numpyro.sample("g1", dist.Uniform(-0.1, 0.1))
        g2 = numpyro.sample("g2", dist.Uniform(-0.1, 0.1))
    im = draw_gal(lf, hlr, x, y, g1, g2)
    numpyro.sample("obs", dist.Normal(im, math.sqrt(BACKGROUND)), obs=data)


def main():
    # setup gpu
    assert GPU in [0, 1, 2, 3]
    print(f"Using GPU {GPU}")
    jax.config.update("jax_default_device", jax.devices()[GPU])

    data, _ = add_noise(TRUE_IMAGE, n=N_VEC * N_CHAIN)

    nuts_kernel = NUTS(prob_model)
    samples = run_chains(
        data,
        nuts_kernel,
        n_vec=N_VEC,
        n_warmup=N_WARMUP,
        n_samples=N_SAMPLES,
        seed=SEED,
    )

    save_samples(samples, "single_gal.hdf5", group=f"run_{ID}")


if __name__ == "__main__":
    main()
