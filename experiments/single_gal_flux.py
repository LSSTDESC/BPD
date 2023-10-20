#!/usr/bin/env python
"""Run inference on a single galaxy with flux as a parameter."""
import math
import os
from pathlib import Path

import galsim as _galsim
import jax
import jax_galsim as galsim
import numpy as np
import numpyro
import numpyro.distributions as dist
import yaml
from numpyro.infer import NUTS

from bpd.chains import run_chains
from bpd.draw import add_noise
from bpd.save_load import save_samples

GSPARAMS = galsim.GSParams(minimum_fft_size=512, maximum_fft_size=512)

# read yaml config
config_filename = "single_gal_flux.yaml"
cwd = Path(os.path.dirname(os.path.abspath(__file__)))
with open(cwd / "configs" / config_filename, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# set up image parameters
SLEN = config["image"]["slen"]
PIXEL_SCALE = config["image"]["pixel_scale"]
BACKGROUND = config["image"]["background"]
NOISE_FACTOR = config["image"]["noise_factor"]
PSF_HLR = config["psf"]["psf_hlr"]

# true galaxy parameters
HLR = config["galaxy"]["hlr"]
FLUX = config["galaxy"]["flux"]

# chains parameters
N_CHAIN = config["chains"]["n_chain"]
N_VEC = config["chains"]["n_vec"]
N_WARMUP = config["chains"]["n_warmup"]
N_SAMPLES = config["chains"]["n_samples"]

# gpu and seed
GPU = config["gpu"]
SEED = config["seed"]
ID = config["run_id"]


# get true image
def _draw_gal():
    """Draw the target truth galaxy image."""
    gal = _galsim.Gaussian(flux=FLUX, half_light_radius=HLR)
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


def main():
    # setup gpu
    assert GPU in [0, 1, 2, 3]
    print(f"Using GPU {GPU}")
    jax.config.update("jax_default_device", jax.devices()[GPU])

    nuts_kernel = NUTS(prob_model)

    true_image = _draw_gal()
    data, _ = add_noise(true_image, n=N_VEC * N_CHAIN)

    samples = run_chains(
        data,
        nuts_kernel,
        n_vec=N_VEC,
        n_warmup=N_WARMUP,
        n_samples=N_SAMPLES,
        seed=SEED,
    )

    save_samples(samples, "single_gal_flux.hdf5", group=f"run_{ID}")


if __name__ == "__main__":
    main()
