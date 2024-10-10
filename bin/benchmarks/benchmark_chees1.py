#!/usr/bin/env python3

"""Here we run a variable number of chains on a single galaxy and noise realization (NUTS)."""

import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import time
from functools import partial
from pathlib import Path

import blackjax
import galsim
import jax
import jax.numpy as jnp
import jax_galsim as xgalsim
import numpy as np
import optax
from jax import jit as jjit
from jax import random, vmap
from jax.scipy import stats

from bpd.chains import inference_loop
from bpd.draw import add_noise
from bpd.measure import get_snr

print("devices available:", jax.devices())

SCRATCH_DIR = Path("/pscratch/sd/i/imendoza/data/cache_chains")


# GPU preamble
GPU = jax.devices("gpu")[0]

jax.config.update("jax_default_device", GPU)

LOG_FILE = Path(__file__).parent / "log.txt"


PIXEL_SCALE = 0.2
BACKGROUND = 1e4
SLEN = 53
PSF_HLR = 0.7
GSPARAMS = xgalsim.GSParams(minimum_fft_size=256, maximum_fft_size=256)

LOG_FLUX = 4.5
HLR = 0.9
G1 = 0.05
G2 = 0.0
X = 0.0
Y = 0.0

TRUE_PARAMS = {"f": LOG_FLUX, "hlr": HLR, "g1": G1, "g2": G2, "x": X, "y": Y}

# make sure relevant things are in GPU
TRUE_PARAMS_GPU = jax.device_put(TRUE_PARAMS, device=GPU)
BACKGROUND_GPU = jax.device_put(BACKGROUND, device=GPU)
BOUNDS = {
    "f": (-1.0, 9.0),
    "hlr": (0.01, 5.0),
    "g1": (-0.7, 0.7),
    "g2": (-0.7, 0.7),
    "x": 1,  # sigma (in pixels)
    "y": 1,  # sigma (in pixels)
}
BOUNDS_GPU = jax.device_put(BOUNDS, device=GPU)


# run setup
IS_MATRIX_DIAGONAL = True
N_WARMUPS = 500
N_SAMPLES = 1000
SEED = 42
TAG = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# chees setup
LR = 1e-3
INIT_STEP_SIZE = 0.1

ALL_N_CHAINS = (1, 5, 10, 25, 50, 100, 150, 200, 300, 500)


# sample from ball around some dictionary of true params
def sample_ball(rng_key, center_params: dict):
    new = {}
    keys = random.split(rng_key, len(center_params.keys()))
    rng_key_dict = {p: k for p, k in zip(center_params, keys)}
    for p in center_params:
        centr = center_params[p]
        if p == "f":
            new[p] = random.uniform(
                rng_key_dict[p], shape=(), minval=centr - 0.25, maxval=centr + 0.25
            )
        elif p == "hlr":
            new[p] = random.uniform(
                rng_key_dict[p], shape=(), minval=centr - 0.2, maxval=centr + 0.2
            )
        elif p in {"g1", "g2"}:
            new[p] = random.uniform(
                rng_key_dict[p], shape=(), minval=centr - 0.025, maxval=centr + 0.025
            )
        elif p in {"x", "y"}:
            new[p] = random.uniform(
                rng_key_dict[p], shape=(), minval=centr - 0.5, maxval=centr + 0.5
            )
    return new


def _draw_gal():
    gal = galsim.Gaussian(flux=10**LOG_FLUX, half_light_radius=HLR)
    gal = gal.shift(dx=X, dy=Y)
    gal = gal.shear(g1=G1, g2=G2)

    psf = galsim.Gaussian(flux=1.0, half_light_radius=PSF_HLR)
    gal_conv = galsim.Convolve([gal, psf])
    image = gal_conv.drawImage(nx=SLEN, ny=SLEN, scale=PIXEL_SCALE)
    return image.array


def draw_gal(f, hlr, g1, g2, x, y):
    # x, y arguments in pixels
    gal = xgalsim.Gaussian(flux=10**f, half_light_radius=hlr)
    gal = gal.shift(dx=x * PIXEL_SCALE, dy=y * PIXEL_SCALE)
    gal = gal.shear(g1=g1, g2=g2)

    psf = xgalsim.Gaussian(flux=1, half_light_radius=PSF_HLR)
    gal_conv = xgalsim.Convolve([gal, psf]).withGSParams(GSPARAMS)
    image = gal_conv.drawImage(nx=SLEN, ny=SLEN, scale=PIXEL_SCALE)
    return image.array


def _logprob_fn(params, data):

    # prior
    prior = jnp.array(0.0, device=GPU)
    for p in ("f", "hlr", "g1", "g2"):  # uniform priors
        b1, b2 = BOUNDS_GPU[p]
        prior += stats.uniform.logpdf(params[p], b1, b2 - b1)

    for p in ("x", "y"):  # normal
        sigma = BOUNDS_GPU[p]
        prior += stats.norm.logpdf(params[p], sigma)

    # likelihood
    model = draw_gal(**params)
    likelihood_pp = stats.norm.logpdf(data, loc=model, scale=jnp.sqrt(BACKGROUND_GPU))
    likelihood = jnp.sum(likelihood_pp)

    return prior + likelihood


def _log_setup(snr: float):
    with open(LOG_FILE, "a") as f:
        print(file=f)
        print(
            f"""Running benchmark chees 1 with configuration as follows. Variable number of chains.
    
    The sampler used is NUTS with standard warmup.

    TAG: {TAG}
    SEED: {SEED} 

    Overall sampler configuration (fixed):
        n_samples: {N_SAMPLES}
        n_warmups: {N_WARMUPS}
        diagonal matrix: {IS_MATRIX_DIAGONAL}
        learning_rate: {LR}
        init_step_size: {INIT_STEP_SIZE}

    galaxy parameters:
        LOG_FLUX: {LOG_FLUX}
        HLR: {HLR}
        G1: {G1}
        G2: {G2}
        X: {X}
        Y: {Y}

    prior bounds: {BOUNDS}

    other parameters:
        slen: {SLEN}
        psf_hlr: {PSF_HLR}
        background: {BACKGROUND}  
        snr: {snr}
    """,
            file=f,
        )


def do_warmup(rng_key, positions, data, n_chains: int = None):
    """Cannot jit!, but seems to automatically compile after running once."""
    logdensity = partial(_logprob_fn, data=data)
    warmup = blackjax.chees_adaptation(logdensity, n_chains)
    optim = optax.adam(LR)
    # `positions` = PyTree where each leaf has shape (num_chains, ...)
    return warmup.run(rng_key, positions, INIT_STEP_SIZE, optim, N_WARMUPS)


def do_inference(rng_key, init_states, data, tuned_params: dict):
    """Also won't jit for unknown reasons"""
    _logdensity = partial(_logprob_fn, data=data)
    kernel = blackjax.dynamic_hmc(_logdensity, **tuned_params).step
    return inference_loop(rng_key, init_states, kernel=kernel, n_samples=N_SAMPLES)


def main():
    print("TAG:", TAG)
    snr = get_snr(_draw_gal(), BACKGROUND)
    print("galaxy snr:", snr)

    # get data
    _data = add_noise(_draw_gal(), BACKGROUND, rng=np.random.default_rng(SEED), n=1)[0]
    data_gpu = jax.device_put(_data, device=GPU)
    print("data info:", data_gpu.devices(), type(data_gpu), data_gpu.shape)

    # collect random keys we need
    rng_key = random.key(SEED)
    rng_key = jax.device_put(rng_key, device=GPU)

    ball_key, warmup_key, sample_key = random.split(rng_key, 3)

    warmup_keys = random.split(warmup_key, len(ALL_N_CHAINS))
    ball_keys = random.split(ball_key, max(ALL_N_CHAINS))
    sample_keys = random.split(sample_key, max(ALL_N_CHAINS))
    assert sample_keys.shape == (max(ALL_N_CHAINS),)

    # get initial positions for all chains
    all_init_positions = vmap(sample_ball, in_axes=(0, None))(
        ball_keys, TRUE_PARAMS_GPU
    )
    assert all_init_positions["f"].shape == (max(ALL_N_CHAINS),)

    # jit and vmap functions to run chains
    _run_inference = vmap(do_inference, in_axes=(0, 0, None, None))

    # results
    results = {n: {} for n in ALL_N_CHAINS}

    for ii, n_chains in enumerate(ALL_N_CHAINS):
        print(f"n_chains: {n_chains}")

        _key1 = warmup_keys[ii]
        _keys2 = sample_keys[:n_chains]
        _init_positions = {p: q[:n_chains] for p, q in all_init_positions.items()}

        _run_warmup = partial(do_warmup, n_chains=n_chains)

        # compilation times for warmup
        t1 = time.time()
        (_sts, _tp), _ = jax.block_until_ready(
            _run_warmup(_key1, _init_positions, data_gpu)
        )
        t2 = time.time()
        results[n_chains]["warmup_comp_time"] = t2 - t1

        # inference compilation time
        if ii == 0:
            t1 = time.time()
            _ = jax.block_until_ready(_run_inference(_keys2, _sts, data_gpu, _tp))
            t2 = time.time()
            results["inference_comp_time"] = t2 - t1

        # run times
        t1 = time.time()
        (last_states, tuned_params), adapt_info = jax.block_until_ready(
            _run_warmup(_key1, _init_positions, data_gpu)
        )
        t2 = time.time()
        results[n_chains]["warmup_run_time"] = t2 - t1

        t1 = time.time()
        states, infos = jax.block_until_ready(
            _run_inference(_keys2, last_states, data_gpu, tuned_params)
        )
        t2 = time.time()
        results[n_chains]["inference_run_time"] = t2 - t1

        # save states and info for future reference
        results[n_chains]["states"] = states
        results[n_chains]["info"] = infos
        results[n_chains]["adapt_info"] = adapt_info
        results[n_chains]["step_size"] = tuned_params["step_size"]

    results["data"] = data_gpu
    results["init_positions"] = all_init_positions

    filename = f"results_chees_benchmark1_{TAG}.npy"
    filepath = SCRATCH_DIR.joinpath(filename)
    jnp.save(filepath, results)

    _log_setup(snr)
    with open(LOG_FILE, "a") as f:
        print(file=f)
        print(f"results were saved to {filepath}", file=f)


if __name__ == "__main__":
    main()
