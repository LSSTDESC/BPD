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
from jax import jit as jjit
from jax import random, vmap
from jax.scipy import stats

from bpd.chains import inference_loop
from bpd.measure import get_snr
from bpd.noise import add_noise

print("devices available:", jax.devices())

SCRATCH_DIR = Path("/pscratch/sd/i/imendoza/data/cache_chains")
LOG_FILE = Path(__file__).parent / "log.txt"


# GPU preamble
GPU = jax.devices("gpu")[0]

jax.config.update("jax_default_device", GPU)


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
IS_MATRIX_DIAGONAL = False
N_WARMUPS = 500
MAX_DOUBLINGS = 5
N_SAMPLES = 1000
SEED = 42
TAG = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

ALL_N_CHAINS = (1, 5, 10, 25, 50, 100, 150, 200)


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
            f"""Running benchmark 1 with configuration as follows. Variable number of chains.
    
    The sampler used is NUTS with standard warmup.

    TAG: {TAG}
    SEED: {SEED} 

    Overall sampler configuration (fixed):
        max doublings: {MAX_DOUBLINGS}
        n_samples: {N_SAMPLES}
        n_warmups: {N_WARMUPS}
        diagonal matrix: {IS_MATRIX_DIAGONAL}

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


# vmap only rng_key
def do_warmup(rng_key, init_position: dict, data):

    _logdensity = partial(_logprob_fn, data=data)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        _logdensity,
        progress_bar=False,
        is_mass_matrix_diagonal=IS_MATRIX_DIAGONAL,
        max_num_doublings=MAX_DOUBLINGS,
        initial_step_size=0.1,
        target_acceptance_rate=0.90,
    )
    return warmup.run(
        rng_key, init_position, N_WARMUPS
    )  # (init_states, tuned_params), adapt_info


def do_inference(rng_key, init_state, data, step_size: float, inverse_mass_matrix):
    _logdensity = partial(_logprob_fn, data=data)
    kernel = blackjax.nuts(
        _logdensity,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        max_num_doublings=MAX_DOUBLINGS,
    ).step
    return inference_loop(
        rng_key, init_state, kernel=kernel, n_samples=N_SAMPLES
    )  # state, info


def main():
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

    ball_keys = random.split(ball_key, max(ALL_N_CHAINS))
    warmup_keys = random.split(warmup_key, max(ALL_N_CHAINS))
    sample_keys = random.split(sample_key, max(ALL_N_CHAINS))
    assert warmup_keys.shape == (200,)

    # get initial positions for all chains
    all_init_positions = vmap(sample_ball, in_axes=(0, None))(
        ball_keys, TRUE_PARAMS_GPU
    )
    assert all_init_positions["f"].shape == (200,)

    # jit and vmap functions to run chains
    # same data, multiple chains
    _run_warmup = vmap(jjit(do_warmup), in_axes=(0, 0, None))
    _run_inference = vmap(jjit(do_inference), in_axes=(0, 0, None, 0, 0))

    # results
    results = {n: {} for n in ALL_N_CHAINS}

    for ii, n_chains in enumerate(ALL_N_CHAINS):
        print(f"n_chains: {n_chains}")

        _keys1 = warmup_keys[:n_chains]
        _keys2 = sample_keys[:n_chains]
        _init_positions = {p: q[:n_chains] for p, q in all_init_positions.items()}

        if ii == 0:

            # compilation times
            t1 = time.time()
            (_sts, _tp), _ = jax.block_until_ready(
                _run_warmup(_keys1, _init_positions, data_gpu)
            )
            t2 = time.time()
            results[n_chains]["warmup_comp_time"] = t2 - t1

            t1 = time.time()
            _ = jax.block_until_ready(
                _run_inference(
                    _keys2, _sts, data_gpu, _tp["step_size"], _tp["inverse_mass_matrix"]
                )
            )
            t2 = time.time()
            results[n_chains]["inference_comp_time"] = t2 - t1

        # run times
        t1 = time.time()
        (init_states, tuned_params), adapt_info = jax.block_until_ready(
            _run_warmup(_keys1, _init_positions, data_gpu)
        )
        t2 = time.time()
        results[n_chains]["warmup_run_time"] = t2 - t1

        t1 = time.time()
        states, infos = jax.block_until_ready(
            _run_inference(
                _keys2,
                init_states,
                data_gpu,
                tuned_params["step_size"],
                tuned_params["inverse_mass_matrix"],
            )
        )
        t2 = time.time()
        results[n_chains]["inference_run_time"] = t2 - t1

        # save states and info for future reference
        results[n_chains]["states"] = states
        results[n_chains]["info"] = infos
        results[n_chains]["adapt_info"] = adapt_info
        results[n_chains]["tuned_params"] = tuned_params
        results[n_chains]["data"] = data_gpu
        results[n_chains]["init_positions"] = all_init_positions

    filename = f"results_benchmark1_{TAG}.npy"
    filepath = SCRATCH_DIR.joinpath(filename)
    jnp.save(filepath, results)

    _log_setup(snr)
    with open(LOG_FILE, "a") as f:
        print(file=f)
        print(f"results were saved to {filepath}", file=f)


if __name__ == "__main__":
    main()