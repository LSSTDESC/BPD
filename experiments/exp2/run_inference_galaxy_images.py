#!/usr/bin/env python3
"""Check chains ran on a variety of galaxies with different SNR, initialization from the prior."""

import time
from functools import partial

import jax.numpy as jnp
import typer
from jax import random, vmap
from jax._src.prng import PRNGKeyArray

from bpd import DATA_DIR
from bpd.chains import run_sampling_nuts, run_warmup_nuts
from bpd.draw import draw_gaussian
from bpd.initialization import init_with_prior
from bpd.pipelines.image_samples import (
    get_target_images,
    get_true_params_from_galaxy_params,
    loglikelihood,
    logprior,
    logtarget,
    sample_target_galaxy_params_simple,
)


def sample_prior(
    rng_key: PRNGKeyArray,
    *,
    flux_bds: tuple = (2.5, 4.0),
    hlr_bds: tuple = (0.7, 2.0),
    shape_noise: float = 0.3,
    g1: float = 0.02,
    g2: float = 0.0,
) -> dict[str, float]:
    k1, k2, k3 = random.split(rng_key, 3)

    lf = random.uniform(k1, minval=flux_bds[0], maxval=flux_bds[1])
    hlr = random.uniform(k2, minval=hlr_bds[0], maxval=hlr_bds[1])

    other_params = sample_target_galaxy_params_simple(
        k3, shape_noise=shape_noise, g1=g1, g2=g2
    )

    return {"lf": lf, "hlr": hlr, **other_params}


def _sample_prior_init(rng_key: PRNGKeyArray):
    prior_samples = sample_prior(rng_key)
    truth_samples = get_true_params_from_galaxy_params(prior_samples)
    return truth_samples


INIT_FNC = partial(init_with_prior, prior=_sample_prior_init)


def main(
    seed: int,
    n_samples: int = 500,
    shape_noise: float = 0.3,
    sigma_e_int: float = 0.5,
    slen: int = 53,
    fft_size: int = 256,
    background: float = 1.0,
    initial_step_size: float = 0.1,
):
    rng_key = random.key(seed)
    pkey, nkey, ikey, rkey = random.split(rng_key, 4)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / f"test_image_sampling_{seed}"
    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)
    fpath = dirpath / f"chain_results_{seed}.npy"

    # setup target density
    draw_fnc = partial(draw_gaussian, slen=slen, fft_size=fft_size)
    _loglikelihood = partial(loglikelihood, draw_fnc=draw_fnc, background=background)
    _logprior = partial(logprior, sigma_e=sigma_e_int)
    _logtarget = partial(
        logtarget, logprior_fnc=_logprior, loglikelihood_fnc=_loglikelihood
    )

    # setup nuts functions
    _run_warmup1 = partial(
        run_warmup_nuts,
        logtarget=_logtarget,
        initial_step_size=initial_step_size,
        max_num_doublings=5,
        n_warmup_steps=500,
    )
    _run_warmup = vmap(vmap(jit(_run_warmup1), in_axes=(0, 0, None)))

    _run_sampling1 = partial(
        run_sampling_nuts,
        logtarget=_logtarget,
        n_samples=n_samples,
        max_num_doublings=5,
    )
    _run_sampling = vmap(vmap(jit(_run_sampling1), in_axes=(0, 0, 0, None)))

    results = {}
    for n_gals in (1, 1, 5, 10, 20, 25, 50, 100, 250, 500):  # repeat 1 == compilation
        print("n_gals:", n_gals)

        # generate data and parameters
        pkeys = random.split(pkey, n_gals)
        galaxy_params = vmap(partial(sample_prior, shape_noise=shape_noise))(pkeys)
        assert galaxy_params["x"].shape == (n_gals,)

        # get images
        draw_params = {**galaxy_params}
        draw_params["f"] = 10 ** draw_params.pop("lf")
        target_images = get_target_images(
            nkey, draw_params, background=background, slen=slen
        )
        assert target_images.shape == (n_gals, slen, slen)
        true_params = vmap(get_true_params_from_galaxy_params)(galaxy_params)

        # initialize positions
        ikeys = random.split(ikey, (n_gals, 4))
        init_positions = vmap(vmap(INIT_FNC, in_axes=(0, None)))(ikeys, true_params)

        gkeys = random.split(rkey, (n_gals, 4, 2))
        wkeys = gkeys[..., 0]
        ikeys = gkeys[..., 1]

        # warmup
        t1 = time.time()
        init_states, tuned_params, adapt_info = _run_warmup(
            wkeys, init_positions, target_images
        )
        t2 = time.time()
        t_warmup = t2 - t1
        tuned_params.pop("max_num_doublings")  # set above, not jittable

        # inference
        t1 = time.time()
        samples, _ = _run_sampling(ikeys, init_states, tuned_params, target_images)
        t2 = time.time()
        t_sampling = t2 - t1

        results[n_gals] = {}
        results[n_gals]["t_warmup"] = t_warmup
        results[n_gals]["t_sampling"] = t_sampling
        results[n_gals]["samples"] = samples
        results[n_gals]["truth"] = true_params
        results[n_gals]["adapt_info"] = adapt_info

    jnp.save(fpath, results)


if __name__ == "__main__":
    typer.run(main)
