#!/usr/bin/env python3
"""This experiment is more attuned to what we will use for final results."""

import time
from functools import partial
from typing import Callable

import jax.numpy as jnp
import typer
from jax import Array, jit, random, vmap
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats

from bpd import DATA_DIR
from bpd.chains import run_sampling_nuts, run_warmup_nuts
from bpd.draw import draw_gaussian
from bpd.prior import ellip_prior_e1e2
from bpd.sample import (
    get_target_images,
    sample_target_galaxy_params_simple,
)


def logprior(
    params: dict[str, Array],
    *,
    sigma_e: float,
    sigma_x: float = 0.5,  # pixels
    flux_bds: tuple = (-1.0, 9.0),
    hlr_bds: tuple = (0.01, 5.0),
) -> Array:
    prior = jnp.array(0.0)

    f1, f2 = flux_bds
    prior += stats.uniform.logpdf(params["lf"], f1, f2 - f1)

    h1, h2 = hlr_bds
    prior += stats.uniform.logpdf(params["hlr"], h1, h2 - h1)

    prior += stats.norm.logpdf(params["x"], loc=0.0, scale=sigma_x)
    prior += stats.norm.logpdf(params["y"], loc=0.0, scale=sigma_x)

    e1e2 = jnp.stack((params["e1"], params["e2"]), axis=-1)
    prior += jnp.log(ellip_prior_e1e2(e1e2, sigma=sigma_e))

    return prior


def loglikelihood(
    params: dict[str, Array],
    data: Array,
    *,
    draw_fnc: Callable,
    background: float,
):
    _draw_params = {**params}
    _draw_params["f"] = 10 ** _draw_params.pop("lf")
    model = draw_fnc(**_draw_params)
    likelihood_pp = stats.norm.logpdf(data, loc=model, scale=jnp.sqrt(background))
    return jnp.sum(likelihood_pp)


def _init_fnc(image: Array):
    assert image.ndim == 2
    assert image.shape[0] == image.shape[1]
    flux = image.sum()
    hlr = 0.95  # median size
    e1 = 0.0
    e2 = 0.0
    x = 0.0
    y = 0.0
    return {"lf": jnp.log10(flux), "hlr": hlr, "e1": e1, "e2": e2, "x": x, "y": y}


def logtarget(
    params: dict[str, Array],
    data: Array,
    *,
    logprior_fnc: Callable,
    loglikelihood_fnc: Callable,
):
    return logprior_fnc(params) + loglikelihood_fnc(params, data)


def sample_prior(
    rng_key: PRNGKeyArray,
    *,
    shape_noise: float,
    mean_logflux: float = 2.6,
    sigma_logflux: float = 0.4,
    hlr_bds: tuple[float, float] = (0.7, 1.2),
    g1: float = 0.02,
    g2: float = 0.0,
) -> dict[str, float]:
    k1, k2, k3 = random.split(rng_key, 3)

    lf = random.normal(k1) * sigma_logflux + mean_logflux
    hlr = random.uniform(k2, minval=hlr_bds[0], maxval=hlr_bds[1])
    other_params = sample_target_galaxy_params_simple(
        k3, shape_noise=shape_noise, g1=g1, g2=g2
    )

    return {"lf": lf, "hlr": hlr, **other_params}


def main(
    seed: int,
    n_samples: int = 500,
    shape_noise: float = 0.3,
    sigma_e_int: float = 0.5,
    slen: int = 63,  # adjust depending on HLR bds
    fft_size: int = 256,
    background: float = 1.0,
    initial_step_size: float = 0.1,
):
    rng_key = random.key(seed)
    pkey, nkey, rkey = random.split(rng_key, 3)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / f"exp21_{seed}"
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

        # initialize positions
        init_positions = vmap(_init_fnc)(target_images)
        init_positions = {
            k: v.reshape(-1, 1).repeat(4, axis=1) for k, v in init_positions.items()
        }

        gkeys = random.split(rkey, (n_gals, 4, 2))
        wkeys = gkeys[..., 0]
        skeys = gkeys[..., 1]

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
        samples, _ = _run_sampling(skeys, init_states, tuned_params, target_images)
        t2 = time.time()
        t_sampling = t2 - t1

        results[n_gals] = {}
        results[n_gals]["t_warmup"] = t_warmup
        results[n_gals]["t_sampling"] = t_sampling
        results[n_gals]["samples"] = samples
        results[n_gals]["truth"] = galaxy_params
        results[n_gals]["adapt_info"] = adapt_info
        results[n_gals]["tuned_params"] = tuned_params

    jnp.save(fpath, results)


if __name__ == "__main__":
    typer.run(main)
