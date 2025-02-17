#!/usr/bin/env python3
"""This experiment is more attuned to what we will use for final results."""

import time
from functools import partial
from typing import Callable

import jax.numpy as jnp
import typer
from jax import Array, jit, random, vmap
from jax._src.prng import PRNGKeyArray

from bpd import DATA_DIR
from bpd.chains import run_sampling_nuts, run_warmup_nuts
from bpd.draw import draw_gaussian
from bpd.io import save_dataset
from bpd.likelihood import gaussian_image_loglikelihood
from bpd.pipelines import logtarget_images
from bpd.prior import interim_gprops_logprior
from bpd.sample import (
    get_target_images,
    get_true_params_from_galaxy_params,
    sample_target_galaxy_params_simple,
)


def _sample_prior(
    rng_key: PRNGKeyArray,
    *,
    shape_noise: float,
    mean_logflux: float,
    sigma_logflux: float,
    mean_loghlr: float,
    sigma_loghlr: float,
    g1: float = 0.02,
    g2: float = 0.0,
    min_logflux: float = 0.0,
) -> dict[str, float]:
    k1, k2, k3 = random.split(rng_key, 3)

    lf = random.normal(k1) * sigma_logflux + mean_logflux
    lf = lf.clip(min=min_logflux)

    lhlr = random.normal(k2) * sigma_loghlr + mean_loghlr
    other_params = sample_target_galaxy_params_simple(
        k3, shape_noise=shape_noise, g1=g1, g2=g2
    )

    return {"lf": lf, "lhlr": lhlr, **other_params}


def _init_fnc(key: PRNGKeyArray, image: Array, true_params: dict):
    assert image.ndim == 2
    assert image.shape[0] == image.shape[1]
    flux = image.sum()

    k1, k2, k3 = random.split(key, 3)

    tlhlr = true_params["lhlr"]
    lhlr = random.uniform(k1, shape=(), minval=tlhlr - 0.015, maxval=tlhlr + 0.015)

    te1 = true_params["e1"]
    e1 = random.uniform(k2, shape=(), minval=te1 - 0.1, maxval=te1 + 0.1)

    te2 = true_params["e2"]
    e2 = random.uniform(k3, shape=(), minval=te2 - 0.1, maxval=te2 + 0.1)
    return {
        "lf": jnp.log10(flux),
        "lhlr": lhlr,
        "e1": e1,
        "e2": e2,
        "dx": 0.0,
        "dy": 0.0,
    }


def logtarget(
    params: dict[str, Array],
    data: Array,
    *,
    logprior_fnc: Callable,
    loglikelihood_fnc: Callable,
):
    return logprior_fnc(params) + loglikelihood_fnc(params, data)


def _run_warmup(
    key, init_pos, data, fixed_params, *, logtarget: Callable, initial_step_size: float
):
    return run_warmup_nuts(
        key,
        init_pos,
        data,
        logtarget=partial(logtarget, fixed_params=fixed_params),
        initial_step_size=initial_step_size,
        max_num_doublings=5,
        n_warmup_steps=500,
    )


def _run_sampling(
    key, istates, tp, data, fixed_params, *, logtarget: Callable, n_samples: int
):
    return run_sampling_nuts(
        key,
        istates,
        tp,
        data,
        logtarget=partial(logtarget, fixed_params=fixed_params),
        n_samples=n_samples,
        max_num_doublings=5,
    )


def main(
    seed: int,
    tag: str,
    n_samples: int = 500,
    shape_noise: float = 0.3,
    sigma_e_int: float = 0.35,
    slen: int = 63,  # adjust depending on HLR bds
    fft_size: int = 256,
    background: float = 1.0,
    initial_step_size: float = 0.1,
    mean_logflux: float = 3.0,
    sigma_logflux: float = 0.4,
    mean_loghlr: float = -0.1,
    sigma_loghlr: float = 0.05,
    min_logflux: float = 2.4,
    mode: str = "short",
):
    assert mode in ("short", "long")
    rng_key = random.key(seed)
    pkey, nkey, rkey = random.split(rng_key, 3)
    pkey1, pkey2 = random.split(pkey)

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)
    fpath = dirpath / f"chain_results_{seed}.npz"

    # setup target density
    draw_fnc = partial(draw_gaussian, slen=slen, fft_size=fft_size)
    _logprior = partial(
        interim_gprops_logprior, sigma_e=sigma_e_int, free_flux_hlr=True, free_dxdy=True
    )
    _loglikelihood = partial(
        gaussian_image_loglikelihood,
        draw_fnc=draw_fnc,
        background=background,
        free_flux_hlr=True,
        free_dxdy=True,
    )
    _logtarget = partial(
        logtarget_images, logprior_fnc=_logprior, loglikelihood_fnc=_loglikelihood
    )

    _run_warmup2 = partial(
        _run_warmup, logtarget=_logtarget, initial_step_size=initial_step_size
    )
    _run_sampling2 = partial(_run_sampling, logtarget=_logtarget, n_samples=n_samples)

    run_warmup = vmap(vmap(jit(_run_warmup2), in_axes=(0, 0, None, None)))
    run_sampling = vmap(vmap(jit(_run_sampling2), in_axes=(0, 0, 0, None, None)))

    sample_prior = partial(
        _sample_prior,
        shape_noise=shape_noise,
        mean_logflux=mean_logflux,
        sigma_logflux=sigma_logflux,
        mean_loghlr=mean_loghlr,
        sigma_loghlr=sigma_loghlr,
        min_logflux=min_logflux,
    )

    results = {}

    opt1 = (1, 1, 5, 10, 25, 50, 100, 250)
    opt2 = (1, 1, 5, 10, 25, 50, 100, 250, 500, 1000)
    all_n_gals = opt1 if mode == "short" else opt2

    for n_gals in all_n_gals:  # repeat 1 == compilation
        print("n_gals:", n_gals)

        pkeys1 = random.split(pkey1, n_gals)
        galaxy_params = vmap(sample_prior)(pkeys1)
        assert galaxy_params["x"].shape == (n_gals,)

        # get images
        draw_params = {**galaxy_params}
        draw_params["f"] = 10 ** draw_params.pop("lf")
        draw_params["hlr"] = 10 ** draw_params.pop("lhlr")
        target_images = get_target_images(
            nkey, draw_params, background=background, slen=slen
        )
        assert target_images.shape == (n_gals, slen, slen)

        true_params = vmap(get_true_params_from_galaxy_params)(galaxy_params)
        pkeys2 = random.split(pkey2, (n_gals, 4))

        # initialize chain positions
        init_positions = vmap(vmap(_init_fnc, in_axes=(0, None, None)))(
            pkeys2, target_images, true_params
        )
        fixed_params = {"x": galaxy_params["x"], "y": galaxy_params["y"]}

        rkeys = random.split(rkey, (n_gals, 4, 2))
        wkeys = rkeys[..., 0]
        skeys = rkeys[..., 1]

        # optimization + warmup
        t1 = time.time()
        init_states, tuned_params, adapt_info = run_warmup(
            wkeys, init_positions, target_images, fixed_params
        )
        t2 = time.time()
        t_warmup = t2 - t1
        tuned_params.pop("max_num_doublings")  # set above, not jittable

        # inference
        t1 = time.time()
        samples, _ = run_sampling(
            skeys, init_states, tuned_params, target_images, fixed_params
        )
        t2 = time.time()
        t_sampling = t2 - t1

        # for logging
        true_params["dx"] = jnp.zeros_like(true_params.pop("x"))
        true_params["dy"] = jnp.zeros_like(true_params.pop("y"))

        n_gals_str = str(n_gals)
        results[n_gals_str] = {}
        results[n_gals_str]["t_warmup"] = t_warmup
        results[n_gals_str]["t_sampling"] = t_sampling

        if n_gals == all_n_gals[-1]:  # no need to save everything just the last one
            results[n_gals_str]["samples"] = samples
            results[n_gals_str]["truth"] = true_params
            results[n_gals_str]["tuned_params"] = tuned_params
            results[n_gals_str]["adapt_position"] = adapt_info.state.position

    save_dataset(results, fpath, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
