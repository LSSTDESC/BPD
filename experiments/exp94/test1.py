#!/usr/bin/env python3


import os
from functools import partial

os.environ["JAX_ENABLE_X64"] = "True"

import time

import jax
import jax.numpy as jnp
import numpy as np
import typer

# compute effective sample size and r-hat
from jax import Array, jit, random, vmap
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats
from ornax.hmc import ensemble_hmc
from tqdm import tqdm

from bpd import DATA_DIR
from bpd.draw import draw_exponential
from bpd.io import save_dataset
from bpd.prior import ellip_prior_e1e2
from bpd.sample import (
    get_target_images,
    get_true_params_from_galaxy_params,
    sample_galaxy_params_skew,
)
from bpd.utils import DEFAULT_HYPERPARAMS


def _init_fnc(key: PRNGKeyArray, *, data: Array, true_params: dict):
    image = data
    assert image.ndim == 2
    assert image.shape[0] == image.shape[1]
    flux = image.sum()

    k1, k2, k3, k4, k5, k6 = random.split(key, 6)

    _lf = jnp.log10(flux)
    lf = random.uniform(k1, shape=(), minval=_lf - 0.01, maxval=_lf + 0.01)

    tlhlr = true_params["lhlr"]
    lhlr = random.uniform(k2, shape=(), minval=tlhlr - 0.015, maxval=tlhlr + 0.015)

    te1 = true_params["e1"]
    e1 = random.uniform(k3, shape=(), minval=te1 - 0.1, maxval=te1 + 0.1)

    te2 = true_params["e2"]
    e2 = random.uniform(k4, shape=(), minval=te2 - 0.1, maxval=te2 + 0.1)
    return {
        "lf": lf,
        "lhlr": lhlr,
        "e1": e1,
        "e2": e2,
        "dx": random.uniform(k5, shape=(), minval=-0.1, maxval=0.1),
        "dy": random.uniform(k6, shape=(), minval=-0.1, maxval=0.1),
    }


def initialize_params(key, image, true_params):
    """For vmap"""
    return _init_fnc(key, data=image, true_params=true_params)


def logprior_flat(
    x: Array,
    *,
    sigma_e: float,
    sigma_x: float = 0.5,  # pixels
    flux_bds: tuple = (-1.0, 9.0),
    hlr_bds: tuple = (-2.0, 1.0),
):
    assert x.shape == (6,)
    prior = jnp.array(0.0)
    lf, lhlr, dx, dy, e1, e2 = x

    f1, f2 = flux_bds
    prior += stats.uniform.logpdf(lf, f1, f2 - f1)

    h1, h2 = hlr_bds
    prior += stats.uniform.logpdf(lhlr, h1, h2 - h1)

    prior += stats.norm.logpdf(dx, loc=0.0, scale=sigma_x)
    prior += stats.norm.logpdf(dy, loc=0.0, scale=sigma_x)

    e1e2 = jnp.stack((e1, e2), axis=-1)
    prior += jnp.log(ellip_prior_e1e2(e1e2, sigma=sigma_e))

    return prior


def gaussian_image_loglikelihood_flat(
    x: Array,
    data: Array,
    fixed_params: dict[str, Array],
    *,
    draw_fnc,
    background: float,
):
    assert x.shape == (6,)
    assert data.ndim == 2
    _draw_params = {}
    lf, lhlr, dx, dy, e1, e2 = x

    _draw_params["x"] = dx + fixed_params["x"]
    _draw_params["y"] = dy + fixed_params["y"]
    _draw_params["f"] = 10**lf
    _draw_params["hlr"] = 10**lhlr
    _draw_params["e1"] = e1
    _draw_params["e2"] = e2
    model = draw_fnc(**_draw_params)
    likelihood_pp = stats.norm.logpdf(data, loc=model, scale=jnp.sqrt(background))
    return jnp.sum(likelihood_pp)


def my_logtarget(x: Array, *, data, fixed_params, logprior, loglikelihood):
    return logprior(x) + loglikelihood(x, data, fixed_params)


def run_ensemble_hmc(
    rng_key,
    data,
    init_positions,
    fixed_params,
    *,
    logprior,
    loglikelihood,
    n_samples=500,
    n_walkers=12,
    leapfrog_step_size=None,
    n_leapfrog_steps=None,
):
    _logtarget = partial(
        my_logtarget,
        data=data,
        fixed_params=fixed_params,
        logprior=logprior,
        loglikelihood=loglikelihood,
    )
    chain, acc, loglike = ensemble_hmc(
        rng_key,
        _logtarget,
        n_dims=6,
        n_samples=n_samples,
        params_init=init_positions,
        verbose=False,
        n_walkers=n_walkers,
        leapfrog_step_size=leapfrog_step_size,
        n_leapfrog_steps=n_leapfrog_steps,
    )

    return chain, acc, loglike


def main(seed: int = 42, overwrite: bool = False):
    # directory structure
    dirpath = DATA_DIR / "cache_chains" / "exp94"
    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)

    n_gals_list = [1, 5, 10, 25, 50, 100, 250, 500]
    # n_gals_list = [1, 5]
    max_n_gals = max(n_gals_list)

    n_samples: int = 600
    sigma_e_int: float = 0.3
    g1: float = 0.02
    g2: float = 0.0
    slen: int = 63
    fft_size: int = 256
    background: float = 1.0

    _params_order = ("lf", "lhlr", "dx", "dy", "e1", "e2")

    rng_key = random.key(seed)
    pkey, nkey, gkey = random.split(rng_key, 3)

    # to avoid chains with only a few galaxies getting their timing biased, we cap the lf
    # at 1000.

    # galaxy parameters from prior
    galaxy_params = sample_galaxy_params_skew(
        pkey, n=max_n_gals, g1=g1, g2=g2, **DEFAULT_HYPERPARAMS
    )
    assert galaxy_params["x"].shape == (max_n_gals,)
    assert galaxy_params["e1"].shape == (max_n_gals,)
    galaxy_params["lf"] = jnp.clip(galaxy_params["lf"], max=3)

    # now get corresponding target images
    draw_params = {**galaxy_params}
    draw_params["f"] = 10 ** draw_params.pop("lf")
    draw_params["hlr"] = 10 ** draw_params.pop("lhlr")
    target_images = get_target_images(
        nkey, draw_params, background=background, slen=slen, draw_type="exponential"
    )
    assert target_images.shape == (max_n_gals, slen, slen)

    # interim samples are on 'sheared ellipticity'
    true_params = vmap(get_true_params_from_galaxy_params)(galaxy_params)
    true_params["dx"] = jnp.zeros_like(true_params["x"])
    true_params["dy"] = jnp.zeros_like(true_params["y"])
    fixed_params = {
        "x": true_params.pop("x"),
        "y": true_params.pop("y"),
    }

    # setup prior and likelihood
    _logprior = partial(logprior_flat, sigma_e=sigma_e_int)
    _draw_fnc = partial(draw_exponential, slen=slen, fft_size=fft_size)
    _loglikelihood = partial(
        gaussian_image_loglikelihood_flat,
        draw_fnc=_draw_fnc,
        background=background,
    )

    n_dims = len(_params_order)  # number of parameters
    n_walkers = max(2 * n_dims, 10)
    leapfrog_step_size = 0.05
    n_leapfrog_steps = 20

    init_key, run_key = random.split(gkey)

    _init_keys = random.split(init_key, (max_n_gals, n_walkers))
    _init_vectorized = vmap(vmap(initialize_params, in_axes=(0, None, None)))
    init_params = _init_vectorized(_init_keys, target_images, true_params)

    # convert to array in the correct order
    _init_params_array = np.zeros((max_n_gals, n_walkers, 6))
    for ii, k in enumerate(_params_order):
        _init_params_array[..., ii] = init_params[k]

    _init_params_array = jnp.array(_init_params_array, device=jax.devices()[0])

    run_keys = random.split(run_key, max_n_gals)

    # jit
    _pipe = partial(
        run_ensemble_hmc,
        logprior=_logprior,
        loglikelihood=_loglikelihood,
        n_walkers=n_walkers,
        leapfrog_step_size=leapfrog_step_size,
        n_leapfrog_steps=n_leapfrog_steps,
        n_samples=n_samples,
    )
    pipe = jit(vmap(_pipe))
    _, _, _ = pipe(
        run_keys[0, None],
        target_images[0, None],
        _init_params_array[0, None],
        {k: v[0, None] for k, v in fixed_params.items()},
    )

    out = {}

    for n_gals in tqdm(n_gals_list, desc="N Galaxies"):
        t1 = time.time()
        samples, _, _ = pipe(
            run_keys[:n_gals],
            target_images[:n_gals],
            _init_params_array[:n_gals],
            {k: v[:n_gals] for k, v in fixed_params.items()},
        )
        t2 = time.time()

        if n_gals == max_n_gals:
            samples_dict = {}
            for jj, p in enumerate(_params_order):
                samples_dict[p] = (
                    samples[..., jj].transpose(0, 2, 1).reshape(max_n_gals, 12, -1)
                )
            save_dataset(samples_dict, dirpath / "samples.npz", overwrite=overwrite)

        out[str(n_gals)] = t2 - t1

    save_dataset(out, dirpath / "timing.npz", overwrite=overwrite)


if __name__ == "__main__":
    typer.run(main)
