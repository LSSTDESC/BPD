#!/usr/bin/env python3
from functools import partial

import jax.numpy as jnp
import numpy as np
import typer
from jax import jit, random, vmap

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import save_dataset
from bpd.pipelines import logtarget_toy_ellips, pipeline_shear_inference_simple
from bpd.sample import sample_noisy_ellipticities_unclipped


def get_mbias_mc(
    seed: int,
    *,
    shape_noise: float,
    sigma_e_int: float,
    n_gals: int,
    g1=0.02,
    g2=0.0,
    sigma_m=1e-5,
    n_samples_gals=300,
    n_samples_shear=1000,
):
    k = random.key(seed)
    k1, k2, k3 = random.split(k, 3)
    true_g = jnp.array([g1, g2])

    # same ellipticities with same noise, but opposite shear
    e_obs_plus, _, e_int = sample_noisy_ellipticities_unclipped(
        k1, g=true_g, sigma_m=sigma_m, sigma_e=shape_noise, n=n_gals
    )
    e_obs_minus, _, _ = sample_noisy_ellipticities_unclipped(
        k1, g=-true_g, sigma_m=sigma_m, sigma_e=shape_noise, n=n_gals
    )

    _logtarget = partial(logtarget_toy_ellips, sigma_m=sigma_m, sigma_e_int=sigma_e_int)

    k2s = random.split(k2, n_gals)

    _pipe = jit(
        partial(
            run_inference_nuts,
            logtarget=_logtarget,
            n_samples=n_samples_gals,
            initial_step_size=0.01,
            max_num_doublings=2,
            n_warmup_steps=500,
        )
    )
    pipe = vmap(_pipe, in_axes=(0, 0, 0))

    _ = pipe(k2s[:2], e_obs_plus[:2], e_int[:2])

    e1e2p = pipe(k2s, e_obs_plus, e_int)
    e1e2m = pipe(k2s, e_obs_minus, e_int)

    raw_pipeline = partial(
        pipeline_shear_inference_simple,
        init_g=jnp.array([0.0, 0.0]),
        sigma_e=shape_noise,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples_shear,
        initial_step_size=0.01,
    )
    pipe = jit(raw_pipeline)

    gp = pipe(k3, e1e2p)
    gm = pipe(k3, e1e2m)

    assert gp.ndim == 2
    m = (gp[..., 0].mean() - gm[..., 0].mean()) / 2 / 0.02 - 1

    return m, gp, gm


def main(
    seed: int,
    tag: str = typer.Option(),
    n_exps: int = typer.Option(),
    shape_noise: float = 1e-2,
    sigma_e_int: float = 5e-2,
    n_gals: int = 10_000,
):
    # load data
    dirpath = DATA_DIR / "cache_chains" / tag
    fpath = dirpath / f"mc_exp_samples_{seed}.npz"

    # get array of seeds from numpy
    rng = np.random.RandomState(seed)
    seeds = rng.uniform(low=0, high=2**31, size=(n_exps,)).astype(int)
    seeds = jnp.array(seeds)

    _func = jit(
        partial(
            get_mbias_mc,
            shape_noise=shape_noise,
            sigma_e_int=sigma_e_int,
            n_gals=n_gals,
        )
    )

    ms, gps, gms = vmap(_func)(seeds)

    save_dataset({"m": ms, "gp": gps, "gm": gms}, fpath, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
