#!/usr/bin/env python3
from functools import partial

import jax.numpy as jnp
import typer
from jax import jit, random, vmap

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import save_dataset
from bpd.pipelines import (
    logtarget_toy_ellips,
)
from bpd.sample import sample_noisy_ellipticities_unclipped


def main(
    seed: int,
    tag: str,
    n_gals: int = 10_000,
    shape_noise: float = 1e-2,
    sigma_e_int: float = 5e-2,
    sigma_m: float = 1e-5,
    g1: float = 0.02,
    g2: float = 0.0,
    n_samples: int = 300,
):
    k = random.key(seed)
    k1, k2 = random.split(k)
    true_g = jnp.array([g1, g2])

    dirpath = DATA_DIR / "cache_chains" / tag
    if not dirpath.exists():
        dirpath.mkdir(exist_ok=True)
    out_fpath = dirpath / "shape_samples.npz"

    # same ellipticities with same noise, but opposite shear
    e_obs_plus, e_sheared_plus, _ = sample_noisy_ellipticities_unclipped(
        k1, g=true_g, sigma_m=sigma_m, sigma_e=shape_noise, n=n_gals
    )
    e_obs_minus, e_sheared_minus, _ = sample_noisy_ellipticities_unclipped(
        k1, g=-true_g, sigma_m=sigma_m, sigma_e=shape_noise, n=n_gals
    )

    _logtarget = partial(logtarget_toy_ellips, sigma_m=sigma_m, sigma_e_int=sigma_e_int)

    k2s = random.split(k2, (n_gals, 2))

    _pipe = jit(
        partial(
            run_inference_nuts,
            logtarget=_logtarget,
            n_samples=n_samples,
            initial_step_size=0.01,
            max_num_doublings=2,
            n_warmup_steps=500,
        )
    )
    pipe = vmap(_pipe, in_axes=(0, 0, 0))

    _ = pipe(k2s[:2, 0], e_obs_plus[:2], e_sheared_plus[:2])
    e_post_plus = pipe(k2s[:, 0], e_obs_plus, e_sheared_plus)
    e_post_minus = pipe(k2s[:, 1], e_obs_minus, e_sheared_minus)

    save_dataset(
        {
            "e1e2p": e_post_plus,
            "e1e2m": e_post_minus,
            "g1": g1,
            "g2": g2,
            "sigma_e": shape_noise,
            "sigma_e_int": sigma_e_int,
            "sigma_m": sigma_m,
        },
        out_fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
