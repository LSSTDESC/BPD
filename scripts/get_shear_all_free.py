#!/usr/bin/env python3
"""This file creates toy samples of ellipticities and saves them to .hdf5 file."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import typer
from jax import Array

from bpd import DATA_DIR
from bpd.io import load_dataset_jax
from bpd.pipelines import pipeline_shear_inference
from bpd.prior import interim_gprops_logprior, true_all_params_logprior


def interim_logprior(post_params: dict[str, Array], sigma_e_int: float):
    # we do not evaluate dxdy as we assume it's the same as the true prior and they cancel
    return interim_gprops_logprior(
        post_params, sigma_e=sigma_e_int, free_flux_hlr=True, free_dxdy=False
    )


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_fname: str = typer.Option(),
    mode: str = "",
    initial_step_size: float = 1e-2,
    n_samples: int = 3000,
    overwrite: bool = False,
):
    assert mode in ("plus", "minus", "")
    mode_txt = f"_{mode}" if mode else ""

    # directory structure
    dirpath = DATA_DIR / "cache_chains" / tag
    samples_fpath = dirpath / samples_fname
    assert dirpath.exists()
    assert samples_fpath.exists(), "ellipticity samples file does not exist"
    out_fpath = dirpath / f"g_samples_{seed}{mode_txt}.npy"

    if out_fpath.exists() and not overwrite:
        raise IOError("overwriting...")

    ds = load_dataset_jax(samples_fpath)

    # data
    samples = ds["samples"]
    post_params = {
        "lf": samples["lf"],
        "lhlr": samples["lhlr"],
        "e1": samples["e1"],
        "e2": samples["e2"],
    }

    # prior parameters
    hyper = ds["hyper"]
    g1 = hyper["g1"]
    g2 = hyper["g2"]
    true_g = jnp.array([g1, g2])
    sigma_e_int = hyper["sigma_e_int"]
    sigma_e = hyper["sigma_e"]
    mean_logflux = hyper["mean_logflux"]
    sigma_logflux = hyper["sigma_logflux"]
    mean_loghlr = hyper["mean_loghlr"]
    sigma_loghlr = hyper["sigma_loghlr"]

    # setup priors
    logprior_fnc = partial(
        true_all_params_logprior,
        sigma_e=sigma_e,
        mean_logflux=mean_logflux,
        sigma_logflux=sigma_logflux,
        mean_loghlr=mean_loghlr,
        sigma_loghlr=sigma_loghlr,
    )
    interim_logprior_fnc = partial(interim_logprior, sigma_e_int=sigma_e_int)

    rng_key = jax.random.key(seed)
    g_samples = pipeline_shear_inference(
        rng_key,
        post_params,
        init_g=true_g,
        logprior=logprior_fnc,
        interim_logprior=interim_logprior_fnc,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )

    np.save(out_fpath, np.asarray(g_samples))


if __name__ == "__main__":
    typer.run(main)
