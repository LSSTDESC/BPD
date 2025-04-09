#!/usr/bin/env python3

from functools import partial

import jax
import jax.numpy as jnp
import typer
from jax import Array, random

from bpd import DATA_DIR
from bpd.io import load_dataset_jax, save_dataset
from bpd.jackknife import run_bootstrap_shear_pipeline
from bpd.pipelines import pipeline_shear_inference
from bpd.prior import interim_gprops_logprior, true_all_params_trunc_logprior


def _interim_logprior(post_params: dict[str, Array], sigma_e_int: float):
    # we do not evaluate dxdy as we assume it's the same as the true prior and they cancel
    return interim_gprops_logprior(
        post_params, sigma_e=sigma_e_int, free_flux_hlr=True, free_dxdy=False
    )


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_plus_fname: str = typer.Option(),
    samples_minus_fname: str = typer.Option(),
    initial_step_size: float = 0.1,
    n_samples: int = 1000,
    n_boots: int = 25,
    no_bar: bool = False,
):
    dirpath = DATA_DIR / "cache_chains" / tag
    samples_plus_fpath = dirpath / samples_plus_fname
    samples_minus_fpath = dirpath / samples_minus_fname
    assert samples_plus_fpath.exists() and samples_minus_fpath.exists()
    fpath = dirpath / f"g_samples_boots_{seed}.npz"

    rng_key = random.key(seed)

    dsp = load_dataset_jax(samples_plus_fpath)
    dsm = load_dataset_jax(samples_minus_fpath)

    # positions are not used as we assume true and interim prior cancels
    samples_plus = dsp["samples"]
    post_params_plus = {
        "lf": samples_plus["lf"],
        "lhlr": samples_plus["lhlr"],
        "e1": samples_plus["e1"],
        "e2": samples_plus["e2"],
    }

    samples_minus = dsm["samples"]
    post_params_minus = {
        "lf": samples_minus["lf"],
        "lhlr": samples_minus["lhlr"],
        "e1": samples_minus["e1"],
        "e2": samples_minus["e2"],
    }

    g1 = dsp["hyper"]["g1"]
    g2 = dsp["hyper"]["g2"]
    true_g = jnp.array([g1, g2])
    sigma_e = dsp["hyper"]["sigma_e"]
    sigma_e_int = dsp["hyper"]["sigma_e_int"]
    mean_logflux = dsp["hyper"]["mean_logflux"]
    sigma_logflux = dsp["hyper"]["sigma_logflux"]
    min_logflux = dsp["hyper"]["min_logflux"]
    mean_loghlr = dsp["hyper"]["mean_loghlr"]
    sigma_loghlr = dsp["hyper"]["sigma_loghlr"]

    g1m = dsm["hyper"]["g1"]
    g2m = dsm["hyper"]["g2"]
    true_gm = jnp.array([g1m, g2m])

    assert jnp.all(true_g == -true_gm)
    assert sigma_e == dsm["hyper"]["sigma_e"]
    assert sigma_e_int == dsm["hyper"]["sigma_e_int"]
    assert mean_logflux == dsm["hyper"]["mean_logflux"]
    assert mean_loghlr == dsm["hyper"]["mean_loghlr"]
    assert jnp.all(dsp["truth"]["e1"] == dsm["truth"]["e1"])
    assert jnp.all(dsp["truth"]["f"] == dsm["truth"]["f"])

    logprior_fnc = partial(
        true_all_params_trunc_logprior,
        sigma_e=sigma_e,
        mean_logflux=mean_logflux,
        sigma_logflux=sigma_logflux,
        mean_loghlr=mean_loghlr,
        sigma_loghlr=sigma_loghlr,
        min_logflux=min_logflux,
    )
    interim_logprior_fnc = partial(_interim_logprior, sigma_e_int=sigma_e_int)

    raw_pipeline = partial(
        pipeline_shear_inference,
        init_g=jnp.array([0.0, 0.0]),
        logprior=logprior_fnc,
        interim_logprior=interim_logprior_fnc,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )

    @jax.jit
    def pipe(k, d):
        out = raw_pipeline(k, d)
        return {"g1": out[..., 0], "g2": out[..., 1]}

    samples_plus, samples_minus = run_bootstrap_shear_pipeline(
        rng_key,
        post_params_plus=post_params_plus,
        post_params_minus=post_params_minus,
        shear_pipeline=pipe,
        n_gals=samples_plus["e1"].shape[0],
        n_boots=n_boots,
        no_bar=no_bar,
    )

    gp = jnp.stack([samples_plus["g1"], samples_plus["g2"]], axis=-1)
    gm = jnp.stack([samples_minus["g1"], samples_minus["g2"]], axis=-1)

    assert gp.shape == (n_boots, n_samples, 2)
    assert gm.shape == (n_boots, n_samples, 2)

    save_dataset({"gp": gp, "gm": gm}, fpath, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
