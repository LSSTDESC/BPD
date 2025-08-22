#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import jax.numpy as jnp
import typer
from jax import jit, random, vmap

from bpd import DATA_DIR
from bpd.io import load_dataset_jax, save_dataset
from bpd.pipelines import pipeline_shear_inference
from bpd.prior import interim_gprops_logprior, true_all_params_skew_logprior
from bpd.utils import process_in_batches


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_plus_fpath: str = typer.Option(),
    samples_minus_fpath: str = typer.Option(),
    initial_step_size: float = 0.001,
    n_splits: int = 500,
    n_gals: int | None = None,
):
    rng_key = random.key(seed)

    dirpath = DATA_DIR / "cache_chains" / tag
    pfpath = Path(samples_plus_fpath)
    mfpath = Path(samples_minus_fpath)
    fpath = dirpath / f"g_samples_{seed}_errs.npz"

    assert dirpath.exists()
    assert pfpath.exists(), "ellipticity samples file does not exist"
    assert mfpath.exists(), "ellipticity samples file does not exist"

    ds_plus = load_dataset_jax(pfpath)
    ds_minus = load_dataset_jax(mfpath)

    if n_gals is None:
        n_gals = ds_plus["samples"]["e1"].shape[0]

    ppp = {
        "lf": ds_plus["samples"]["lf"][:n_gals],
        "lhlr": ds_plus["samples"]["lhlr"][:n_gals],
        "e1": ds_plus["samples"]["e1"][:n_gals],
        "e2": ds_plus["samples"]["e2"][:n_gals],
    }

    ppm = {
        "lf": ds_minus["samples"]["lf"][:n_gals],
        "lhlr": ds_minus["samples"]["lhlr"][:n_gals],
        "e1": ds_minus["samples"]["e1"][:n_gals],
        "e2": ds_minus["samples"]["e2"][:n_gals],
    }

    sigma_e = ds_plus["hyper"]["shape_noise"]
    sigma_e_int = ds_plus["hyper"]["sigma_e_int"]
    mean_logflux = ds_plus["hyper"]["mean_logflux"]
    sigma_logflux = ds_plus["hyper"]["sigma_logflux"]
    mean_loghlr = ds_plus["hyper"]["mean_loghlr"]
    sigma_loghlr = ds_plus["hyper"]["sigma_loghlr"]
    a_logflux = ds_plus["hyper"]["a_logflux"]

    assert ds_plus["hyper"]["g1"] == -ds_minus["hyper"]["g1"]
    assert ds_plus["hyper"]["g2"] == -ds_minus["hyper"]["g2"]
    assert sigma_e == ds_minus["hyper"]["shape_noise"]
    assert sigma_e_int == ds_minus["hyper"]["sigma_e_int"]
    assert mean_logflux == ds_minus["hyper"]["mean_logflux"]
    assert mean_loghlr == ds_minus["hyper"]["mean_loghlr"]
    assert jnp.all(ds_plus["truth"]["e1"] == ds_minus["truth"]["e1"])
    assert jnp.all(ds_plus["truth"]["lf"] == ds_minus["truth"]["lf"])

    logprior_fnc = partial(
        true_all_params_skew_logprior,
        sigma_e=sigma_e,
        mean_logflux=mean_logflux,
        sigma_logflux=sigma_logflux,
        a_logflux=a_logflux,
        mean_loghlr=mean_loghlr,
        sigma_loghlr=sigma_loghlr,
    )
    interim_logprior_fnc = partial(
        interim_gprops_logprior,
        sigma_e=sigma_e_int,
        free_flux_hlr=True,
        free_dxdy=False,  # dxdy are not used in shear inference
    )

    raw_pipeline = partial(
        pipeline_shear_inference,
        init_g=jnp.array([0.0, 0.0]),
        logprior=logprior_fnc,
        interim_logprior=interim_logprior_fnc,
        n_samples=1000,
        initial_step_size=initial_step_size,
        max_num_doublings=2,
    )
    pipe = jit(raw_pipeline)

    split_size = ppp["e1"].shape[0] // n_splits
    assert split_size * n_splits == ppp["e1"].shape[0], "dimensions do not match"

    # Reshape samples
    ppp = {k: jnp.reshape(v, (n_splits, split_size, -1)) for k, v in ppp.items()}
    ppm = {k: jnp.reshape(v, (n_splits, split_size, -1)) for k, v in ppm.items()}

    # run shear inference pipeline
    keys = random.split(rng_key, n_splits)

    gp = process_in_batches(
        vmap(pipe), keys, ppp, n_points=ppp["e1"].shape[0], batch_size=100
    )
    gm = process_in_batches(
        vmap(pipe), keys, ppm, n_points=ppm["e1"].shape[0], batch_size=100
    )

    assert gp.shape == (n_splits, 1000, 2), "shear samples do not match"
    assert gm.shape == (n_splits, 1000, 2), "shear samples do not match"

    save_dataset(
        {
            "plus": {"g1": gp[:, :, 0], "g2": gp[:, :, 1]},
            "minus": {"g1": gm[:, :, 0], "g2": gm[:, :, 1]},
        },
        fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
