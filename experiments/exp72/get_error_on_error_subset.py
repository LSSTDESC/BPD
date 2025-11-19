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
    n_repeats: int = 10,
    n_samples: int = 1000,
):
    rng_key = random.key(seed)
    k1, k2 = random.split(rng_key, 2)

    dirpath = DATA_DIR / "cache_chains" / tag
    pfpath = Path(samples_plus_fpath)
    mfpath = Path(samples_minus_fpath)
    fpath = dirpath / f"subset_error_on_error_{seed}.npz"

    assert dirpath.exists()
    assert pfpath.exists(), "ellipticity samples file does not exist"
    assert mfpath.exists(), "ellipticity samples file does not exist"

    dsp = load_dataset_jax(pfpath)
    dsm = load_dataset_jax(mfpath)

    total_n_gals = dsp["samples"]["e1"].shape[0]

    sigma_e = dsp["hyper"]["shape_noise"]
    sigma_e_int = dsp["hyper"]["sigma_e_int"]
    mean_logflux = dsp["hyper"]["mean_logflux"]
    sigma_logflux = dsp["hyper"]["sigma_logflux"]
    mean_loghlr = dsp["hyper"]["mean_loghlr"]
    sigma_loghlr = dsp["hyper"]["sigma_loghlr"]
    a_logflux = dsp["hyper"]["a_logflux"]

    assert dsp["hyper"]["g1"] == -dsm["hyper"]["g1"]
    assert dsp["hyper"]["g2"] == -dsm["hyper"]["g2"]
    assert sigma_e == dsm["hyper"]["shape_noise"]
    assert sigma_e_int == dsm["hyper"]["sigma_e_int"]
    assert mean_logflux == dsm["hyper"]["mean_logflux"]
    assert mean_loghlr == dsm["hyper"]["mean_loghlr"]
    assert jnp.all(dsp["truth"]["e1"] == dsm["truth"]["e1"])
    assert jnp.all(dsp["truth"]["lf"] == dsm["truth"]["lf"])

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
        n_samples=n_samples,
        initial_step_size=initial_step_size,
        max_num_doublings=2,
    )
    pipe = jit(raw_pipeline)

    k1s = random.split(k1, n_repeats)
    k2s = random.split(k2, n_repeats)
    gps = []
    gms = []

    for ii in range(n_repeats):
        # randomly shuffle galaxies
        subset = random.choice(
            k1s[ii], jnp.arange(total_n_gals), shape=(total_n_gals,), replace=False
        )

        _ppp = {
            "lf": dsp["samples"]["lf"][subset],
            "lhlr": dsp["samples"]["lhlr"][subset],
            "e1": dsp["samples"]["e1"][subset],
            "e2": dsp["samples"]["e2"][subset],
        }

        _ppm = {
            "lf": dsm["samples"]["lf"][subset],
            "lhlr": dsm["samples"]["lhlr"][subset],
            "e1": dsm["samples"]["e1"][subset],
            "e2": dsm["samples"]["e2"][subset],
        }

        split_size = _ppp["e1"].shape[0] // n_splits
        assert split_size * n_splits == _ppp["e1"].shape[0], "dimensions do not match"

        # Reshape samples
        ppp = {k: jnp.reshape(v, (n_splits, split_size, -1)) for k, v in _ppp.items()}
        ppm = {k: jnp.reshape(v, (n_splits, split_size, -1)) for k, v in _ppm.items()}

        # run shear inference pipeline
        keys = random.split(k2s[ii], n_splits)

        gp = process_in_batches(
            vmap(pipe), keys, ppp, n_points=ppp["e1"].shape[0], batch_size=50
        )
        gm = process_in_batches(
            vmap(pipe), keys, ppm, n_points=ppm["e1"].shape[0], batch_size=50
        )
        gps.append(gp)
        gms.append(gm)

    gp = jnp.stack(gps, axis=0)
    gm = jnp.stack(gms, axis=0)

    save_dataset(
        {
            "plus": {"g1": gp[..., 0], "g2": gp[..., 1]},
            "minus": {"g1": gm[..., 0], "g2": gm[..., 1]},
        },
        fpath,
        overwrite=True,
    )

    assert gp.shape == (n_repeats, n_splits, n_samples, 2), "shear samples do not match"
    assert gm.shape == (n_repeats, n_splits, n_samples, 2), "shear samples do not match"


if __name__ == "__main__":
    typer.run(main)
