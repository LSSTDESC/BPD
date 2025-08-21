#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import jax.numpy as jnp
import typer
from jax import jit, random, vmap

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import load_dataset_jax, save_dataset
from bpd.pipelines import init_all_params, logtarget_all_free


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_plus_fpath: str = typer.Option(),
    samples_minus_fpath: str = typer.Option(),
    initial_step_size: float = 0.01,
    n_splits: int = 500,
    n_samples: int = 1000,
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

    samples_plus = ds_plus["samples"]
    ppp = {
        "lf": samples_plus["lf"],
        "lhlr": samples_plus["lhlr"],
        "e1": samples_plus["e1"],
        "e2": samples_plus["e2"],
    }

    samples_minus = ds_minus["samples"]
    ppm = {
        "lf": samples_minus["lf"],
        "lhlr": samples_minus["lhlr"],
        "e1": samples_minus["e1"],
        "e2": samples_minus["e2"],
    }

    g1 = ds_plus["hyper"]["g1"]
    g2 = ds_plus["hyper"]["g2"]
    true_g = jnp.array([g1, g2])
    sigma_e = ds_plus["hyper"]["shape_noise"]
    sigma_e_int = ds_plus["hyper"]["sigma_e_int"]
    mean_logflux = ds_plus["hyper"]["mean_logflux"]
    sigma_logflux = ds_plus["hyper"]["sigma_logflux"]
    a_logflux = ds_plus["hyper"]["a_logflux"]
    mean_loghlr = ds_plus["hyper"]["mean_loghlr"]
    sigma_loghlr = ds_plus["hyper"]["sigma_loghlr"]

    g1m = ds_minus["hyper"]["g1"]
    g2m = ds_minus["hyper"]["g2"]
    true_gm = jnp.array([g1m, g2m])

    assert jnp.all(true_g == -true_gm)
    assert sigma_e == ds_minus["hyper"]["shape_noise"]
    assert sigma_e_int == ds_minus["hyper"]["sigma_e_int"]
    assert mean_logflux == ds_minus["hyper"]["mean_logflux"]
    assert mean_loghlr == ds_minus["hyper"]["mean_loghlr"]
    assert sigma_logflux == ds_minus["hyper"]["sigma_logflux"]
    assert sigma_loghlr == ds_minus["hyper"]["sigma_loghlr"]
    assert a_logflux == ds_minus["hyper"]["a_logflux"]
    assert jnp.all(ds_plus["truth"]["e1"] == ds_minus["truth"]["e1"])
    assert jnp.all(ds_plus["truth"]["lf"] == ds_minus["truth"]["lf"])

    k1, k2 = random.split(rng_key)
    _logtarget = jit(partial(logtarget_all_free, sigma_e_int=sigma_e_int))

    # Reshape samples
    split_size = ppp["e1"].shape[0] // n_splits
    assert split_size * n_splits == ppp["e1"].shape[0], "dimensions do not match"
    ppp = {k: jnp.reshape(v, (n_splits, split_size, 300)) for k, v in ppp.items()}
    ppm = {k: jnp.reshape(v, (n_splits, split_size, 300)) for k, v in ppm.items()}

    # initialize positions randomly uniform from true parameters
    k1s = random.split(k1, n_splits)
    true_params = {
        "sigma_e": sigma_e,
        "mean_logflux": mean_logflux,
        "sigma_logflux": sigma_logflux,
        "a_logflux": a_logflux,
        "mean_loghlr": mean_loghlr,
        "sigma_loghlr": sigma_loghlr,
    }
    _init_fnc = partial(
        init_all_params,
        true_params=true_params,
        p=0.1,
    )
    init_positions = vmap(_init_fnc)(k1s)
    init_positions["g"] = jnp.zeros((n_splits, 2))

    # setup pipeline
    pipe = jit(
        partial(
            run_inference_nuts,
            logtarget=_logtarget,
            n_samples=n_samples,
            initial_step_size=initial_step_size,
            max_num_doublings=7,
            n_warmup_steps=1000,
        )
    )

    # jit function quickly
    print("JITting function...")
    _ = pipe(
        k2,
        data={k: v[0] for k, v in ppp.items()},
        init_positions={k: v[0] for k, v in init_positions.items()},
    )

    # run shear inference pipeline
    k2s = random.split(k2, n_splits)

    print("Running inference...")
    samples_plus = vmap(pipe)(k2s, ppp, init_positions)
    samples_minus = vmap(pipe)(k2s, ppm, init_positions)
    assert samples_plus["g"].shape == (n_splits, n_samples, 2), (
        "shear samples do not match"
    )
    save_dataset(
        {
            "plus": {
                "g1": samples_plus["g"][:, :, 0],
                "g2": samples_plus["g"][:, :, 1],
                **samples_plus,
            },
            "minus": {
                "g1": samples_minus["g"][:, :, 0],
                "g2": samples_minus["g"][:, :, 1],
                **samples_minus,
            },
        },
        fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
