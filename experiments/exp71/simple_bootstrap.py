#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import typer
from jax import jit

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import load_dataset_jax, save_dataset
from bpd.jackknife import run_bootstrap_shear_pipeline
from bpd.pipelines import logtarget_shear_and_sn


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_plus_fpath: str = typer.Option(),
    samples_minus_fpath: str = typer.Option(),
    initial_step_size: float = 0.1,
    n_samples: int = 1000,
    n_boots: int = 30,
    no_bar: bool = False,
):
    dirpath = DATA_DIR / "cache_chains" / tag
    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)

    samples_plus_fpath = Path(samples_plus_fpath)
    samples_minus_fpath = Path(samples_minus_fpath)
    assert samples_plus_fpath.exists() and samples_minus_fpath.exists()
    fpath = dirpath / f"g_samples_boots_{seed}.npz"

    dsp = load_dataset_jax(samples_plus_fpath)
    dsm = load_dataset_jax(samples_minus_fpath)

    e1p = dsp["samples"]["e1"]
    e2p = dsp["samples"]["e2"]
    e1e2p = jnp.stack([e1p, e2p], axis=-1)

    e1m = dsm["samples"]["e1"]
    e2m = dsm["samples"]["e2"]
    e1e2m = jnp.stack([e1m, e2m], axis=-1)

    sigma_e_int = dsp["hyper"]["sigma_e_int"]
    assert dsp["hyper"]["shape_noise"] == dsm["hyper"]["shape_noise"]
    assert sigma_e_int == dsm["hyper"]["sigma_e_int"]
    assert jnp.all(dsp["truth"]["e1"] == dsm["truth"]["e1"])
    assert jnp.all(dsp["truth"]["lf"] == dsm["truth"]["lf"])

    rng_key = jax.random.key(seed)
    _logtarget = jit(partial(logtarget_shear_and_sn, sigma_e_int=sigma_e_int))
    raw_pipeline = partial(
        run_inference_nuts,
        init_positions={"g": jnp.array([0.0, 0.0]), "sigma_e": 0.2},
        logtarget=_logtarget,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
        max_num_doublings=3,
    )
    _pipe = jit(raw_pipeline)
    pipe = lambda k, d: _pipe(k, d["e1e2"])

    samples_plus, samples_minus = run_bootstrap_shear_pipeline(
        rng_key,
        post_params_plus={"e1e2": e1e2p},
        post_params_minus={"e1e2": e1e2m},
        shear_pipeline=pipe,
        n_gals=dsp["samples"]["e1"].shape[0],
        n_boots=n_boots,
        no_bar=no_bar,
    )

    assert samples_plus["g"].shape == (n_boots, n_samples, 2)
    assert samples_minus["g"].shape == (n_boots, n_samples, 2)

    save_dataset(
        {
            "plus": {
                "g1": samples_plus["g"][:, 0],
                "g2": samples_plus["g"][:, 1],
                "sigma_e": samples_plus["sigma_e"],
            },
            "minus": {
                "g1": samples_minus["g"][:, 0],
                "g2": samples_minus["g"][:, 1],
                "sigma_e": samples_minus["sigma_e"],
            },
        },
        fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
