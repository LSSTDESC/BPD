#!/usr/bin/env python3

from functools import partial

import jax
import jax.numpy as jnp
import typer
from jax import jit

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import load_dataset_jax, save_dataset
from bpd.jackknife import run_jackknife_shear_pipeline
from bpd.pipelines import logtarget_shear_and_sn


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_plus_fname: str = typer.Option(),
    samples_minus_fname: str = typer.Option(),
    initial_step_size: float = 0.1,
    n_samples: int = 1000,
    overwrite: bool = False,
    n_jacks: int = 100,
    no_bar: bool = False,
    start: int = 0,
    end: int = 100,
):
    dirpath = DATA_DIR / "cache_chains" / tag
    samples_plus_fpath = dirpath / samples_plus_fname
    samples_minus_fpath = dirpath / samples_minus_fname
    assert samples_plus_fpath.exists() and samples_minus_fpath.exists()

    if start == 0 and end == n_jacks:
        fpath = dirpath / f"g_samples_jack_{seed}.npz"
    else:
        fpath = dirpath / f"g_samples_jack_{seed}_{start}.npz"

    if fpath.exists() and not overwrite:
        raise IOError("overwriting...")

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

    # run
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
    pipe = jit(raw_pipeline)

    g_plus, g_minus = run_jackknife_shear_pipeline(
        rng_key,
        post_params_plus={"e1e2": e1e2p},
        post_params_minus={"e1e2": e1e2m},
        shear_pipeline=pipe,
        n_gals=e1e2p.shape[0],
        n_jacks=n_jacks,
        start=start,
        end=end,
        no_bar=no_bar,
    )

    assert g_plus.shape[1:] == (n_samples, 2)
    assert g_minus.shape[1:] == (n_samples, 2)

    save_dataset({"g_plus": g_plus, "g_minus": g_minus}, fpath, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
