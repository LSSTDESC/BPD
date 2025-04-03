#!/usr/bin/env python3

from functools import partial

import jax
import jax.numpy as jnp
import typer
from jax import jit

from bpd import DATA_DIR
from bpd.io import load_dataset_jax, save_dataset
from bpd.jackknife import run_jackknife_shear_pipeline
from bpd.pipelines import pipeline_shear_inference_simple


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_fname: str = typer.Option(),
    initial_step_size: float = 0.01,
    n_samples: int = 1000,
    n_jacks: int = 100,
    no_bar: bool = False,
    start: int = 0,
    end: int | None = None,
):
    if end is None:
        end = n_jacks

    # load data
    dirpath = DATA_DIR / "cache_chains" / tag
    samples_fpath = dirpath / samples_fname
    assert samples_fpath.exists()

    if start == 0 and end == n_jacks:
        fpath = dirpath / f"g_samples_jack_{seed}.npz"
    else:
        fpath = dirpath / f"g_samples_jack_{seed}_{start}.npz"

    ds = load_dataset_jax(samples_fpath)
    e1e2p = ds["e1e2p"]
    e1e2m = ds["e1e2m"]
    sigma_e = ds["sigma_e"]
    sigma_e_int = ds["sigma_e_int"]

    rng_key = jax.random.key(seed)
    raw_pipeline = partial(
        pipeline_shear_inference_simple,
        init_g=jnp.array([0.0, 0.0]),
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )
    raw_pipeline_jitted = jit(raw_pipeline)
    pipe = lambda k, d: raw_pipeline_jitted(k, d["e1e2"])

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
