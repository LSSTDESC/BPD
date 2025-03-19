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
    samples_plus_fname: str = typer.Option(),
    samples_minus_fname: str = typer.Option(),
    initial_step_size: float = 1e-3,
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

    samples_plus_ds = load_dataset_jax(samples_plus_fpath)
    samples_minus_ds = load_dataset_jax(samples_minus_fpath)

    e_post_plus = samples_plus_ds["e_post"]
    e_post_minus = samples_minus_ds["e_post"]

    true_g = samples_plus_ds["true_g"]
    sigma_e = samples_plus_ds["sigma_e"]
    sigma_e_int = samples_plus_ds["sigma_e_int"]
    assert jnp.all(true_g == -samples_minus_ds["true_g"])
    assert sigma_e == samples_minus_ds["sigma_e"]
    assert sigma_e_int == samples_minus_ds["sigma_e_int"]
    assert jnp.all(samples_plus_ds["e1_true"] == samples_minus_ds["e1_true"])
    assert jnp.all(samples_plus_ds["f"] == samples_minus_ds["f"])

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
    pipeline = lambda k, d: raw_pipeline_jitted(k, d["e1e2"])

    g_plus, g_minus = run_jackknife_shear_pipeline(
        rng_key,
        post_params_plus={"e1e2": e_post_plus},
        post_params_minus={"e1e2": e_post_minus},
        shear_pipeline=pipeline,
        n_gals=e_post_plus.shape[0],
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
