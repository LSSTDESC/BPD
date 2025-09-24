#!/usr/bin/env python3

from functools import partial

import jax
import jax.numpy as jnp
import typer
from jax import random

from bpd import DATA_DIR
from bpd.bootstrap import run_bootstrap
from bpd.io import load_dataset_jax, save_dataset
from bpd.pipelines import pipeline_shear_inference_simple


def main(
    seed: int,
    tag: str = typer.Option(),
    samples_plus_fname: str = typer.Option(),
    samples_minus_fname: str = typer.Option(),
    n_gals: int | None = None,
    initial_step_size: float = 0.1,
    n_samples: int = 1000,
    n_boots: int = 125,
    no_bar: bool = False,
):
    dirpath = DATA_DIR / "cache_chains" / tag
    samples_plus_fpath = dirpath / samples_plus_fname
    samples_minus_fpath = dirpath / samples_minus_fname
    assert samples_plus_fpath.exists() and samples_minus_fpath.exists()
    fpath = dirpath / f"g_samples_boots_{seed}.npz"

    dsp = load_dataset_jax(samples_plus_fpath)
    dsm = load_dataset_jax(samples_minus_fpath)

    rng_key = random.key(seed)
    k1, k2 = random.split(rng_key)

    total_n_gals = dsp["samples"]["e1"].shape[0]
    if n_gals is None:
        n_gals = total_n_gals
    assert n_gals <= total_n_gals
    subset = random.choice(k1, jnp.arange(total_n_gals), shape=(n_gals,), replace=False)

    e1p = dsp["samples"]["e1"][subset]
    e2p = dsp["samples"]["e2"][subset]
    e1e2p = jnp.stack([e1p, e2p], axis=-1)

    e1m = dsm["samples"]["e1"][subset]
    e2m = dsm["samples"]["e2"][subset]
    e1e2m = jnp.stack([e1m, e2m], axis=-1)

    sigma_e = dsp["hyper"]["shape_noise"]
    sigma_e_int = dsp["hyper"]["sigma_e_int"]
    assert sigma_e == dsm["hyper"]["shape_noise"]
    assert sigma_e_int == dsm["hyper"]["sigma_e_int"]
    assert jnp.all(dsp["truth"]["e1"] == dsm["truth"]["e1"])
    assert jnp.all(dsp["truth"]["lf"] == dsm["truth"]["lf"])

    raw_pipeline = partial(
        pipeline_shear_inference_simple,
        init_g=jnp.array([0.0, 0.0]),
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples,
        initial_step_size=initial_step_size,
    )

    @jax.jit
    def pipe(k, d):
        out = raw_pipeline(k, d["e1e2"])
        return {"g1": out[..., 0], "g2": out[..., 1]}

    # jit
    print("JITing function")
    _ = pipe(k2, {"e1e2": e1e2p[:2]})
    print("JITing done")

    samples_plus = run_bootstrap(
        k2,
        post_params={"e1e2": e1e2p},
        n_gals=e1e2p.shape[0],
        n_boots=n_boots,
        no_bar=no_bar,
    )
    samples_minus = run_bootstrap(
        k2,
        post_params={"e1e2": e1e2m},
        n_gals=e1e2m.shape[0],
        n_boots=n_boots,
        no_bar=no_bar,
    )

    assert samples_plus["g1"].shape == (n_boots, n_samples)
    assert samples_minus["g1"].shape == (n_boots, n_samples)

    save_dataset(
        {
            "plus": {"g1": samples_plus["g1"], "g2": samples_plus["g2"]},
            "minus": {"g1": samples_minus["g1"], "g2": samples_minus["g2"]},
        },
        fpath,
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
