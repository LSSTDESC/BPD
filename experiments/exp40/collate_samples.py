#!/usr/bin/env python3
import os

import jax.numpy as jnp
import typer

from bpd import DATA_DIR
from bpd.io import load_dataset, save_dataset


def main(seed: int = 42):
    dirpath = DATA_DIR / "cache_chains" / f"exp40_{seed}"
    newpath = dirpath / "interim_samples_42.npz"

    if newpath.exists():
        os.remove(newpath)

    full_ds = {}

    for fp in dirpath.iterdir():
        if "interim_samples_" in fp.name:
            ds = load_dataset(fp)

            for k, v in ds.items():
                if k in ("e_post", "e1_true", "e2_true", "f"):
                    if k in full_ds:
                        full_ds[k] = jnp.concatenate([full_ds[k], v])
                    else:
                        full_ds[k] = ds[k]
                else:
                    full_ds[k] = ds[k]

        save_dataset(full_ds, newpath, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
