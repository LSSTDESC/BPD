#!/usr/bin/env python3
import os

import jax.numpy as jnp
import typer

from bpd import DATA_DIR
from bpd.io import load_dataset, save_dataset


def main(
    seed: int, tag: str = typer.Option(), mode: str = typer.Option(), n_files: int = 4
):
    assert mode in ("plus", "minus", "")
    mode_txt = f"_{mode}" if mode else ""
    dirpath = DATA_DIR / "cache_chains" / tag
    newpath = dirpath / f"interim_samples_{seed}{mode_txt}.npz"

    if newpath.exists():
        os.remove(newpath)

    full_ds = {}

    for ii in range(n_files):
        fp = dirpath / f"interim_samples_{seed}{ii}{mode_txt}.npz"
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
