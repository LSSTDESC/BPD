#!/usr/bin/env python3
import os

import jax.numpy as jnp
import typer

from bpd import DATA_DIR
from bpd.io import load_dataset, save_dataset


def main(seed: int, tag: str = typer.Option(), mode: str = typer.Option()):
    assert mode in ("plus", "minus", "")
    mode_txt = f"_{mode}" if mode else ""
    dirpath = DATA_DIR / "cache_chains" / tag
    newpath = dirpath / f"interim_samples_{seed}{mode_txt}.npz"

    if newpath.exists():
        os.remove(newpath)

    full_ds = {}

    # first collect all relevant files
    fps = []
    for fp in dirpath.iterdir():
        cond1 = fp.name.startswith(f"interim_samples_{seed}")
        cond2 = fp.name != newpath.name
        cond3 = mode in fp.name
        if cond1 and cond2 and cond3:
            fps.append(fp)

    fps = sorted(fps)  # paths have intrinsic ordering just like strings
    for fp in fps:
        ds = load_dataset(fp)
        for k1 in full_ds:
            if k1 in ("samples", "truth"):
                for k2 in ds[k1]:
                    if k2 in full_ds[k1]:
                        full_ds[k1][k2] = jnp.concatenate([full_ds[k1][k2], ds[k1][k2]])
                    else:
                        full_ds[k1][k2] = ds[k1][k2]
            else:
                full_ds[k1] = ds[k1]

        save_dataset(full_ds, newpath, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
