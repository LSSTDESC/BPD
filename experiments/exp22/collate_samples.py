#!/usr/bin/env python3
import os

import typer

from bpd import DATA_DIR
from bpd.io import load_dataset, merge_dicts, save_dataset


def main(seed: int, tag: str = typer.Option()):
    dirpath = DATA_DIR / "cache_chains" / tag
    newpath = dirpath / f"full_samples_{seed}.npz"

    if newpath.exists():
        os.remove(newpath)

    full_ds = {}

    # first collect all relevant files
    fps = []
    for fp in dirpath.iterdir():
        cond1 = fp.name.startswith(f"full_samples_{seed}")
        cond2 = fp.name != newpath.name
        if cond1 and cond2:
            fps.append(fp)

    fps = sorted(fps)  # paths have intrinsic ordering just like strings
    for fp in fps:
        ds = load_dataset(fp)
        full_ds = merge_dicts(full_ds, ds, axis=0)

    save_dataset(full_ds, newpath, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
