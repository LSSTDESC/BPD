#!/usr/bin/env python3
import os

import typer

from bpd import DATA_DIR
from bpd.io import load_dataset, merge_dicts, save_dataset


def main(
    seed: int,
    tag: str = typer.Option(),
    mode: str = typer.Option(),
    start_string: str = "interim_samples",
):
    assert mode in ("plus", "minus", "")
    mode_txt = f"_{mode}" if mode else ""
    dirpath = DATA_DIR / "cache_chains" / tag
    newpath = dirpath / f"{start_string}_{seed}{mode_txt}.npz"

    if newpath.exists():
        os.remove(newpath)

    # first collect all relevant files
    fps = []
    for fp in dirpath.iterdir():
        cond1 = fp.name.startswith(start_string)
        cond2 = fp.name != newpath.name  # technically not necessary
        cond3 = mode in fp.name
        if cond1 and cond2 and cond3:
            fps.append(fp)

    full_ds = {}
    fps = sorted(fps)  # paths have intrinsic ordering just like strings
    for fp in fps:
        ds = load_dataset(fp)
        full_ds = merge_dicts(full_ds, ds, axis=0)

    save_dataset(full_ds, newpath, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
