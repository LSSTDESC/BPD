"""Methods to save datasets using h5py."""

# useful resource: https://gist.github.com/gilliss/7e1d451b42441e77ae33a8b790eeeb73

from pathlib import Path

import jax.numpy as jnp
from jax import Array


def save_dataset(
    ds: dict[str, Array], fpath: str | Path, overwrite: bool = False
) -> None:
    if Path(fpath).exists() and not overwrite:
        raise IOError("overwriting existing ds")
    assert Path(fpath).suffix == ".npz"

    jnp.savez(fpath, **ds)


def load_dataset(fpath: str) -> dict[str, Array]:
    assert Path(fpath).exists(), "file path does not exists"
    assert Path(fpath).suffix == ".npz"
    ds = {}

    npzfile = jnp.load(fpath)
    for k in npzfile.files:
        ds[k] = npzfile[k]
    return ds
