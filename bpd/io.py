"""Methods to save datasets using h5py."""

# useful resource: https://gist.github.com/gilliss/7e1d451b42441e77ae33a8b790eeeb73

from pathlib import Path

import h5py
import jax.numpy as jnp
from jax import Array


def save_dataset_h5py(
    ds: dict[str, Array], fpath: str, overwrite: bool = False
) -> None:
    if Path(fpath).exists() and not overwrite:
        raise IOError("overwriting existing ds")

    with h5py.File(fpath, "w") as f:
        for k, v in ds.items():
            f.create_dataset(k, data=v)


def load_dataset_h5py(fpath: str) -> dict[str, Array]:
    assert Path(fpath).exists(), "file path does not exists"
    ds = {}
    with h5py.File(fpath, "r") as f:
        for k, v in f.items():
            ds[k] = jnp.array(v[...])
    return ds
