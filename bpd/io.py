"""Methods to save datasets using h5py."""

# useful resource: https://gist.github.com/gilliss/7e1d451b42441e77ae33a8b790eeeb73

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy import ndarray


def flatten_dict(ds: dict | Array, level: list):
    tmp_dict = {}
    for key, val in ds.items():
        assert isinstance(key, str), "all keys need to be str"
        if isinstance(val, dict):
            tmp_dict.update(flatten_dict(val, [*level, key]))
        else:
            tmp_dict[".".join([*level, key])] = val
    return tmp_dict


def unflatten_dict(ds: dict):
    resultDict = dict()
    for key, value in ds.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def merge_dicts(d1: dict, d2: dict, axis=0):
    d = {}
    joint_keys = set(d1.keys()).intersection(set(d2.keys()))
    only_d1_keys = set(d1.keys()).difference(set(d2.keys()))
    only_d2_keys = set(d2.keys()).difference(set(d1.keys()))

    for k in joint_keys:
        v1 = d1[k]
        v2 = d2[k]
        if isinstance(v1, Array) and isinstance(v2, Array):
            v = jnp.concatenate([v1, v2], axis=axis)
            d[k] = v
        elif (
            isinstance(v1, ndarray)
            and isinstance(v2, ndarray)
            and v1.ndim > 0
            and v2.ndim > 0
        ):
            v = np.concatenate([v1, v2], axis=axis)
            d[k] = v
        elif isinstance(v1, dict) and isinstance(v2, dict):
            d[k] = merge_dicts(v1, v2, axis=axis)
        elif v1 == v2 and isinstance(v1, ArrayLike) and isinstance(v2, ArrayLike):
            d[k] = v1
        else:
            raise ValueError("Dicts are not compatible for merging")
    for k in only_d1_keys:
        d[k] = d1[k]
    for k in only_d2_keys:
        d[k] = d2[k]

    return d


def convert_dict_to_numpy(ds: dict):
    """Convert flat dict of jax arrays to numpy arrays."""
    return {k: np.asarray(v) for k, v in ds.items()}


def save_dataset(ds: dict, fpath: str | Path, overwrite: bool = False) -> None:
    """Save dataset consisting of dicts of jax arrays (nested) to a flat numpy .npz dict"""
    if Path(fpath).exists() and not overwrite:
        raise IOError("overwriting existing ds")
    assert Path(fpath).suffix == ".npz"
    flat_ds = flatten_dict(ds, level=[])
    flat_ds_np = convert_dict_to_numpy(flat_ds)
    np.savez(fpath, **flat_ds_np)


def load_dataset(fpath: str) -> dict:
    assert Path(fpath).exists(), "file path does not exists"
    assert Path(fpath).suffix == ".npz"

    ds = {}
    npzfile = np.load(fpath)
    for k in npzfile.files:
        ds[k] = npzfile[k]
    return unflatten_dict(ds)


def load_dataset_jax(fpath: str) -> dict:
    assert Path(fpath).exists(), "file path does not exists"
    assert Path(fpath).suffix == ".npz"

    ds = {}
    npzfile = np.load(fpath)
    for k in npzfile.files:
        ds[k] = jnp.array(npzfile[k])
    return unflatten_dict(ds)
