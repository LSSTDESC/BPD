from typing import Dict, List

import h5py
import numpy as np


def save_samples(samples: Dict[str, np.ndarray], filename: str, group: str = None):
    """Save samples to file using hdf5 format."""
    with h5py.File(filename, "w") as f:
        for k, v in samples.items():
            if group is not None:
                k = f"{group}/{k}"
            f.create_dataset(k, data=v)


def load_samples(filename: str, groups: List[str] = None) -> Dict[str, np.ndarray]:
    """Load samples from file using hdf5 format."""
    samples = {}
    with h5py.File(filename, "r") as f:
        if groups is not None:
            for g in groups:
                assert g in f
                samples[g] = {k: np.asarray(v) for k, v in f[g].items()}
        else:
            samples = {k: np.asarray(v) for k, v in f.items()}
    return samples
