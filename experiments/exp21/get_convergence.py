#!/usr/bin/env python3
"""Compute convergence metrics (R-hat and ESS) from results of experiment"""

import numpy as np
import typer
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
from tqdm import tqdm

from bpd import DATA_DIR
from bpd.io import load_dataset, save_dataset


def get_ess_all(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One histogram of ESS and R-hat per parameter."""
    print("INFO: Computing convergence plots...")
    n_gals, n_chains_per_gal, n_samples = samples["lf"].shape

    ess_dict = {}
    for p in samples:
        ess = []
        n_samples_total = n_samples * n_chains_per_gal
        for ii in tqdm(range(n_gals)):
            chains = samples[p][ii]
            assert chains.shape == (n_chains_per_gal, n_samples)
            ess.append(effective_sample_size(chains) / n_samples_total)
        ess_dict[p] = np.array(ess)

    return ess_dict


def get_rhat_all(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One histogram of ESS and R-hat per parameter."""
    print("INFO: Computing convergence plots...")
    n_gals, n_chains_per_gal, n_samples = samples["lf"].shape

    rhat_dict = {}
    for p in samples:
        rhat = []
        for ii in tqdm(range(n_gals)):
            chains = samples[p][ii]
            assert chains.shape == (n_chains_per_gal, n_samples)
            rhat.append(potential_scale_reduction(chains))
        rhat_dict[p] = np.array(rhat)

    return rhat_dict


def main(seed: int, tag: str):
    exp_dir = DATA_DIR / "cache_chains" / tag

    fpath = exp_dir / f"chain_results_{seed}.npy"

    results = load_dataset(fpath)
    max_n_gal = max(int(k) for k in results)
    samples = results[max_n_gal]["samples"]

    ess = get_ess_all(samples)
    rhats = get_rhat_all(samples)

    save_dataset(
        {"ess": ess, "rhat": rhats},
        exp_dir / f"convergence_results_{seed}.npz",
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
