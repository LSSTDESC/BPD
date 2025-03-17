#!/usr/bin/env python3
"""Compute convergence metrics (R-hat and ESS) from results of experiment"""

import numpy as np
import typer
from arviz import ess
from blackjax.diagnostics import potential_scale_reduction
from tqdm import tqdm

from bpd import DATA_DIR
from bpd.io import load_dataset, save_dataset


def get_ess_all(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Obtain ESS for all chains and parameters."""
    print("INFO: Computing ESS...")
    n_gals, n_chains_per_gal, n_samples_per_gal = samples["lf"].shape

    n_samples = n_chains_per_gal * n_samples_per_gal
    ess_dict = {}

    for ii in tqdm(range(n_gals)):
        params = {k: v[ii] for k, v in samples.items()}
        value = ess(params)

        for k in samples:
            kval = value[k].data.item() / n_samples
            if k not in ess_dict:
                ess_dict[k] = [kval]
            else:
                ess_dict[k].append(kval)
    ess_dict = {k: np.array(v) for k, v in ess_dict.items()}
    return ess_dict


def get_rhat_all(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Obtain r-hats for all chains and parameters."""
    print("INFO: Computing R-hats...")
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

    fpath = exp_dir / f"full_samples_{seed}.npz"

    results = load_dataset(fpath)
    samples = results["samples"]

    ess_array = get_ess_all(samples)
    rhats = get_rhat_all(samples)

    save_dataset(
        {"ess": ess_array, "rhat": rhats},
        exp_dir / f"convergence_results_{seed}.npz",
        overwrite=True,
    )


if __name__ == "__main__":
    typer.run(main)
