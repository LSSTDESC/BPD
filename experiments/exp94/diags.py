#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from arviz import ess
from blackjax.diagnostics import potential_scale_reduction
from tqdm import tqdm

from bpd import DATA_DIR
from bpd.io import load_dataset


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


def main():
    dirpath = DATA_DIR / "cache_chains" / "exp94"
    expath = Path(__file__).parent

    samples_dict = load_dataset(dirpath / "samples.npz")

    # discard first 100 samples for burn in
    samples_dict_final = {k: v[:, :, 100:] for k, v in samples_dict.items()}
    ess_dict = get_ess_all(samples_dict_final)
    rhat_dict = get_rhat_all(samples_dict_final)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for p in ess_dict:
        ax.hist(ess_dict[p], bins=15, histtype="step", label=p)
    ax.legend()
    ax.set_xlabel("Relative ESS")
    fig.savefig(expath / "ess.png", format="png")

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for p in rhat_dict:
        ax.hist(rhat_dict[p], bins=15, histtype="step", label=p)
    ax.legend()
    ax.set_xlabel("Rhat")
    fig.savefig(expath / "rhat.png", format="png")

    # timing
    timing_results = load_dataset(dirpath / "timing.npz")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    n_gals_arr = np.array([int(x) for x in list(timing_results.keys())])
    times = np.array([float(x) for x in list(timing_results.values())])

    ax1.plot(n_gals_arr, times, "-o")
    ax1.set_xlabel("# of Galaxies")
    ax1.set_ylabel("Total time (sec)")
    ax1.set_yscale("log")
    ax1.set_xscale("log")

    # time per sample
    times_per_sample = times / (n_gals_arr * 600 * 12)

    # timing (log)
    ax2.plot(n_gals_arr, times_per_sample, "-o")
    ax2.set_xlabel("# of Galaxies")
    ax2.set_ylabel("Avg. Time per sample (sec)")
    ax2.set_yscale("log")
    ax2.set_xscale("log")

    plt.tight_layout()

    fig.savefig(expath / "timing_log.png", format="png")

    # timing (not log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

    ax1.plot(n_gals_arr, times, "-o")
    ax1.set_xlabel("# of Galaxies")
    ax1.set_ylabel("Total time (sec)")

    ax2.plot(n_gals_arr, times_per_sample, "-o")
    ax2.set_xlabel("# of Galaxies")
    ax2.set_ylabel("Avg. Time per sample (sec)")

    plt.tight_layout()

    fig.savefig(expath / "timing.png", format="png")


if __name__ == "__main__":
    typer.run(main)
