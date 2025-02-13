#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Iterable

import cycler
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from bpd import DATA_DIR
from bpd.diagnostics import get_contour_plot
from bpd.io import load_dataset


def make_trace_plots(
    samples: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    fpath: str,
    n_examples: int = 50,
) -> None:
    """Make example figure showing example trace plots for each parameter."""
    n_gals, _, _ = samples["lf"].shape
    indices = np.random.choice(np.arange(0, n_gals), size=(n_examples,), replace=False)
    make_trace_at_indices(indices, samples, truth, fpath)


def make_trace_at_indices(
    indices: Iterable,
    samples: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    fpath: str,
):
    with PdfPages(fpath) as pdf:
        for idx in tqdm(indices, desc="Making traces"):
            chains = {k: v[idx] for k, v in samples.items()}
            tv = {p: q[idx].item() for p, q in truth.items()}

            fig, axes = plt.subplots(6, 1, figsize=(18, 7))
            fig.suptitle(f"Trace plots, index: {idx}")

            for ii, p in enumerate(chains):
                ax = axes.ravel()[ii]
                ax.set_ylabel(p, fontsize=18)
                ax.axhline(tv[p], color="k", linestyle="--")
                for jj in range(4):
                    ax.plot(chains[p][jj])

            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)


def make_contour_plots(
    samples: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    n_examples: int = 10,
) -> None:
    """Make example figure showing example contour plots of galaxy properties"""
    fname = "figs/contours.pdf"
    n_gals, _, _ = samples["lf"].shape
    with PdfPages(fname) as pdf:
        for _ in tqdm(range(n_examples), desc="Making contours"):
            idx = np.random.choice(np.arange(0, n_gals)).item()
            true_params = {p: q[idx].item() for p, q in truth.items()}

            # save one contour per galaxy for now
            samples_list = [{k: v[idx, 0] for k, v in samples.items()}]
            names = ["post0"]
            fig = get_contour_plot(samples_list, names, true_params, figsize=(12, 12))
            pdf.savefig(fig)
            plt.close(fig)


def make_timing_plots(results: dict, max_n_gal: str) -> None:
    print("INFO: Making timing plots...")
    fname = "figs/timing.pdf"
    all_n_gals = [n_gals for n_gals in results]

    # cycler from blue to red
    color = plt.cm.coolwarm(np.linspace(0, 1, len(all_n_gals)))
    cycles = cycler.cycler("color", color)

    t_per_obj_dict = {}
    n_samples_array = np.arange(0, 1001, 1)

    _, n_chains_per_gal, n_samples = results[max_n_gal]["samples"]["lf"].shape

    for n_gals_str in all_n_gals:
        t_warmup = results[n_gals_str]["t_warmup"]
        t_sampling = results[n_gals_str]["t_sampling"]

        n_gals = int(n_gals_str)

        n_chains = n_gals * n_chains_per_gal

        t_per_obj_warmup = t_warmup / n_chains
        t_per_obj_per_sample_sampling = t_sampling / (n_chains * n_samples)
        t_per_obj_arr = (
            t_per_obj_warmup + t_per_obj_per_sample_sampling * n_samples_array
        )
        t_per_obj_dict[n_gals] = t_per_obj_arr

    with PdfPages(fname) as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.set_prop_cycle(cycles)

        ax.set_ylabel("Time per obj per GPU core (sec)", fontsize=14)
        ax.set_xlabel("# samples", fontsize=14)

        for n_gals, t_per_obj_array in t_per_obj_dict.items():
            n_chains = 4 * n_gals
            ax.plot(n_samples_array, t_per_obj_array, label=f"n_chains:{n_chains}")

        ax.legend()

        pdf.savefig(fig)
        plt.close(fig)


def make_adaptation_hists(tuned_params: dict, pnames: dict):
    fname = "figs/tuned_hists.pdf"

    step_sizes = tuned_params["step_size"]
    imm = tuned_params["inverse_mass_matrix"]

    with PdfPages(fname) as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.hist(step_sizes.flatten(), bins=25)
        ax.axvline(step_sizes.flatten().mean(), linestyle="--", color="k", label="mean")
        ax.set_xlabel("Step sizes")
        ax.legend()

        pdf.savefig(fig)
        plt.close(fig)

        for ii, p in enumerate(pnames):
            diag_elems = imm[:, :, ii].flatten()
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.hist(diag_elems, bins=25)
            ax.axvline(diag_elems.mean(), linestyle="--", color="k", label="mean")
            ax.set_xlabel(f"Diag Mass Matrix for {p}")
            ax.legend()

            pdf.savefig(fig)
            plt.close(fig)


def make_convergence_histograms(conv_results: dict) -> set:
    """One histogram of ESS and R-hat per parameter."""
    print("INFO: Computing convergence plots...")
    fname = "figs/convergence_hist.pdf"
    ess_dict = conv_results["ess"]
    rhats_dict = conv_results["rhat"]

    # print rhat outliers
    outliers_indices = set()
    for p in rhats_dict:
        rhat = rhats_dict[p]
        _ess = ess_dict[p]
        outliers = (rhat < 0.99) | (rhat > 1.05) | (_ess < 0.25)
        n_outliers = sum(outliers)
        with open("figs/outliers.txt", "a", encoding="utf-8") as f:
            print(f"Number of R-hat outliers for {p}: {n_outliers}", file=f)
        indices = np.argwhere(outliers)
        outliers_indices = outliers_indices.union(set(indices.ravel()))

    with PdfPages(fname) as pdf:
        for p in rhats_dict:
            rhat_p = rhats_dict[p]
            ess_p = ess_dict[p]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
            fig.suptitle(p, fontsize=18)

            ax1.hist(rhat_p, bins=25, range=(0.98, 1.1))
            ax2.hist(ess_p, bins=25)
            ax2.axvline(ess_p.mean(), linestyle="--", color="k", label="mean")

            ax1.set_xlabel("R-hat")
            ax2.set_ylabel("ESS")
            ax2.legend()

            pdf.savefig(fig)
            plt.close(fig)

    return outliers_indices


def main(seed: int, tag: str):
    np.random.seed(seed)

    wdir = DATA_DIR / "cache_chains" / tag
    results_fpath = wdir / f"chain_results_{seed}.npz"
    conv_fpath = wdir / f"convergence_results_{seed}.npz"

    assert results_fpath.exists() and conv_fpath.exists()
    results = load_dataset(results_fpath)
    conv_results = load_dataset(conv_fpath)
    max_n_gal = str(max(int(k) for k in results))
    samples = results[max_n_gal]["samples"]
    truth = results[max_n_gal]["truth"]
    tuned_params = results[max_n_gal]["tuned_params"]
    param_names = list(samples.keys())

    # make plots
    make_trace_plots(samples, truth, fpath="figs/traces.pdf")
    make_contour_plots(samples, truth)
    make_timing_plots(results, max_n_gal)
    make_adaptation_hists(tuned_params, param_names)

    # on adaption too
    adapt_states = results[max_n_gal]["adapt_position"]
    make_trace_plots(adapt_states, truth, fpath="figs/traces_adapt.pdf")

    if Path("figs/outliers.txt").exists():
        os.remove("figs/outliers.txt")

    out_indices = make_convergence_histograms(conv_results)

    # outliers
    if len(out_indices) > 0:
        make_trace_at_indices(out_indices, samples, truth, fpath="figs/traces_out.pdf")
    else:  # avoid confusion with previous
        if Path("figs/traces_out.pdf").exists():
            os.remove("figs/traces_out.pdf")


if __name__ == "__main__":
    typer.run(main)
