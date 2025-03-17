#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from bpd import DATA_DIR
from bpd.diagnostics import get_contour_plot
from bpd.io import load_dataset
from bpd.plotting import get_timing_figure


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
    fpath: str,
    n_examples: int = 10,
) -> None:
    """Make example figure showing example contour plots of galaxy properties"""
    n_gals, _, _ = samples["lf"].shape
    with PdfPages(fpath) as pdf:
        for _ in tqdm(range(n_examples), desc="Making contours"):
            idx = np.random.choice(np.arange(0, n_gals)).item()
            true_params = {p: q[idx].item() for p, q in truth.items()}

            # save one contour per galaxy for now
            samples_list = [{k: v[idx, 0] for k, v in samples.items()}]
            names = ["post0"]
            fig = get_contour_plot(samples_list, names, true_params, figsize=(12, 12))
            pdf.savefig(fig)
            plt.close(fig)


def make_adaptation_hists(tuned_params: dict, pnames: dict, fpath: Path):
    step_sizes = tuned_params["step_size"]
    imm = tuned_params["inverse_mass_matrix"]

    with PdfPages(fpath) as pdf:
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


def make_convergence_histograms(
    conv_results: dict, fpath: Path, outliers_fpath: Path
) -> set:
    """One histogram of ESS and R-hat per parameter."""
    print("INFO: Computing convergence plots...")
    ess_dict = conv_results["ess"]
    rhats_dict = conv_results["rhat"]

    # print rhat outliers
    outliers_indices = set()
    for p in rhats_dict:
        rhat = rhats_dict[p]
        _ess = ess_dict[p]
        outliers = (
            (rhat < 0.99)
            | (rhat > 1.05)
            | (_ess < 0.1)
            | np.isnan(_ess)
            | np.isnan(rhat)
        )
        n_outliers = sum(outliers)
        with open(outliers_fpath, "a", encoding="utf-8") as f:
            print(f"Number of R-hat outliers for {p}: {n_outliers}", file=f)
        indices = np.argwhere(outliers)
        outliers_indices = outliers_indices.union(set(indices.ravel()))

    with PdfPages(fpath) as pdf:
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


def make_timing_plots(results, max_n_gal, fpath: str | Path):
    fig = get_timing_figure(results, max_n_gal)

    with PdfPages(fpath) as pdf:
        pdf.savefig(fig)
        plt.close(fig)


def main(seed: int, tag: str):
    np.random.seed(seed)
    figdir = Path("figs") / str(seed)
    figdir.mkdir(exist_ok=True)

    wdir = DATA_DIR / "cache_chains" / tag
    full_fpath = wdir / f"full_samples_{seed}.npz"
    timing_fpath = wdir / f"timing_results{seed}.npz"

    conv_fpath = wdir / f"convergence_results_{seed}.npz"

    assert full_fpath.exists()
    assert conv_fpath.exists()
    assert timing_fpath.exists()

    full_results = load_dataset(full_fpath)
    conv_results = load_dataset(conv_fpath)
    timing_results = load_dataset(timing_fpath)

    samples = full_results["samples"]
    truth = full_results["truth"]
    max_n_gal = max(int(k) for k in timing_results)
    tuned_params = timing_results[max_n_gal]["tuned_params"]
    param_names = list(samples.keys())

    # make plots of full samples
    make_trace_plots(samples, truth, fpath=figdir / "traces.pdf")
    make_contour_plots(samples, truth, fpath=figdir / "contours.pdf")
    make_timing_plots(timing_results, max_n_gal, fpath=figdir / "timing.pdf")
    make_adaptation_hists(tuned_params, param_names, fpath=figdir / "adaptation.pdf")

    # on adaption too
    adapt_states = timing_results[max_n_gal]["adapt_position"]
    make_trace_plots(adapt_states, truth, fpath=figdir / "traces_adapt.pdf")

    outliers_fpath = figdir / "outliers.txt"
    if Path(outliers_fpath).exists():
        os.remove(outliers_fpath)

    out_indices = make_convergence_histograms(
        conv_results, fpath=figdir / "convergence.pdf", outliers_fpath=outliers_fpath
    )

    # outliers
    traces_out_fpath = figdir / "traces_out.pdf"
    if len(out_indices) > 0:
        make_trace_at_indices(out_indices, samples, truth, fpath=traces_out_fpath)
    else:  # avoid confusion with previous
        if traces_out_fpath.exists():
            os.remove(traces_out_fpath)


if __name__ == "__main__":
    typer.run(main)
