#!/usr/bin/env python3

import os
from typing import Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from pathlib import Path

import cycler
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
from jax import Array
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from bpd import DATA_DIR
from bpd.diagnostics import get_contour_plot


def make_trace_plots(
    samples_dict: dict[str, Array],
    truth: dict[str, Array],
    fpath: str,
    n_examples: int = 50,
) -> None:
    """Make example figure showing example trace plots for each parameter."""
    n_gals, _, _ = samples_dict["lf"].shape
    indices = np.random.choice(np.arange(0, n_gals), size=(n_examples,), replace=False)
    make_trace_at_indices(indices, samples_dict, truth, fpath)


def make_trace_at_indices(
    indices: Iterable,
    samples_dict: dict[str, Array],
    truth: dict[str, Array],
    fpath: str,
):
    with PdfPages(fpath) as pdf:
        for idx in tqdm(indices, desc="Making traces"):
            chains = {k: v[idx] for k, v in samples_dict.items()}
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
    samples_dict: dict[str, Array], truth: dict[str, Array], n_examples: int = 10
) -> None:
    """Make example figure showing example contour plots of galaxy properties"""
    fname = "figs/contours.pdf"
    n_gals, _, _ = samples_dict["lf"].shape
    with PdfPages(fname) as pdf:
        for _ in tqdm(range(n_examples), desc="Making contours"):
            idx = np.random.choice(np.arange(0, n_gals)).item()
            true_params = {p: q[idx].item() for p, q in truth.items()}

            # save one contour per galaxy for now
            samples_list = [{k: v[idx, 0] for k, v in samples_dict.items()}]
            names = ["post0"]
            fig = get_contour_plot(samples_list, names, true_params, figsize=(12, 12))
            pdf.savefig(fig)
            plt.close(fig)


def make_convergence_histograms(samples_dict: dict[str, Array]) -> None:
    """One histogram of ESS and R-hat per parameter."""
    print("INFO: Computing convergence plots...")
    fname = "figs/convergence_hist.pdf"
    n_gals, n_chains_per_gal, n_samples = samples_dict["lf"].shape

    if Path("figs/outliers.txt").exists():
        os.remove("figs/outliers.txt")

    # compute convergence metrics
    rhats = {p: [] for p in samples_dict}
    ess = {p: [] for p in samples_dict}
    for ii in tqdm(range(n_gals), desc="Computing convergence metrics"):
        for p in samples_dict:
            chains = samples_dict[p][ii]
            n_samples_total = n_samples * n_chains_per_gal
            assert chains.shape == (n_chains_per_gal, n_samples)
            rhats[p].append(potential_scale_reduction(chains))
            ess[p].append(effective_sample_size(chains) / n_samples_total)

    # print rhat outliers
    outliers_indices = set()
    for p in rhats:
        rhat = np.array(rhats[p])
        _ess = np.array(ess[p])
        outliers = (rhat < 0.99) | (rhat > 1.05) | (_ess < 0.25)
        n_outliers = sum(outliers)
        with open("figs/outliers.txt", "a", encoding="utf-8") as f:
            print(f"Number of R-hat outliers for {p}: {n_outliers}", file=f)
        indices = np.argwhere(outliers)
        outliers_indices = outliers_indices.union(set(indices.ravel()))

    with PdfPages(fname) as pdf:
        for p in samples_dict:
            rhat_p = rhats[p]
            ess_p = ess[p]

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


def make_timing_plots(results_dict: dict) -> None:
    print("INFO: Making timing plots...")
    fname = "figs/timing.pdf"
    all_n_gals = [int(n_gals) for n_gals in results_dict]

    # cycler from blue to red
    color = plt.cm.coolwarm(np.linspace(0, 1, len(all_n_gals)))
    cycles = cycler.cycler("color", color)

    t_per_obj_dict = {}
    n_samples_array = jnp.arange(0, 1001, 1)

    _, n_chains_per_gal, n_samples = results_dict[1]["samples"]["lf"].shape

    for n_gals in all_n_gals:
        t_warmup = results_dict[n_gals]["t_warmup"]
        t_sampling = results_dict[n_gals]["t_sampling"]

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


def main():
    np.random.seed(42)

    fpath = DATA_DIR / "cache_chains" / "exp21_42" / "chain_results_42.npy"
    assert fpath.exists()
    results = jnp.load(fpath, allow_pickle=True).item()
    max_n_gal = max(results.keys())
    samples = results[max_n_gal]["samples"]
    truth = results[max_n_gal]["truth"]

    if "x" in truth:
        truth["dx"] = jnp.zeros_like(truth.pop("x"))
        truth["dy"] = jnp.zeros_like(truth.pop("y"))

    tuned_params = results[max_n_gal]["tuned_params"]

    param_names = list(samples.keys())

    # make plots
    make_trace_plots(samples, truth, fpath="figs/traces.pdf")
    make_contour_plots(samples, truth)
    out_indices = make_convergence_histograms(samples)
    make_timing_plots(results)

    # on adaption too
    adapt_states = results[max_n_gal]["adapt_info"].state.position
    make_trace_plots(adapt_states, truth, fpath="figs/traces_adapt.pdf")

    # outliers
    if len(out_indices) > 0:
        make_trace_at_indices(out_indices, samples, truth, fpath="figs/traces_out.pdf")
    else:  # avoid confusion with previous
        if Path("figs/traces_out.pdf").exists():
            os.remove("figs/traces_out.pdf")

    make_adaptation_hists(tuned_params, param_names)


if __name__ == "__main__":
    main()
