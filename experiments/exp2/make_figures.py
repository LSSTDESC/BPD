#!/usr/bin/env python3

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

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
    samples_dict: dict[str, Array], truth: dict[str, Array], n_examples: int = 25
) -> None:
    """Make example figure showing example trace plots for each parameter."""
    # by default, we choose 10 random traces to plot in 1 PDF file.
    fname = "figs/traces.pdf"
    assert samples_dict["lf"].shape == (250, 4, 500)
    n_gals = samples_dict["lf"].shape[0]

    with PdfPages(fname) as pdf:
        for _ in tqdm(range(n_examples), desc="Making traces"):
            idx = np.random.choice(np.arange(0, n_gals)).item()
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
    assert samples_dict["lf"].shape == (250, 4, 500)
    n_gals = samples_dict["lf"].shape[0]

    with PdfPages(fname) as pdf:
        for _ in tqdm(range(n_examples), desc="Making contours"):
            idx = np.random.choice(np.arange(0, n_gals)).item()
            true_params = {p: q[idx].item() for p, q in truth.items()}

            # save one contour per galaxy for now
            samples_list = [{k: v[idx, 0] for k, v in samples_dict.items()}]
            names = ["post0"]
            fig = get_contour_plot(samples_list, names, true_params, figsize=(10, 10))
            pdf.savefig(fig)
            plt.close(fig)


def make_convergence_histograms(samples_dict: dict[str, Array]) -> None:
    """One histogram of ESS and R-hat per parameter."""
    print("INFO: Computing convergence plots...")
    fname = "figs/convergence_hist.pdf"

    # compute convergence metrics
    rhats = {p: [] for p in samples_dict}
    ess = {p: [] for p in samples_dict}
    for ii in tqdm(range(250), desc="Computing convergence metrics"):
        for p in samples_dict:
            chains = samples_dict[p][ii]
            assert chains.shape == (4, 500)
            rhats[p].append(potential_scale_reduction(chains))
            ess[p].append(effective_sample_size(chains) / 2000)

    with PdfPages(fname) as pdf:
        for p in samples_dict:
            rhat_p = rhats[p]
            ess_p = ess[p]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
            fig.suptitle(p, fontsize=18)

            ax1.hist(rhat_p, bins=25, range=(1, 2))
            ax2.hist(ess_p, bins=25)

            ax1.set_xlabel("R-hat")
            ax2.set_ylabel("ESS")

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

    for n_gals in all_n_gals:
        t_warmup = results_dict[n_gals]["t_warmup"]
        t_sampling = results_dict[n_gals]["t_sampling"]

        t_per_obj_warmup = t_warmup / (4 * n_gals)
        t_per_obj_per_sample_sampling = t_sampling / (4 * n_gals * 500)
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
    fpath = (
        DATA_DIR / "cache_chains" / "test_image_sampling_42" / "chain_results_42.npy"
    )
    assert fpath.exists()
    results = jnp.load(fpath, allow_pickle=True).item()
    samples = results[250]["samples"]
    truth = results[250]["truth"]

    assert samples["lf"].shape == (250, 4, 500)
    assert truth["lf"].shape == (250,)

    # make plots
    make_trace_plots(samples, truth)
    make_contour_plots(samples, truth)
    make_convergence_histograms(samples)
    make_timing_plots(results)


if __name__ == "__main__":
    main()
