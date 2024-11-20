#!/usr/bin/env python3

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
from jax import Array
from matplotlib.backends.backend_pdf import PdfPages

from bpd import DATA_DIR
from bpd.diagnostics import get_contour_plot, get_gauss_pc_fig, get_pc_fig


def make_trace_plots(
    samples_dict: dict[str, Array], truth: dict[str, Array], n_examples: int = 25
) -> None:
    """Make example figure showing example trace plots for each parameter."""
    # by default, we choose 10 random traces to plot in 1 PDF file.
    fname = "figs/traces.pdf"
    assert samples_dict["lf"].shape == (250, 4, 500)
    n_gals = samples_dict["lf"].shape[0]

    with PdfPages(fname) as pdf:
        for _ in range(n_examples):
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
        for _ in range(n_examples):
            idx = np.random.choice(np.arange(0, n_gals)).item()
            true_params = {p: q[idx].item() for p, q in truth.items()}

            # save one contour per galaxy for now
            samples_list = [{k: v[idx, 0] for k, v in samples_dict.items()}]
            names = ["post0"]
            fig = get_contour_plot(samples_list, names, true_params, figsize=(8, 8))
            pdf.savefig(fig)
            plt.close(fig)


def make_convergence_histograms(samples_dict: dict[str, Array]) -> None:
    """One histogram of ESS and R-hat per parameter."""
    fname = "figs/convergence_hist.pdf"

    with PdfPages(fname) as pdf:
        for p in samples_dict:
            chains = samples_dict[p]
            assert chains.shape == (250, 4, 500)
            rhats = [potential_scale_reduction(chains[ii]) for ii in range(250)]
            essr = [effective_sample_size(chains[jj]) / 2000 for jj in range(250)]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
            fig.suptitle(p, fontsize=18)

            ax1.hist(rhats, bins=25)
            ax2.hist(essr, bins=25)

            pdf.savefig(fig)
            plt.close(fig)


# def make_timing_plots(results_dict: dict) -> None:
#     fname = "figs/multiplicative_bias_hist.pdf"
#     with PdfPages(fname) as pdf:
#         g1 = g_samples[:, :, 0]
#         mbias = (g1.mean(axis=1) - 0.02) / 0.02
#         fig, ax = plt.subplots(1, 1, figsize=(7, 7))
#         ax.hist(mbias, bins=31, histtype="step")

#         pdf.savefig(fig)
#         plt.close(fig)


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


if __name__ == "__main__":
    main()
