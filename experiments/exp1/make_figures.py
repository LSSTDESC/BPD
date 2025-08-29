#!/usr/bin/env python3

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"


import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from jax import Array
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from bpd import DATA_DIR
from bpd.diagnostics import get_contour_plot, get_gauss_pc_fig, get_pc_fig


def make_trace_plots(g_samples: Array, n_examples: int = 25) -> None:
    """Make example figure showing example trace plots of shear posteriors."""
    # by default, we choose 10 random traces to plot in 1 PDF file.
    fname = "figs/traces.pdf"
    with PdfPages(fname) as pdf:
        assert g_samples.ndim == 3
        n_post = g_samples.shape[0]
        indices = np.random.choice(np.arange(n_post), (n_examples,))

        for ii in tqdm(indices, desc="Saving traces"):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
            g1 = g_samples[ii, :, 0]
            g2 = g_samples[ii, :, 0]

            ax1.plot(g1)
            ax2.plot(g2)
            ax1.set_ylabel("g1")
            ax2.set_ylabel("g2")

            ax1.set_title(f"Index: {ii}")

            pdf.savefig(fig)
            plt.close(fig)


def make_contour_plots(g_samples: Array, n_examples: int = 25) -> None:
    """Make example figure showing example contour plots of shear posterios"""
    # by default, we choose 10 random contours to plot in 1 PDF file.
    fname = "figs/contours.pdf"
    with PdfPages(fname) as pdf:
        assert g_samples.ndim == 3
        n_post = g_samples.shape[0]
        indices = np.random.choice(np.arange(n_post), (n_examples,))

        truth = {"g1": 0.02, "g2": 0.0}

        for ii in tqdm(indices, desc="Saving contours"):
            g_dict = {"g1": g_samples[ii, :, 0], "g2": g_samples[ii, :, 1]}
            fig = get_contour_plot([g_dict], [f"post_{ii}"], truth)
            plt.suptitle(f"Index: {ii}")
            pdf.savefig(fig)
            plt.close(fig)


def make_posterior_calibration(g_samples: Array) -> None:
    """Output posterior calibration figure."""
    # make two types, assuming gaussianity and one not assuming gaussianity.
    fname = "figs/calibration.pdf"
    with PdfPages(fname) as pdf:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        get_gauss_pc_fig(ax1, g_samples[..., 0], truth=0.02, param_name="g1 (gauss)")
        get_pc_fig(ax2, g_samples[..., 0], truth=0.02, param_name="g1 (full)")
        pdf.savefig(fig)
        plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        get_gauss_pc_fig(ax1, g_samples[..., 1], truth=0.0, param_name="g2 (gauss)")
        get_pc_fig(ax2, g_samples[..., 1], truth=0.0, param_name="g2 (full)")
        pdf.savefig(fig)
        plt.close(fig)


def make_histogram_mbias(g_samples: Array) -> None:
    fname = "figs/multiplicative_bias_hist.pdf"
    with PdfPages(fname) as pdf:
        g1 = g_samples[:, :, 0]

        mbias = (g1.mean(axis=1) - 0.02) / 0.02
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.hist(mbias, bins=25, histtype="step")
        ax.axvline(mbias.mean(), linestyle="--", color="k", label="mean")
        ax.set_xlabel("Multiplicative bias on g1")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)


def make_histogram_means_and_stds(g_samples: Array) -> None:
    fname = "figs/mean_std_hist.pdf"
    with PdfPages(fname) as pdf:
        g1 = g_samples[:, :, 0]
        g2 = g_samples[:, :, 1]

        means = g1.mean(axis=1)
        stds = g1.std(axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.hist(means, bins=31, histtype="step")
        ax1.set_title(f"Std of means: {means.std():.4g}")
        ax1.set_xlabel("Mean of posteriors of g1")
        ax1.axvline(means.mean(), linestyle="--", color="k", label="mean")
        ax1.legend()

        ax2.hist(stds, bins=31, histtype="step")
        ax2.set_xlabel("Std of posteriors of g1")
        ax2.axvline(stds.mean(), linestyle="--", color="k", label="mean")
        ax2.legend()
        pdf.savefig(fig)
        plt.close(fig)

        means = g2.mean(axis=1)
        stds = g2.std(axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.hist(means, bins=31, histtype="step")
        ax1.set_title(f"Std of means: {means.std():.4g}")
        ax1.set_xlabel("Mean of posteriors of g2")
        ax1.axvline(means.mean(), linestyle="--", color="k", label="mean")
        ax1.legend()

        ax2.hist(stds, bins=31, histtype="step")
        ax2.set_xlabel("Std of posteriors of g2")
        ax2.axvline(stds.mean(), linestyle="--", color="k", label="mean")
        ax2.legend()
        pdf.savefig(fig)
        plt.close(fig)


def main(seed: int = 44):
    np.random.seed(seed)
    pdir = DATA_DIR / "cache_chains" / f"toy_shear_{seed}"
    assert pdir.exists()
    all_g_samples = []
    for fpath in pdir.iterdir():
        if "g_samples" in fpath.name:
            _g_samples = jnp.load(fpath)
            all_g_samples.append(_g_samples)
    g_samples = jnp.concatenate(all_g_samples, axis=0)
    assert g_samples.shape == (1000, 3000, 2)

    # make plots
    make_trace_plots(g_samples)
    make_contour_plots(g_samples)
    make_posterior_calibration(g_samples)
    make_histogram_mbias(g_samples)
    make_histogram_means_and_stds(g_samples)


if __name__ == "__main__":
    typer.run(main)
