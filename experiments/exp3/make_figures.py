#!/usr/bin/env python3

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"


import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from jax import Array
from matplotlib.backends.backend_pdf import PdfPages

from bpd import DATA_DIR
from bpd.diagnostics import get_contour_plot
from bpd.io import load_dataset


def make_trace_plots(g_samples: Array) -> None:
    """Make trace plots of g1, g2."""
    fname = "figs/traces.pdf"
    with PdfPages(fname) as pdf:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        g1 = g_samples[:, 0]
        g2 = g_samples[:, 1]

        ax1.plot(g1)
        ax2.plot(g2)

        pdf.savefig(fig)
        plt.close(fig)


def make_contour_plots(g_samples: Array, n_examples=10) -> None:
    """Make figure of contour plot on g1, g2."""
    fname = "figs/contours.pdf"
    with PdfPages(fname) as pdf:
        truth = {"g1": 0.02, "g2": 0.0}
        g_dict = {"g1": g_samples[:, 0], "g2": g_samples[:, 1]}
        fig = get_contour_plot([g_dict], ["post"], truth)
        pdf.savefig(fig)
        plt.close(fig)


def make_scatter_shape_plots(e_post: Array, n_examples: int = 10) -> None:
    """Show example scatter plots of interim posterior ellipticitites."""
    # make two types, assuming gaussianity and one not assuming gaussianity.
    fname = "figs/scatter_shapes.pdf"

    n_gals, _, _ = e_post.shape

    with PdfPages(fname) as pdf:
        # individual
        for _ in range(n_examples):
            idx = np.random.choice(np.arange(0, n_gals))
            e1, e2 = e_post[idx, :, 0], e_post[idx, :, 1]
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.scatter(e1, e2, marker="x")
            ax.set_title(f"Samples ellipticity index: {idx}")
            ax.set_xlabel("e1", fontsize=14)
            ax.set_ylabel("e2", fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)

        # clusters
        n_clusters = 50
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.set_xlabel("e1", fontsize=14)
        ax.set_ylabel("e2", fontsize=14)
        fig.suptitle(f"{n_clusters} galaxies plotted")
        for _ in range(n_clusters):
            idx = np.random.choice(np.arange(0, n_gals))
            e1, e2 = e_post[idx, :, 0], e_post[idx, :, 1]
            ax.scatter(e1, e2, marker="x")
        pdf.savefig(fig)
        plt.close(fig)


def make_hists(g_samples: Array, e1_samples: Array) -> None:
    """Make histograms of g1 along with std and expected std."""
    fname = "figs/hists.pdf"
    with PdfPages(fname) as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        g1 = g_samples[:, 0]
        e1_std = e1_samples.std()
        g1_exp_std = e1_std / jnp.sqrt(len(e1_samples))

        ax.hist(g1, bins=25, histtype="step")
        ax.axvline(g1.mean(), linestyle="--", color="k")
        ax.set_title(f"Std g1: {g1.std():.4g}; Expected g1 std: {g1_exp_std:.4g}")

        pdf.savefig(fig)
        plt.close(fig)


def main(seed: int = 43):
    # load data
    pdir = DATA_DIR / "cache_chains" / f"test_fixed_shear_inference_images_{seed}"
    e_post_dict = load_dataset(pdir / f"e_post_{seed}.npz")
    e_post_samples = e_post_dict["e_post"]
    g_samples = jnp.load(pdir / f"g_samples_{seed}_{seed}.npy")

    e1_samples = e_post_dict["e1"]

    # make plots
    make_scatter_shape_plots(e_post_samples)
    make_trace_plots(g_samples)
    make_contour_plots(g_samples)
    make_hists(g_samples, e1_samples)


if __name__ == "__main__":
    typer.run(main)
