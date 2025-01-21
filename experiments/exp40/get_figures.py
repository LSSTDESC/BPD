#!/usr/bin/env python3

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from jax import Array
from matplotlib.backends.backend_pdf import PdfPages

from bpd import DATA_DIR
from bpd.diagnostics import get_contour_plot
from bpd.io import load_dataset


def make_trace_plots(g_samples: Array, mode: str, seed: int) -> None:
    """Make trace plots of g1, g2."""
    fname = f"figs/{seed}/traces_{mode}.pdf"
    with PdfPages(fname) as pdf:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        g1 = g_samples[:, 0]
        g2 = g_samples[:, 1]

        ax1.plot(g1)
        ax2.plot(g2)

        pdf.savefig(fig)
        plt.close(fig)


def make_scatter_shape_plots(e_post: Array, seed: int, n_examples: int = 10) -> None:
    """Show example scatter plots of interim posterior ellipticitites."""
    # make two types, assuming gaussianity and one not assuming gaussianity.
    fname = f"figs/{seed}/scatter_shapes.pdf"

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


def make_hists(g_samples: Array, mode: str, seed: int) -> None:
    """Make histograms of g1 along with std and expected std."""
    fname = f"figs/{seed}/hists_{mode}.pdf"
    with PdfPages(fname) as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        g1 = g_samples[:, 0]

        ax.hist(g1, bins=25, histtype="step")
        ax.axvline(g1.mean(), linestyle="--", color="k")
        ax.set_title(f"Std g1: {g1.std():.4g}")

        pdf.savefig(fig)
        plt.close(fig)


def make_contour_plots(
    g_samples: Array,
    mode: str,
    g1_true: float,
    g2_true: float,
    seed: int,
) -> None:
    """Make figure of contour plot on g1, g2."""
    fname = f"figs/{seed}/contours_{mode}.pdf"
    with PdfPages(fname) as pdf:
        truth = {"g1": g1_true, "g2": g2_true}
        g_dict = {"g1": g_samples[:, 0], "g2": g_samples[:, 1]}
        fig = get_contour_plot([g_dict], ["post"], truth)
        pdf.savefig(fig)
        plt.close(fig)


def main(seed: int):
    np.random.seed(seed)

    if not Path("figs/{seed}").exists():
        Path("figs/{seed}").mkdir(exist_ok=True)

    # load data
    pdir = DATA_DIR / "cache_chains" / f"exp40_{seed}"
    interim_dict = load_dataset(pdir / f"interim_samples_{seed}_plus.npz")
    e_post_samples = interim_dict["e_post"]
    g1, g2 = interim_dict["true_g"]

    g_samples_plus = jnp.load(pdir / f"g_samples_{seed}_{seed}_plus.npy")
    g_samples_minus = jnp.load(pdir / f"g_samples_{seed}_{seed}_minus.npy")

    # make plots
    make_scatter_shape_plots(e_post_samples, seed=seed)

    # plus
    make_trace_plots(g_samples_plus, "plus", seed=seed)
    make_hists(g_samples_plus, "plus", seed=seed)
    make_contour_plots(g_samples_plus, "plus", g1_true=g1, g2_true=g2, seed=seed)

    # minus
    make_trace_plots(g_samples_minus, "minus", seed=seed)
    make_hists(g_samples_minus, "minus", seed=seed)
    make_contour_plots(g_samples_minus, "minus", g1_true=-g1, g2_true=g2, seed=seed)

    # bias
    m_samples = (g_samples_plus[:, 0] - g_samples_minus[:, 0]) * 0.5 / g1 - 1
    c_samples = (g_samples_plus[:, 1] + g_samples_minus[:, 1]) * 0.5

    fname = f"figs/{seed}/contours_bias.pdf"
    with PdfPages(fname) as pdf:
        ct_fig = get_contour_plot(
            [{"m": m_samples, "c": c_samples}], ["m", "c"], {"m": 0.0, "c": 0.0}
        )
        pdf.savefig(ct_fig)
        plt.close(ct_fig)


if __name__ == "__main__":
    typer.run(main)
