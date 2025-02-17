#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.backends.backend_pdf import PdfPages

from bpd import DATA_DIR
from bpd.diagnostics import get_contour_plot
from bpd.io import load_dataset


def make_trace_plots(g_samples: np.ndarray, mode: str, seed: int) -> None:
    """Make trace plots of g1, g2."""
    fname = f"figs/{seed}/traces_{mode}.pdf"
    with PdfPages(fname) as pdf:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        g1 = g_samples[:, 0]
        g2 = g_samples[:, 1]

        ax1.plot(g1)
        ax1.axhline(g1.mean(), color="r", linestyle="--")
        ax1.axhline(0.02 if mode == "plus" else -0.02, color="k", linestyle="--")

        ax2.plot(g2)
        ax2.axhline(g2.mean(), color="r", linestyle="--")
        ax2.axhline(0.0, color="k", linestyle="--")

        pdf.savefig(fig)
        plt.close(fig)


def make_scatter_shape_plots(
    e_post: np.ndarray, seed: int, n_examples: int = 10
) -> None:
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


def make_hists(g_samples: np.ndarray, mode: str, seed: int) -> None:
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
    g_samples: np.ndarray,
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


def get_jack_traces(
    g_plus_jack: np.ndarray,
    g_minus_jack: np.ndarray,
    g1_true: float,
    g2_true: float,
    seed: int,
):
    fname = f"figs/{seed}/jack_traces.pdf"
    assert g_plus_jack.ndim == 3
    assert g_plus_jack.shape[-1] == 2
    n_jack = g_plus_jack.shape[0]
    with PdfPages(fname) as pdf:
        for ii in range(n_jack):
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))

            ax1.plot(g_plus_jack[ii, :, 0])
            ax1.axhline(g1_true, color="k", linestyle="--")
            ax1.axhline(g_plus_jack[ii, :, 0].mean(), color="r", linestyle="--")

            ax2.plot(g_minus_jack[ii, :, 0])
            ax2.axhline(-g1_true, color="k", linestyle="--")
            ax2.axhline(g_minus_jack[ii, :, 0].mean(), color="r", linestyle="--")

            ax3.plot(g_plus_jack[ii, :, 1])
            ax3.axhline(g2_true, color="k", linestyle="--")
            ax3.axhline(g_plus_jack[ii, :, 1].mean(), color="r", linestyle="--")

            ax4.plot(g_minus_jack[ii, :, 1])
            ax4.axhline(-g2_true, color="k", linestyle="--")
            ax4.axhline(g_minus_jack[ii, :, 1].mean(), color="r", linestyle="--")

            pdf.savefig(fig)
            plt.close(fig)


def get_jack_contours(
    g_plus_jack: np.ndarray,
    g_minus_jack: np.ndarray,
    g1_true: float,
    g2_true: float,
    seed: int,
):
    fname = f"figs/{seed}/jack_traces.pdf"
    assert g_plus_jack.ndim == 3
    assert g_plus_jack.shape[-1] == 2
    n_jacks = g_plus_jack.shape[0]
    fname = f"figs/{seed}/contours_jack.pdf"
    with PdfPages(fname) as pdf:
        for _ in range(10):
            idx = np.random.choice(np.arange(0, n_jacks))

            truth = {
                "g1+": g1_true,
                "g2+": g2_true,
                "g1-": -g1_true,
                "g2-": -g2_true,
            }
            g_dict = {
                "g1+": g_plus_jack[idx, :, 0],
                "g2+": g_plus_jack[idx, :, 1],
                "g1-": g_minus_jack[idx, :, 0],
                "g2-": g_minus_jack[idx, :, 1],
            }
            fig = get_contour_plot([g_dict], ["post"], truth, figsize=(10, 10))
            pdf.savefig(fig)
            plt.close(fig)


def get_jack_scatter_plot(
    g_plus_jack: np.ndarray,
    g_minus_jack: np.ndarray,
    g1_true: float,
    g2_true: float,
    seed: int,
):
    fname = f"figs/{seed}/jack_scatter.pdf"
    assert g_plus_jack.ndim == 3
    assert g_plus_jack.shape[-1] == 2
    with PdfPages(fname) as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        g1p = g_plus_jack[:, :, 0].mean(axis=1)
        g1m = g_minus_jack[:, :, 0].mean(axis=1)

        x = g1p - g1_true
        y = g1m + g1_true
        ax.scatter(x, y, marker="o", color="r")
        ax.plot([x.min(), x.max()], [y.min(), y.max()], "k--")

        pdf.savefig(fig)
        plt.close(fig)


def main(seed: int, tag: str = typer.Option()):
    np.random.seed(seed)

    fig_path = Path(f"figs/{seed}")
    if not fig_path.exists():
        Path(fig_path).mkdir(exist_ok=True)

    # load data
    pdir = DATA_DIR / "cache_chains" / tag
    interim_dict = load_dataset(pdir / f"interim_samples_{seed}_plus.npz")
    e_post_samples = interim_dict["e_post"]
    g1, g2 = interim_dict["true_g"]

    g_samples_plus = np.load(pdir / f"g_samples_{seed}_{seed}_plus.npy")
    g_samples_minus = np.load(pdir / f"g_samples_{seed}_{seed}_minus.npy")
    assert isinstance(g_samples_plus, np.ndarray)

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
    m_samples = (g_samples_plus[:, 0] - g_samples_minus[:, 0]) / 2 / g1 - 1
    c_samples = (g_samples_plus[:, 1] + g_samples_minus[:, 1]) / 2

    fname = f"figs/{seed}/contours_bias.pdf"
    with PdfPages(fname) as pdf:
        ct_fig = get_contour_plot(
            [{"m": m_samples, "c": c_samples}], ["m", "c"], {"m": 0.0, "c": 0.0}
        )
        pdf.savefig(ct_fig)
        plt.close(ct_fig)

    # summary of multiplicative/additive bias results (both JK and not)
    # save to a text file

    summary_fpath = f"figs/{seed}/summary.txt"
    with open(summary_fpath, "w", encoding="utf-8") as f:
        txt = (
            "#### Full results ####\n"
            "Units: 1e-3\n"
            "\n"
            f"m_mean: {m_samples.mean() * 1e3:.4g}\n"
            f"3 * m_std: {m_samples.std() * 3 * 1e3:.4g}\n"
            "\n"
            f"c_mean: {c_samples.mean() * 1e3:.4g}\n"
            f"3 * c_std: {c_samples.std() * 3 * 1e3:.4g}\n"
            ""
        )
        print(txt, file=f)

    jack_fpath = pdir / f"g_samples_jack_{seed}_{seed}.npz"
    if jack_fpath.exists():
        jack_ds = load_dataset(jack_fpath)
        g_plus_jack = jack_ds["g_plus"]
        g_minus_jack = jack_ds["g_minus"]
        assert g_plus_jack.ndim == 3
        assert g_plus_jack.shape[-1] == 2
        assert isinstance(g_plus_jack, np.ndarray)
        n_jack = g_plus_jack.shape[0]

        m_jack = (
            g_plus_jack[..., 0].mean(axis=1) - g_minus_jack[..., 0].mean(axis=1)
        ) / 2 / g1 - 1
        c_jack = (
            g_plus_jack[..., 1].mean(axis=1) + g_minus_jack[..., 1].mean(axis=1)
        ) / 2

        m_jack_mean = m_jack.mean().item()
        m_jack_std = np.sqrt(m_jack.var() * (n_jack - 1)).item()

        c_jack_mean = c_jack.mean().item()
        c_jack_std = np.sqrt(c_jack.var() * (n_jack - 1)).item()

        with open(summary_fpath, "a", encoding="utf-8") as f:
            txt = (
                "#### Jackknife results ####\n"
                "Units: 1e-3\n"
                "\n"
                f"m_mean: {m_jack_mean * 1e3:.4g}\n"
                f"3 * m_std: {m_jack_std * 3 * 1e3:.4g}\n"
                "\n"
                f"c_mean: {c_jack_mean * 1e3:.4g}\n"
                f"3 * c_std: {c_jack_std * 3 * 1e3:.4g}\n"
            )
            print(txt, file=f)

        get_jack_traces(g_plus_jack, g_minus_jack, g1, g2, seed)
        get_jack_contours(g_plus_jack, g_minus_jack, g1, g2, seed)
        get_jack_scatter_plot(g_plus_jack, g_minus_jack, g1, g2, seed)


if __name__ == "__main__":
    typer.run(main)
