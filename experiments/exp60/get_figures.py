#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.backends.backend_pdf import PdfPages

from bpd import DATA_DIR
from bpd.diagnostics import get_contour_plot
from bpd.io import load_dataset


def make_traces(params: dict, true_params: dict, mode: str, seed: int) -> None:
    """Make trace plots of g1, g2."""
    fname = f"figs/{seed}/traces_{mode}.pdf"
    with PdfPages(fname) as pdf:
        for k, v in params.items():
            assert v.ndim == 1
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

            ax.plot(v)
            ax.axhline(v.mean(), color="r", linestyle="--")
            ax.axhline(true_params[k], color="k", linestyle="--")
            ax.set_ylabel(k, fontsize=20)

            pdf.savefig(fig)
            plt.close(fig)


def make_contour_plots(
    all_samples: np.ndarray,
    truth: dict,
    mode: str,
    seed: int,
) -> None:
    """Make figure of contour plot on g1, g2."""
    fname = f"figs/{seed}/contours_{mode}.pdf"
    with PdfPages(fname) as pdf:
        fig = get_contour_plot([all_samples], ["post"], truth, figsize=(10, 10))
        pdf.savefig(fig)
        plt.close(fig)


def main(seed: int, tag: str = typer.Option()):
    np.random.seed(seed)

    fig_path = Path(f"figs/{seed}")
    if not fig_path.exists():
        Path(fig_path).mkdir(exist_ok=True, parents=True)

    # load data
    pdir = DATA_DIR / "cache_chains" / tag

    data_plus = load_dataset(pdir / f"shear_samples_{seed}_plus.npz")
    data_minus = load_dataset(pdir / f"shear_samples_{seed}_minus.npz")

    samples_plus = data_plus["samples"]
    samples_minus = data_minus["samples"]

    truth_plus = data_plus["truth"]
    truth_minus = data_minus["truth"]

    # plus
    make_traces(samples_plus, truth_plus, "plus", seed=seed)
    make_contour_plots(samples_plus, truth_plus, "plus", seed=seed)

    # minus
    make_traces(samples_minus, truth_minus, "minus", seed=seed)
    make_contour_plots(samples_minus, truth_minus, "minus", seed=seed)


if __name__ == "__main__":
    typer.run(main)
