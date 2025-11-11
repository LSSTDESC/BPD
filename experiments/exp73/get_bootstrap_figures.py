#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.backends.backend_pdf import PdfPages

from bpd import DATA_DIR
from bpd.diagnostics import get_contour_plot
from bpd.io import load_dataset


def make_one_trace(params: dict, true_params: dict, mode: str, seed: int):
    fname = f"figs/{seed}/full_trace_{mode}.pdf"
    with PdfPages(fname) as pdf:
        fig, ax = plt.subplots(8, 1, figsize=(8, 20))
        for jj, (k, v) in enumerate(params.items()):
            ax[jj].plot(v)
            ax[jj].axhline(v.mean(), color="r", linestyle="--")
            ax[jj].axhline(true_params[k], color="k", linestyle="--")
            ax[jj].set_ylabel(k, fontsize=20)

        pdf.savefig(fig)
        plt.close(fig)


def make_traces(params: dict, true_params: dict, mode: str, seed: int) -> None:
    """Make trace plots of g1, g2."""
    fname = f"figs/{seed}/traces_{mode}.pdf"
    with PdfPages(fname) as pdf:
        for ii in range(10):
            fig, ax = plt.subplots(8, 1, figsize=(8, 20))
            for jj, (k, v) in enumerate(params.items()):
                ax[jj].plot(v[ii])
                ax[jj].axhline(v[ii].mean(), color="r", linestyle="--")
                ax[jj].axhline(true_params[k], color="k", linestyle="--")
                ax[jj].set_ylabel(k, fontsize=20)

            pdf.savefig(fig)
            plt.close(fig)


def make_contour_plots(
    samples: np.ndarray,
    truth: dict,
    mode: str,
    seed: int,
) -> None:
    """Make figure of contour plot on g1, g2."""
    fname = f"figs/{seed}/contours_{mode}.pdf"
    with PdfPages(fname) as pdf:
        for ii in range(10):  # just as an example
            samples_ii = {k: v[ii] for k, v in samples.items()}
            fig = get_contour_plot([samples_ii], ["post"], truth, figsize=(10, 10))
            pdf.savefig(fig)
            plt.close(fig)


def main(
    seed: int,
    boot_name: str = typer.Option(),
    fpp_name: str = typer.Option(),
    tag: str = typer.Option(),
):
    np.random.seed(seed)
    fig_path = Path(f"figs/{seed}")
    if not fig_path.exists():
        Path(fig_path).mkdir(exist_ok=True, parents=True)

    pdir = DATA_DIR / "cache_chains" / tag
    boot_ds = load_dataset(pdir / boot_name)
    truth_plus = {
        "g1": 0.02,
        "g2": 0.0,
        "sigma_e": 0.2,
        "a_logflux": 14.0,
        "mean_logflux": 2.45,
        "sigma_logflux": 0.4,
        "mean_loghlr": -0.4,
        "sigma_loghlr": 0.05,
    }

    # plus full
    fpp = load_dataset(pdir / fpp_name)
    print(fpp["samples"].keys())
    make_one_trace(fpp["samples"], truth_plus, mode="plus", seed=seed)

    # plus
    make_traces(boot_ds["plus"], truth_plus, "plus", seed=seed)
    make_contour_plots(boot_ds["plus"], truth_plus, "plus", seed=seed)


if __name__ == "__main__":
    typer.run(main)
