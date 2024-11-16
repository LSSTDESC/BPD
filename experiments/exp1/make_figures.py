#!/usr/bin/env python3

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import Figure

from bpd import DATA_DIR
from bpd.diagnostics import get_contour_plot, get_gauss_pc_fig, get_pc_fig


def make_trace_plots(g_samples: Array):
    """Make example figure showing example trace plots of shear posteriors."""
    # by default, we choose 10 random traces to plot in 1 PDF file.
    fname = "figs/traces.pdf"
    pdf = PdfPages(fname)

    assert g_samples.ndim == 3
    n_post = g_samples.shape[0]
    indices = np.random.choice(np.arange(n_post), (10,))

    for ii in indices:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
        g1 = g_samples[ii, :, 0]
        g2 = g_samples[ii, :, 0]

        ax1.plot(g1)
        ax2.plot(g2)

        pdf.savefig(fig)
        fig.close()
    pdf.close()


def make_contour_plots(g_samples: Array) -> Figure:
    """Make example figure showing example contour plots of shear posterios"""
    # by default, we choose 10 random contours to plot in 1 PDF file.
    fname = "figs/contours.pdf"
    pdf = PdfPages(fname)

    assert g_samples.ndim == 3
    n_post = g_samples.shape[0]
    indices = np.random.choice(np.arange(n_post), (10,))

    truth = {"g1": 0.02, "g2": 0.0}

    for ii in indices:
        g_dict = {"g1": g_samples[ii, :, 0], "g2": g_samples[ii, :, 1]}
        fig = get_contour_plot([g_dict], [f"post_{ii}"], truth)
        pdf.savefig(fig)
        fig.close()
    pdf.close()


def make_posterior_calibration(g_samples: Array) -> Figure:
    """Output posterior calibration figure."""
    # make two types, assuming gaussianity and one not assuming gaussianity.
    fname = "figs/calibration.pdf"
    pdf = PdfPages(fname)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    get_gauss_pc_fig(ax1, g_samples[..., 0], truth=0.02, param_name="g1 (gauss)")
    get_pc_fig(ax2, g_samples[..., 0], truth=0.02, param_name="g1 (full)")
    pdf.savefig(fig)
    fig.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    get_gauss_pc_fig(ax1, g_samples[..., 1], truth=0.0, param_name="g2 (gauss)")
    get_pc_fig(ax2, g_samples[..., 1], truth=0.0, param_name="g2 (full)")
    pdf.savefig(fig)

    pdf.close()


def main():
    pdir = DATA_DIR / "cache_chains" / "toy_shear_42"
    assert pdir.exsits()
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
