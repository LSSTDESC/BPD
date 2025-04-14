#!/usr/bin/env python3
import os
from tkinter import font

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas
import typer
from chainconsumer import Chain, ChainConfig, ChainConsumer, Truth
from chainconsumer.plotting.config import PlotConfig
from tqdm import tqdm

from bpd import DATA_DIR, HOME_DIR
from bpd.draw import draw_exponential_galsim
from bpd.io import load_dataset
from bpd.plotting import (
    get_jack_bias,
    get_timing_figure,
    set_rc_params,
)
from bpd.sample import sample_galaxy_params_skew
from bpd.utils import get_snr

set_rc_params()

FIG_DIR = HOME_DIR / "paper"
CHAIN_DIR = DATA_DIR / "cache_chains"

INPUT_PATHS = {
    "timing_results": CHAIN_DIR / "exp23_43" / "timing_results_43.npz",
    "timing_conv": CHAIN_DIR / "exp23_43" / "convergence_results_43.npz",
    "exp70_sp": CHAIN_DIR / "exp70_44" / "g_samples_442_plus.npy",
    "exp70_sm": CHAIN_DIR / "exp70_44" / "g_samples_442_minus.npy",
    "exp70_errs": CHAIN_DIR / "exp70_44" / "g_samples_449_errs.npz",
    "exp71_sp": CHAIN_DIR / "exp71_44" / "shear_samples_442_plus.npz",
    "exp71_sm": CHAIN_DIR / "exp71_44" / "shear_samples_442_minus.npz",
    "exp71_errs": CHAIN_DIR / "exp71_44" / "g_samples_449_errs.npz",
    "exp72_sp": CHAIN_DIR / "exp72_45" / "g_samples_452_plus.npy",
    "exp72_sm": CHAIN_DIR / "exp72_45" / "g_samples_452_minus.npy",
    "exp72_errs": CHAIN_DIR / "exp72_45" / "g_samples_454_errs.npz",
    "exp73_sp": CHAIN_DIR / "exp73_45" / "tmp2" / "shear_samples_452_plus.npz",
    "exp73_sm": CHAIN_DIR / "exp73_45" / "tmp2" / "shear_samples_452_minus.npz",
    "exp73_errs": CHAIN_DIR / "exp73_45" / "tmp2" / "g_samples_454_errs.npz",
}


OUT_PATHS = {
    "snr": FIG_DIR / "snr.png",
    "timing": FIG_DIR / "timing.png",
    "contour_shear": FIG_DIR / "contour_shear.png",
    "contour_hyper": FIG_DIR / "contour_hyper.png",
    "bias": FIG_DIR / "bias.png",
}


def make_snr_figure(fpath: str | Path, overwrite: bool = False):
    snr_cache_fpath = HOME_DIR / "paper" / "snr_cache.npy"

    if Path(snr_cache_fpath).exists() and not overwrite:
        snrs = jnp.load(snr_cache_fpath)
    else:
        n_gals = 10_000
        k = jax.random.key(42)
        galaxy_params = sample_galaxy_params_skew(
            k,
            n=n_gals,
            shape_noise=0.2,
            mean_logflux=2.45,
            a_logflux=14.0,
            sigma_logflux=0.4,
            mean_loghlr=-0.4,
            sigma_loghlr=0.05,
        )
        draw_params = {**galaxy_params}
        draw_params["f"] = 10 ** draw_params.pop("lf")
        draw_params["hlr"] = 10 ** draw_params.pop("lhlr")
        draw_params["x"] = jnp.zeros_like(draw_params["x"])
        draw_params["y"] = jnp.zeros_like(draw_params["y"])

        _draw_galsim = partial(draw_exponential_galsim, slen=63)
        noiseless = []

        for ii in tqdm(range(n_gals)):
            _params = {k: v[ii] for k, v in draw_params.items()}
            noiseless.append(_draw_galsim(**_params))

        noiseless = jnp.stack(noiseless, axis=0)

        snrs = []
        for ii in tqdm(range(n_gals)):
            snrs.append(get_snr(noiseless[ii], background=1.0))

        snrs = jnp.stack(snrs, axis=0)
        jnp.save(snr_cache_fpath, snrs)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(snrs, bins=41, histtype="step", density=True, range=(10, 80))
    ax.set_xticks(np.arange(10, 81, 10))
    ax.axvline(
        jnp.median(snrs),
        linestyle="--",
        color="k",
        label=r"\rm Median SNR = {:.2f}".format(jnp.median(snrs)),
    )
    ax.set_xlabel(r"\rm SNR")
    ax.legend()
    fig.savefig(fpath, format="png")


def make_timing_figure(fpath: str | Path):
    set_rc_params(fontsize=24)
    timing_results = load_dataset(INPUT_PATHS["timing_results"])
    max_n_gal = str(max(int(k) for k in timing_results))
    fig = get_timing_figure(results=timing_results, max_n_gal_str=max_n_gal)
    fig.savefig(fpath, format="png")

    set_rc_params()  # reset to default fontsize


def make_contour_shear_figure(fpath: str | Path):
    g_exp72 = np.load(INPUT_PATHS["exp72_sp"])
    samples_exp73 = load_dataset(INPUT_PATHS["exp73_sp"])
    g_exp73 = jnp.stack(
        [samples_exp73["samples"]["g1"], samples_exp73["samples"]["g2"]], axis=-1
    )
    assert g_exp72.ndim == 2
    assert g_exp72.shape[1] == 2
    assert g_exp73.ndim == 2
    assert g_exp73.shape[1] == 2
    assert g_exp72.shape[0] == g_exp73.shape[0]

    data1 = {"g1": g_exp72[:, 0], "g2": g_exp72[:, 1]}
    df1 = pandas.DataFrame.from_dict(data1)

    # Customise the chain when you add it
    c = ChainConsumer()
    chain = Chain(
        samples=df1,
        name="Only Shear",
        marker_style="*",
    )

    c.add_chain(chain)
    c.set_override(ChainConfig(sigmas=[0, 1, 2]))
    c.add_truth(Truth(location={"g1": 0.02, "g2": 0.0}, color="k", line_width=2.0))

    # now the other too
    data2 = {"g1": g_exp73[:, 0], "g2": g_exp73[:, 1]}
    df2 = pandas.DataFrame.from_dict(data2)
    chain2 = Chain(
        samples=df2,
        name="All Free",
        marker_style="*",
    )
    c.add_chain(chain2)
    c.set_override(ChainConfig(sigmas=[0, 1, 2]))
    c.add_truth(Truth(location={"g1": 0.02, "g2": 0.0}, color="k", line_width=2.0))

    c.set_plot_config(
        PlotConfig(
            usetex=True,
            plot_hists=False,
            labels={"g1": "$g_{1}$", "g2": "$g_{2}$"},
            label_font_size=34,
            tick_font_size=24,
            diagonal_tick_labels=False,
        )
    )
    fig = c.plotter.plot(figsize=(10, 10))

    fig.savefig(fpath, format="png")


def make_contour_hyper_figure(fpath: str | Path):
    ds = load_dataset(INPUT_PATHS["exp73_sp"])
    samples = ds["samples"]
    truth = ds["truth"]

    samples.pop("g1")
    samples.pop("g2")
    truth.pop("g1")
    truth.pop("g2")

    names = (
        "sigma_e",
        "a_logflux",
        "mean_logflux",
        "sigma_logflux",
        "mean_loghlr",
        "sigma_loghlr",
    )

    out = {k: samples[k] for k in names}
    df = pandas.DataFrame.from_dict(out)

    c = ChainConsumer()
    chain = Chain(samples=df, name="Example I", marker_style="*")

    c.add_chain(chain)
    c.set_override(ChainConfig(sigmas=[0, 1, 2]))
    c.add_truth(Truth(location=truth, color="k", line_width=4.0))

    c.set_plot_config(
        PlotConfig(
            usetex=True,
            plot_hists=True,
            labels={
                "sigma_e": r"$\sigma_{e}$",
                "a_logflux": r"$a_{f}$",
                "mean_logflux": r"$\mu_{f}$",
                "sigma_logflux": r"$\sigma_{f}$",
                "mean_loghlr": r"$\mu_{s}$",
                "sigma_loghlr": r"$\sigma_{s}$",
            },
            label_font_size=60,
            summarise=True,
            summary_font_size=45,
            tick_font_size=50,
            diagonal_tick_labels=True,
            spacing=0.0,
            max_ticks=3,
        )
    )
    fig = c.plotter.plot(figsize=(50, 50))
    fig.tight_layout()

    fig.savefig(fpath, format="png")


def make_bias_figure(fpath: str | Path):
    jack_ds1 = load_dataset(INPUT_PATHS["shear_jack_1"])
    g_plus_jack = jack_ds1["g_plus"]
    g_minus_jack = jack_ds1["g_minus"]

    m_mean, m_std, c_mean, c_std = get_jack_bias(
        g_plus_jack, g_minus_jack, g1_true=0.02
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # multiplicative bias
    ax1.errorbar(x=m_mean, y=0, xerr=m_std * 3, color="k", fmt="-o", capsize=10.0)

    ax1.set_yticks([0.0, 0.25, 0.5])
    ax1.set_yticklabels([r"\rm Only Shear", r"\rm Example 1", r"\rm Example 2"])
    ax1.set_ylim(-0.1, 0.75)
    ax1.set_title(r"\rm Multiplicative bias $m$")
    ax1.set_xlim(-0.01, 0.01)

    area = np.linspace(-2e-3, 2e-3, 1000)
    ax1.fill_between(area, y1=-0.1, y2=1.1, alpha=0.25, color="k", label="Requirement")
    ax1.legend()

    # additive bias
    ax2.errorbar(x=c_mean, y=0, xerr=c_std * 3, color="k", fmt="-o", capsize=10.0)

    ax2.set_title(r"\rm Additive bias $c$")
    ax2.set_xlim(-0.005, 0.005)

    area = np.linspace(-2e-3, 2e-3, 1000)
    ax2.fill_between(area, y1=-0.1, y2=1.1, alpha=0.25, color="k")

    fig.savefig(fpath, format="png")


def main(overwrite: bool = False):
    # make_snr_figure(OUT_PATHS["snr"], overwrite=overwrite)
    # make_timing_figure(OUT_PATHS["timing"])
    # make_contour_shear_figure(OUT_PATHS["contour_shear"])
    make_contour_hyper_figure(OUT_PATHS["contour_hyper"])
    # make_bias_figure(OUT_PATHS["bias"])


if __name__ == "__main__":
    typer.run(main)
