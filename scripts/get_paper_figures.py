#!/usr/bin/env python3
import os

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
from jax import vmap
from tqdm import tqdm

from bpd import DATA_DIR, HOME_DIR
from bpd.draw import draw_exponential_galsim
from bpd.io import load_dataset
from bpd.plotting import (
    get_jack_bias,
    get_timing_figure,
    set_rc_params,
)
from bpd.sample import sample_galaxy_params_simple
from bpd.utils import get_snr

set_rc_params()

FIG_DIR = HOME_DIR / "paper"
CHAIN_DIR = DATA_DIR / "cache_chains"

INPUT_PATHS = {
    "timing_results": CHAIN_DIR / "exp23_42" / "timing_results_42.npz",
    "timing_conv": CHAIN_DIR / "exp23_42" / "convergence_results_42.npz",
    "only_shear_post": CHAIN_DIR / "exp46_43" / "g_samples_43_43_plus.npy",
    "shear_jack_1": CHAIN_DIR / "exp46_43" / "g_samples_jack_43.npz",
    "samples_all_free": CHAIN_DIR / "exp61_42" / "shear_samples_42_plus.npz",
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
        keys = jax.random.split(k, n_gals)
        _sample_fnc = partial(
            sample_galaxy_params_simple,
            shape_noise=0.2,
            mean_logflux=2.5,
            sigma_logflux=0.4,
            mean_loghlr=-0.5,
            sigma_loghlr=0.05,
        )
        galaxy_params = vmap(_sample_fnc)(keys)
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
    ax.hist(snrs, bins=41, histtype="step", density=True)
    ax.axvline(jnp.median(snrs), linestyle="--", color="k", label=r"\rm Median")
    ax.set_xlabel(r"\rm SNR")
    ax.legend()
    fig.savefig(fpath, format="png")


def make_timing_figure(fpath: str | Path):
    timing_results = load_dataset(INPUT_PATHS["timing_results"])

    max_n_gal = str(max(int(k) for k in timing_results))
    fig = get_timing_figure(results=timing_results, max_n_gal=max_n_gal)
    fig.savefig(fpath, format="png")


def make_contour_shear_figure(fpath: str | Path):
    shear_post1 = np.load(INPUT_PATHS["only_shear_post"])
    assert shear_post1.ndim == 2
    assert shear_post1.shape[1] == 2

    g1 = shear_post1[:, 0]
    g2 = shear_post1[:, 1]

    data = {"g1": g1, "g2": g2}
    df = pandas.DataFrame.from_dict(data)

    c = ChainConsumer()
    # Customise the chain when you add it
    chain = Chain(
        samples=df,
        name="Only Shear",
        marker_style="*",
    )

    c.add_chain(chain)
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
    ds = load_dataset(INPUT_PATHS["samples_all_free"])
    samples = ds["samples"]
    truth = ds["truth"]

    samples.pop("g1")
    samples.pop("g2")
    truth.pop("g1")
    truth.pop("g2")

    names = ("sigma_e", "mean_logflux", "sigma_logflux", "mean_loghlr", "sigma_loghlr")

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
                "mean_logflux": r"$\mu_{f}$",
                "sigma_logflux": r"$\sigma_{f}$",
                "mean_loghlr": r"$\mu_{s}$",
                "sigma_loghlr": r"$\sigma_{s}$",
            },
            label_font_size=75,
            summary_font_size=65,
            tick_font_size=55,
            diagonal_tick_labels=True,
            spacing=0.0,
            max_ticks=4,
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
    make_snr_figure(OUT_PATHS["snr"], overwrite=overwrite)
    make_timing_figure(OUT_PATHS["timing"])
    make_contour_shear_figure(OUT_PATHS["contour_shear"])
    make_contour_hyper_figure(OUT_PATHS["contour_hyper"])
    make_bias_figure(OUT_PATHS["bias"])


if __name__ == "__main__":
    typer.run(main)
