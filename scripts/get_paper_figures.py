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
from tqdm import tqdm

from bpd import DATA_DIR, HOME_DIR
from bpd.draw import draw_exponential_galsim
from bpd.io import load_dataset, save_dataset
from bpd.plotting import (
    get_timing_figure,
    set_rc_params,
)
from bpd.sample import sample_galaxy_params_skew
from bpd.utils import DEFAULT_HYPERPARAMS, get_snr

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
    "exp73_sp": CHAIN_DIR / "exp73_45" / "shear_samples_452_plus.npz",
    "exp73_sm": CHAIN_DIR / "exp73_45" / "shear_samples_452_minus.npz",
    "exp73_errs": CHAIN_DIR / "exp73_45" / "tmp2" / "g_samples_454_errs.npz",
}


OUT_PATHS = {
    "galaxy_distributions": FIG_DIR / "gprop_dists.png",
    "timing": FIG_DIR / "timing.png",
    "timing2": FIG_DIR / "timing2.png",
    "error_bar": FIG_DIR / "error_bar.png",
    "contour_shear": FIG_DIR / "contour_shear.png",
    "contour_hyper": FIG_DIR / "contour_hyper.png",
    "bias": FIG_DIR / "table_bias.txt",
}


def make_distribution_figure(fpath: str | Path, overwrite: bool = False):
    set_rc_params(fontsize=36, legend_fontsize=30)
    cache_fpath = HOME_DIR / "paper" / "gprop_cache.npz"

    if Path(cache_fpath).exists() and not overwrite:
        params = load_dataset(cache_fpath)
    else:
        # this makes it the exact same dataset of exp 70-71
        n_gals = 10_000
        k = jax.random.key(441)
        k1, _, _ = jax.random.split(k, 3)
        galaxy_params = sample_galaxy_params_skew(k1, n=n_gals, **DEFAULT_HYPERPARAMS)
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
        params = {
            "snr": snrs,
            "f": 10 ** galaxy_params["lf"],
            "hlr": 10 ** galaxy_params["lhlr"],
            "e1": galaxy_params["e1"],
            "e2": galaxy_params["e2"],
        }

        # converts to numpy arrays
        save_dataset(params, cache_fpath, overwrite=overwrite)
        params = load_dataset(cache_fpath)  # now it's numpy arrays not jax arrays

    # plot flux, hlr, shapes (in one plot), and snr histograms
    # indicate median with a black dashed line
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.hist(
        params["f"],
        bins=41,
        histtype="step",
        density=True,
        range=(0, 4000),
    )
    ax1.set_xlabel(r"\rm Flux")
    ax1.axvline(
        np.median(params["f"]),
        linestyle="--",
        color="k",
        label=r"\rm Median = {:.2f}".format(np.median(params["f"])),
    )
    # add mean and sigma to title
    ax1.set_title(
        r"$\mu = {:.2f}, \sigma = {:.2f}$".format(
            np.mean(params["f"]), np.std(params["f"])
        )
    )
    ax1.legend()

    ax2.hist(params["snr"], bins=41, histtype="step", density=True, range=(10, 80))
    ax2.set_xticks(np.arange(10, 81, 10))
    ax2.axvline(
        np.median(params["snr"]),
        linestyle="--",
        color="k",
        label=r"\rm Median SNR = {:.2f}".format(np.median(params["snr"])),
    )
    ax2.set_xlabel(r"\rm SNR")
    ax2.legend()
    ax2.set_title(
        r"$\mu = {:.2f}, \sigma = {:.2f}$".format(
            np.mean(params["snr"]), np.std(params["snr"])
        )
    )

    ax3.hist(
        params["hlr"],
        bins=41,
        histtype="step",
        density=True,
    )
    ax3.set_xlabel(r"\rm HLR")
    ax3.axvline(
        np.median(params["hlr"]),
        linestyle="--",
        color="k",
        label=r"\rm Median = {:.2f}".format(np.median(params["hlr"])),
    )
    ax3.set_title(
        r"$\mu = {:.2f}, \sigma = {:.2f}$".format(
            np.mean(params["hlr"]), np.std(params["hlr"])
        )
    )
    ax3.set_xscale("log")
    ax3.legend()

    # e1, e2 overlaid
    _, bins, _ = ax4.hist(
        params["e1"],
        bins=41,
        histtype="step",
        density=True,
        label=r"$\varepsilon_{1}$",
        color="C0",
    )
    ax4.hist(
        params["e2"],
        bins=bins,
        histtype="step",
        density=True,
        label=r"$\varepsilon_{2}$",
        color="C1",
    )
    ax4.set_xlabel(r"$\varepsilon_{1, 2}$")
    mu = np.mean(params["e1"])
    if abs(mu) < 1e-2:
        mu = 0.00
    ax4.set_title(r"$\mu= {:.2f}, \sigma = {:.2f}$".format(mu, np.std(params["e1"])))
    ax4.legend()

    fig.tight_layout()
    fig.savefig(fpath, format="png")
    plt.close(fig)


def make_timing_figure(fpath1: Path, fpath2: Path):
    set_rc_params(fontsize=24)

    # get avg ESS across all galaxy properties
    conv_results = load_dataset(INPUT_PATHS["timing_conv"])
    ess_dict = conv_results["ess"]
    avg_ess = np.mean([np.mean(ess_dict[k]) for k in ess_dict])
    print(f"Avg. ESS: {avg_ess}")

    timing_results = load_dataset(INPUT_PATHS["timing_results"])

    max_n_gal = str(max(int(k) for k in timing_results))
    fig1, fig2 = get_timing_figure(
        results=timing_results, max_n_gal_str=max_n_gal, avg_ess=avg_ess
    )
    fig1.savefig(fpath1, format="png")
    fig2.savefig(fpath2, format="png")
    plt.close(fig1)
    plt.close(fig2)


def make_errorbar_figure(fpath: str | Path):
    set_rc_params()  # reset to default fontsize
    g_exp1 = np.load(INPUT_PATHS["exp70_sp"])
    g_exp3 = np.load(INPUT_PATHS["exp72_sp"])
    c = ChainConsumer()
    assert g_exp1.ndim == 2
    assert g_exp1.shape[1] == 2
    assert g_exp3.ndim == 2
    assert g_exp3.shape[1] == 2

    data1 = {"g1": g_exp1[:, 0], "g2": g_exp1[:, 1]}
    df1 = pandas.DataFrame.from_dict(data1)

    data3 = {"g1": g_exp3[:, 0], "g2": g_exp3[:, 1]}
    df3 = pandas.DataFrame.from_dict(data3)

    # Customise the chain when you add it
    c = ChainConsumer()
    chain1 = Chain(
        samples=df1,
        name="Only Shear",
        marker_style="*",
    )
    chain3 = Chain(
        samples=df3,
        name="All Free",
        marker_style="*",
    )

    c.add_chain(chain1)
    c.add_chain(chain3)

    c.add_truth(Truth(location={"g1": 0.02, "g2": 0.0}, color="k", line_width=2.0))

    c.set_plot_config(
        PlotConfig(
            usetex=True,
            labels={"g1": "$g_{1}$", "g2": "$g_{2}$"},
            label_font_size=30,
            tick_font_size=24,
        )
    )

    fig = c.plotter.plot_summary(
        errorbar=True,
        figsize=4.0,
        # extra_parameter_spacing=0.8,
        # vertical_spacing_ratio=1.5,
    )
    # fig.tight_layout()
    plt.close(fig)
    fig.savefig(fpath, format="png")


def make_contour_shear_figure1(fpath: str | Path):
    set_rc_params()  # reset to default fontsize
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
        name=r"\texttt{all-fixed}",
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
        name=r"\texttt{all-free}",
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
    fig.tight_layout()
    plt.close(fig)
    fig.savefig(fpath, format="png")


def make_contour_hyper_figure(fpath: str | Path):
    set_rc_params()
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
    plt.close(fig)


def get_bias_table(fpath: str | Path):
    """Create a latex table of mean multiplicative and additive bias, as well as their errors from each experiment."""

    # load datasets
    gp1 = np.load(INPUT_PATHS["exp70_sp"])
    gm1 = np.load(INPUT_PATHS["exp70_sm"])
    assert gp1.ndim == 2 and gm1.ndim == 2

    dsp2 = load_dataset(INPUT_PATHS["exp71_sp"])
    dsm2 = load_dataset(INPUT_PATHS["exp71_sm"])
    gp2 = jnp.stack([dsp2["samples"]["g1"], dsp2["samples"]["g2"]], axis=-1)
    gm2 = jnp.stack([dsm2["samples"]["g1"], dsm2["samples"]["g2"]], axis=-1)

    gp3 = np.load(INPUT_PATHS["exp72_sp"])
    gm3 = np.load(INPUT_PATHS["exp72_sm"])
    assert gp3.ndim == 2 and gm3.ndim == 2

    dsp4 = load_dataset(INPUT_PATHS["exp73_sp"])
    dsm4 = load_dataset(INPUT_PATHS["exp73_sm"])
    gp4 = jnp.stack([dsp4["samples"]["g1"], dsp4["samples"]["g2"]], axis=-1)
    gm4 = jnp.stack([dsm4["samples"]["g1"], dsm4["samples"]["g2"]], axis=-1)

    # get mean multiplicative and additive bias for each experiment
    m1_mean = np.mean(gp1[:, 0] - gm1[:, 0]) / 2 / 0.02 - 1
    c1_mean = np.mean(gp1[:, 1] + gm1[:, 1]) / 2

    m2_mean = np.mean(gp2[:, 0] - gm2[:, 0]) / 2 / 0.02 - 1
    c2_mean = np.mean(gp2[:, 1] + gm2[:, 1]) / 2

    m3_mean = np.mean(gp3[:, 0] - gm3[:, 0]) / 2 / 0.02 - 1
    c3_mean = np.mean(gp3[:, 1] + gm3[:, 1]) / 2

    m4_mean = np.mean(gp4[:, 0] - gm4[:, 0]) / 2 / 0.02 - 1
    c4_mean = np.mean(gp4[:, 1] + gm4[:, 1]) / 2

    # get std of multiplicative and additive bias for each experiment
    # we need to load error files
    gps1 = load_dataset(INPUT_PATHS["exp70_errs"])["g_plus"]
    gms1 = load_dataset(INPUT_PATHS["exp70_errs"])["g_minus"]
    gps2 = load_dataset(INPUT_PATHS["exp71_errs"])["plus"]["g"]
    gms2 = load_dataset(INPUT_PATHS["exp71_errs"])["minus"]["g"]
    gps3 = load_dataset(INPUT_PATHS["exp72_errs"])["gp"]
    gms3 = load_dataset(INPUT_PATHS["exp72_errs"])["gm"]
    gps4 = load_dataset(INPUT_PATHS["exp73_errs"])["plus"]["g"]
    gms4 = load_dataset(INPUT_PATHS["exp73_errs"])["minus"]["g"]
    assert gps1.ndim == 3 and gms1.ndim == 3
    assert gps2.ndim == 3 and gms2.ndim == 3
    assert gps3.ndim == 3 and gms3.ndim == 3
    assert gps4.ndim == 3 and gms4.ndim == 3

    m1s = (gps1[:, :, 0].mean(1) - gms1[:, :, 0].mean(1)) / 2 / 0.02 - 1
    c1s = (gps1[:, :, 1].mean(1) + gms1[:, :, 1].mean(1)) / 2
    m1_std = m1s.std() / np.sqrt(len(m1s))
    c1_std = c1s.std() / np.sqrt(len(c1s))

    m2s = (gps2[:, :, 0].mean(1) - gms2[:, :, 0].mean(1)) / 2 / 0.02 - 1
    c2s = (gps2[:, :, 1].mean(1) + gms2[:, :, 1].mean(1)) / 2
    m2_std = m2s.std() / np.sqrt(len(m2s))
    c2_std = c2s.std() / np.sqrt(len(c2s))

    m3s = (gps3[:, :, 0].mean(1) - gms3[:, :, 0].mean(1)) / 2 / 0.02 - 1
    c3s = (gps3[:, :, 1].mean(1) + gms3[:, :, 1].mean(1)) / 2
    m3_std = m3s.std() / np.sqrt(len(m3s))
    c3_std = c3s.std() / np.sqrt(len(c3s))

    m4s = (gps4[:, :, 0].mean(1) - gms4[:, :, 0].mean(1)) / 2 / 0.02 - 1
    c4s = (gps4[:, :, 1].mean(1) + gms4[:, :, 1].mean(1)) / 2
    m4_std = m4s.std() / np.sqrt(len(m4s))
    c4_std = c4s.std() / np.sqrt(len(c4s))

    # create table
    # in format for each m, c: mean +- std
    # use f-strings
    table = r"""
\begin{table}[h]
    \centering
    \begin{tabular}{|c|c|c|c|c|}
        \hline
        \textbf{Experiment} & \textbf{Multiplicative Bias $m$} & \textbf{Additive Bias $c$} \\
        \hline
        1 & $%.3g \pm %.3g$ & $%.3g \pm %.3g$ \\
        2 & $%.3g \pm %.3g$ & $%.3g \pm %.3g$ \\
        3 & $%.3g \pm %.3g$ & $%.3g \pm %.3g$ \\
        4 & $%.3g \pm %.3g$ & $%.3g \pm %.3g$ \\
        \hline
    \end{tabular}
    \caption{Multiplicative and additive bias for each experiment.}
    \label{tab:bias}
\end{table}
    """ % (
        m1_mean / 1e-3,
        m1_std / 1e-3,
        c1_mean / 1e-3,
        c1_std / 1e-3,
        m2_mean / 1e-3,
        m2_std / 1e-3,
        c2_mean / 1e-3,
        c2_std / 1e-3,
        m3_mean / 1e-3,
        m3_std / 1e-3,
        c3_mean / 1e-3,
        c3_std / 1e-3,
        m4_mean / 1e-3,
        m4_std / 1e-3,
        c4_mean / 1e-3,
        c4_std / 1e-3,
    )

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(table)


def main(overwrite: bool = False):
    # make_distribution_figure(OUT_PATHS["galaxy_distributions"], overwrite=overwrite)
    make_timing_figure(OUT_PATHS["timing"], OUT_PATHS["timing2"])
    # make_contour_shear_figure1(OUT_PATHS["contour_shear"])
    # make_contour_hyper_figure(OUT_PATHS["contour_hyper"])
    # get_bias_table(OUT_PATHS["bias"])


if __name__ == "__main__":
    typer.run(main)
