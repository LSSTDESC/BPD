#!/usr/bin/env python3
import os

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
from bpd.diagnostics import get_pc_fig
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
    "exp70_sp": CHAIN_DIR / "exp70_51" / "g_samples_512_plus.npz",
    "exp70_sm": CHAIN_DIR / "exp70_51" / "g_samples_512_minus.npz",
    "exp70_errs": CHAIN_DIR / "exp70_51" / "g_samples_514_errs.npz",
    "exp71_sp": CHAIN_DIR / "exp71_51" / "shear_samples_512_plus.npz",
    "exp71_sm": CHAIN_DIR / "exp71_51" / "shear_samples_512_minus.npz",
    "exp71_errs": CHAIN_DIR / "exp71_51" / "g_samples_514_errs.npz",
    "exp72_sp": CHAIN_DIR / "exp72_51" / "g_samples_512_plus.npz",
    "exp72_sm": CHAIN_DIR / "exp72_51" / "g_samples_512_minus.npz",
    "exp72_errs": CHAIN_DIR / "exp72_51" / "g_samples_514_errs.npz",
    "exp73_sp": CHAIN_DIR / "exp73_51" / "shear_samples_512_plus.npz",
    "exp73_sm": CHAIN_DIR / "exp73_51" / "shear_samples_512_minus.npz",
    "exp73_errs": CHAIN_DIR / "exp73_51" / "g_samples_514_errs_514.npz",
    "exp72_interim_samples": CHAIN_DIR / "exp72_51" / "interim_samples_511_plus.npz",
    "eta_pc": CHAIN_DIR / "exp81_52" / "eta_shear_samples.npz",
}


OUT_PATHS = {
    "galaxy_distributions": FIG_DIR / "gprop_dists.png",
    "timing": FIG_DIR / "timing.png",
    "timing2": FIG_DIR / "timing2.png",
    "error_bar": FIG_DIR / "error_bar.png",
    "contour_shear": FIG_DIR / "contour_shear.png",
    "contour_hyper": FIG_DIR / "contour_hyper.png",
    "bias": FIG_DIR / "table_bias.txt",
    "eta_pc": FIG_DIR / "eta_pc.png",
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
        label=r"\rm Median = {:.2f}".format(np.median(params["snr"])),
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
    ax3.set_xlabel(r"\rm $s$ (HLR)")
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
    data1 = load_dataset(INPUT_PATHS["exp70_sp"])["samples"]
    data3 = load_dataset(INPUT_PATHS["exp72_sp"])["samples"]
    c = ChainConsumer()
    assert "g1" in data1 and "g2" in data1 and "g1" in data3 and "g2" in data3
    df1 = pandas.DataFrame.from_dict(data1)
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


def make_contour_shear_figure(fpath: str | Path):
    set_rc_params()  # reset to default fontsize
    data3 = load_dataset(INPUT_PATHS["exp72_sp"])
    data4 = load_dataset(INPUT_PATHS["exp73_sp"])
    g_exp72 = jnp.stack([data3["samples"]["g1"], data3["samples"]["g2"]], axis=-1)
    g_exp73 = jnp.stack([data4["samples"]["g1"], data4["samples"]["g2"]], axis=-1)
    ds_int = load_dataset(INPUT_PATHS["exp72_interim_samples"])
    e1_mean = ds_int["truth"]["e1"].mean()
    e2_mean = ds_int["truth"]["e2"].mean()
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
    c.set_override(ChainConfig(sigmas=[0, 1, 2], kde=1.0))
    c.add_truth(
        Truth(
            location={"g1": 0.02 + e1_mean, "g2": 0.0 + e2_mean},
            color="k",
            line_width=2.0,
        )
    )

    # now the other too
    data2 = {"g1": g_exp73[:, 0], "g2": g_exp73[:, 1]}
    df2 = pandas.DataFrame.from_dict(data2)
    chain2 = Chain(
        samples=df2,
        name=r"\texttt{all-free}",
        marker_style="*",
    )
    c.add_chain(chain2)
    c.set_override(ChainConfig(sigmas=[0, 1, 2], kde=1.0))

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
    c.set_override(ChainConfig(sigmas=[0, 1, 2], kde=1.0))
    c.add_truth(Truth(location=truth, color="k", line_width=4.0))

    c.set_plot_config(
        PlotConfig(
            usetex=True,
            plot_hists=False,
            labels={
                "sigma_e": r"$\sigma_{e}$",
                "a_logflux": r"$a_{f}$",
                "mean_logflux": r"$\mu_{f}$",
                "sigma_logflux": r"$\sigma_{f}$",
                "mean_loghlr": r"$\mu_{s}$",
                "sigma_loghlr": r"$\sigma_{s}$",
            },
            label_font_size=60,
            summarise=False,
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
    dsp1 = load_dataset(INPUT_PATHS["exp70_sp"])
    dsm1 = load_dataset(INPUT_PATHS["exp70_sm"])
    gp1 = jnp.stack([dsp1["samples"]["g1"], dsp1["samples"]["g2"]], axis=-1)
    gm1 = jnp.stack([dsm1["samples"]["g1"], dsm1["samples"]["g2"]], axis=-1)

    dsp2 = load_dataset(INPUT_PATHS["exp71_sp"])
    dsm2 = load_dataset(INPUT_PATHS["exp71_sm"])
    gp2 = jnp.stack([dsp2["samples"]["g1"], dsp2["samples"]["g2"]], axis=-1)
    gm2 = jnp.stack([dsm2["samples"]["g1"], dsm2["samples"]["g2"]], axis=-1)

    dsp3 = load_dataset(INPUT_PATHS["exp72_sp"])
    dsm3 = load_dataset(INPUT_PATHS["exp72_sm"])
    gp3 = jnp.stack([dsp3["samples"]["g1"], dsp3["samples"]["g2"]], axis=-1)
    gm3 = jnp.stack([dsm3["samples"]["g1"], dsm3["samples"]["g2"]], axis=-1)

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
    gps = []
    gms = []
    for n in ("70", "71", "72", "73"):
        gp1s = load_dataset(INPUT_PATHS[f"exp{n}_errs"])["plus"]["g1"]
        gp2s = load_dataset(INPUT_PATHS[f"exp{n}_errs"])["plus"]["g2"]
        gm1s = load_dataset(INPUT_PATHS[f"exp{n}_errs"])["minus"]["g1"]
        gm2s = load_dataset(INPUT_PATHS[f"exp{n}_errs"])["minus"]["g2"]
        gp = np.stack([gp1s, gp2s], axis=-1)
        gm = np.stack([gm1s, gm2s], axis=-1)
        gps.append(gp)
        gms.append(gm)

    gps1, gps2, gps3, gps4 = gps
    gms1, gms2, gms3, gms4 = gms
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
\begin{table*}
    \centering
    \begin{tabular}{|c|c|c|c|c|c|}
        \hline
        \textbf{Experiment} & \textbf{Properties Used} & \textbf{Prior Known?} & \textbf{Multiplicative Bias $m / 10^{-3}$} & \textbf{Additive Bias $c / 10^{-3}$} \\
        \hline
        \texttt{shapes-fixed} & Ellipticities & Yes & $%.3g \pm %.3g$ & $%.3g \pm %.3f$ \\
        \texttt{shapes-free} & Ellipticities & No & $%.3g \pm %.3g$ & $%.3g \pm %.3f$ \\
        \texttt{all-fixed} & All & Yes & $%.3g \pm %.3g$ & $%.3g \pm %.3f$ \\
        \texttt{all-free} & All & No & $%.3g \pm %.3g$ & $%.3g \pm %.3f$ \\
        \hline
    \end{tabular}
    \caption{\textbf{Multiplicative and additive bias for different settings.}}
    \label{tab:bias}
\end{table*}
    """ % (
        m1_mean / 1e-3,
        3 * m1_std / 1e-3,
        c1_mean / 1e-3,
        3 * c1_std / 1e-3,
        m2_mean / 1e-3,
        3 * m2_std / 1e-3,
        c2_mean / 1e-3,
        3 * c2_std / 1e-3,
        m3_mean / 1e-3,
        3 * m3_std / 1e-3,
        c3_mean / 1e-3,
        3 * c3_std / 1e-3,
        m4_mean / 1e-3,
        3 * m4_std / 1e-3,
        c4_mean / 1e-3,
        3 * c4_std / 1e-3,
    )

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(table)


def make_eta_posterior_calibration_figure(fpath: str | Path):
    set_rc_params(fontsize=24)

    ds = load_dataset(INPUT_PATHS["eta_pc"])
    g_samples = ds["g_samples"]
    true_gs = ds["true_g"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    g1_samples = g_samples[:, :, 0]
    g1_truth = true_gs[:, 0]
    get_pc_fig(ax1, g1_samples, g1_truth)
    ax1.set_ylabel(r"\rm Realized Coverage")
    ax1.set_xlabel(r"\rm Target Coverage")
    ax1.legend(loc="best")
    ax1.set_title(r"\rm $g_{1}$")

    g2_samples = g_samples[:, :, 1]
    g2_truth = true_gs[:, 1]
    ax2.set_xlabel(r"\rm Target Coverage")
    ax2.set_title(r"\rm $g_{2}$")
    get_pc_fig(ax2, g2_samples, g2_truth)

    fig.tight_layout()
    fig.savefig(fpath, format="png")


def main(overwrite: bool = False):
    # make_timing_figure(OUT_PATHS["timing"], OUT_PATHS["timing2"])
    # make_distribution_figure(OUT_PATHS["galaxy_distributions"], overwrite=overwrite)
    # make_contour_shear_figure(OUT_PATHS["contour_shear"])
    make_contour_hyper_figure(OUT_PATHS["contour_hyper"])
    # make_eta_posterior_calibration_figure(OUT_PATHS["eta_pc"])
    # get_bias_table(OUT_PATHS["bias"])


if __name__ == "__main__":
    typer.run(main)
