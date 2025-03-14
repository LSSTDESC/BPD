"""Common functions to plot results."""

import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def set_rc_params(
    figsize=(10, 10),
    fontsize=32,
    title_size="large",
    label_size="medium",
    legend_fontsize="medium",
    tick_label_size="small",
    major_tick_size=10,
    minor_tick_size=5,
    major_tick_width=1.0,
    minor_tick_width=0.8,
    lines_marker_size=10,
    legend_loc="best",
):
    # named size options: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    plt.rcParams.update(
        {
            # font
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica",
            "text.latex.preamble": r"\usepackage{amsmath}",
            "mathtext.fontset": "cm",
            "font.size": fontsize,
            # figure
            "figure.figsize": figsize,
            # axes
            "axes.labelsize": label_size,
            "axes.titlesize": title_size,
            # ticks
            "xtick.labelsize": tick_label_size,
            "ytick.labelsize": tick_label_size,
            "xtick.major.size": major_tick_size,
            "ytick.major.size": major_tick_size,
            "xtick.major.width": major_tick_width,
            "ytick.major.width": major_tick_width,
            "ytick.minor.size": minor_tick_size,
            "xtick.minor.size": minor_tick_size,
            "xtick.minor.width": minor_tick_width,
            "ytick.minor.width": minor_tick_width,
            # markers
            "lines.markersize": lines_marker_size,
            # legend
            "legend.fontsize": legend_fontsize,
            "legend.loc": legend_loc,
            # colors
            "axes.prop_cycle": mpl.cycler(color=CB_color_cycle),
            # images
            "image.cmap": "gray",
            "figure.autolayout": True,
        }
    )


def get_timing_figure(results: dict, max_n_gal: str) -> Figure:
    all_n_gals = [n_gals for n_gals in results]

    # cycler from blue to red
    color = plt.cm.coolwarm(np.linspace(0, 1, len(all_n_gals)))
    cycles = cycler.cycler("color", color)

    t_per_obj_dict = {}
    n_samples_array = np.arange(0, 1001, 1)

    _, n_chains_per_gal, n_samples = results[max_n_gal]["samples"]["lf"].shape

    for n_gals_str in all_n_gals:
        t_warmup = results[n_gals_str]["t_warmup"]
        t_sampling = results[n_gals_str]["t_sampling"]

        n_gals = int(n_gals_str)

        n_chains = n_gals * n_chains_per_gal

        t_per_obj_warmup = t_warmup / n_chains
        t_per_obj_per_sample_sampling = t_sampling / (n_chains * n_samples)
        t_per_obj_arr = (
            t_per_obj_warmup + t_per_obj_per_sample_sampling * n_samples_array
        )
        t_per_obj_dict[n_gals] = t_per_obj_arr

    fig, ax = plt.subplots(1, 1)
    ax.set_prop_cycle(cycles)

    ax.set_ylabel(r"\rm Time per galaxy in a single A100 GPU (sec)")
    ax.set_xlabel(r"\rm \# of samples")

    for n_gals, t_per_obj_array in t_per_obj_dict.items():
        n_chains = 4 * n_gals
        ax.plot(n_samples_array, t_per_obj_array, label=f"${n_chains}$")

    plt.legend(title=r"\rm Number of chains", loc="best")

    return fig


def get_jack_bias(
    g_plus_jack: np.ndarray, g_minus_jack: np.ndarray, g1_true: float
) -> tuple:
    assert g_plus_jack.ndim == 3 and g_minus_jack.ndim == 3
    assert g_plus_jack.shape[-1] == 2 and g_minus_jack.shape[-1] == 2
    assert g1_true > 0
    n_jack = g_plus_jack.shape[0]

    m_jack = (
        g_plus_jack[..., 0].mean(axis=1) - g_minus_jack[..., 0].mean(axis=1)
    ) / 2 / g1_true - 1
    c_jack = (g_plus_jack[..., 1].mean(axis=1) + g_minus_jack[..., 1].mean(axis=1)) / 2

    m_mean = m_jack.mean().item()
    m_std = np.sqrt(m_jack.var() * (n_jack - 1)).item()

    c_mean = c_jack.mean().item()
    c_std = np.sqrt(c_jack.var() * (n_jack - 1)).item()

    return m_mean, m_std, c_mean, c_std
