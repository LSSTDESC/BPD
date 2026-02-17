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
    legend_fontsize="small",
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


def get_timing_figure(
    results: dict, *, max_n_gal_str: str, avg_ess: float, figsize=(10, 10)
) -> Figure:
    all_n_gals = [n_gals for n_gals in results]

    # cycler from blue to red
    color = plt.cm.coolwarm(np.linspace(0, 1, len(all_n_gals)))
    cycles = cycler.cycler("color", color)

    t_per_obj_dict = {}
    n_samples_array = np.arange(0, 1001, 1)

    _, n_samples = results[max_n_gal_str]["samples"]["lf"].shape

    for n_gals_str in all_n_gals:
        t_warmup = results[n_gals_str]["t_warmup"]
        t_sampling = results[n_gals_str]["t_sampling"]

        n_chains = int(n_gals_str)  # new fmt

        t_per_obj_warmup = t_warmup / n_chains
        t_per_obj_per_sample_sampling = t_sampling / (n_chains * n_samples)
        t_per_obj_arr = (
            t_per_obj_warmup + t_per_obj_per_sample_sampling * n_samples_array
        )
        t_per_obj_dict[n_chains] = t_per_obj_arr / avg_ess

        if n_gals_str == max_n_gal_str:
            print(
                f"Global best efficiency: {t_per_obj_per_sample_sampling / avg_ess:.2g} sec"
            )
            print(f"Global best warmup: {t_per_obj_warmup:.2g} sec")

    # first option
    fig1, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_prop_cycle(cycles)

    ax.set_ylabel(r"\rm Galaxies processed per second in one A100 GPU")
    ax.set_xlabel(r"\rm \# of effective samples")

    for n_chains, t_per_obj_array in t_per_obj_dict.items():
        ax.plot(n_samples_array, 1 / t_per_obj_array, label=f"${n_chains}$")

    ax.legend(
        title=r"\rm Galaxies Sampled in Parallel",
        loc="upper right",
        ncol=4,
        fancybox=True,
        shadow=False,
    )

    fig2, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_prop_cycle(cycles)

    ax.set_ylabel(r"\rm Time to process one galaxy in one A100 GPU (sec)")
    ax.set_xlabel(r"\rm \# of effective samples")

    for n_chains, t_per_obj_array in t_per_obj_dict.items():
        ax.plot(n_samples_array, t_per_obj_array, label=f"${n_chains}$")

    ax.legend(
        title=r"\rm Galaxies Sampled in Parallel",
        loc="upper left",
        ncol=4,
        fancybox=True,
        shadow=False,
    )

    return fig1, fig2


def get_total_timing_figure(
    results: dict, *, max_n_gal_str: str, avg_ess: float, figsize=(10, 10)
) -> Figure:
    all_n_gals = [n_gals for n_gals in results]

    _, n_samples = results[max_n_gal_str]["samples"]["lf"].shape

    total_time_warmup = []
    total_time_sampling = []
    n_chains_arr = np.array([int(n_gals) for n_gals in results])

    for n_gals_str in all_n_gals:
        t_warmup = results[n_gals_str]["t_warmup"]
        t_sampling = results[n_gals_str]["t_sampling"] / n_samples * 300 / avg_ess

        total_time_warmup.append(t_warmup)
        total_time_sampling.append(t_sampling)

    total_time_warmup = np.array(total_time_warmup)
    total_time_sampling = np.array(total_time_sampling)
    total_time = total_time_sampling + total_time_warmup

    # first option
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_ylabel(r"\rm Total time (sec)")
    ax.set_xlabel(r"\rm \# of Galaxies")

    ax.plot(n_chains_arr, total_time_warmup, "-o", label=r"\rm Warmup")
    ax.plot(n_chains_arr, total_time_sampling, "-o", label=r"\rm Inference")
    ax.plot(n_chains_arr, total_time, "-o", label=r"\rm Total")

    ax.plot(n_chains_arr, total_time[0] * n_chains_arr, "k--", label=r"\rm Worst")

    ax.legend(loc="best", fancybox=True, shadow=False)

    ax.set_xscale("log")
    ax.set_yscale("log")

    return fig


def get_timing_table(
    results: dict, *, max_n_gal_str: str, avg_ess: float, fpath: str
) -> Figure:
    all_n_gals = [n_gals for n_gals in results]
    warmup_times_per_obj = {}
    inference_times = {}
    t_300_dict = {}  # after warmup

    _, n_samples = results[max_n_gal_str]["samples"]["lf"].shape

    for n_gals_str in all_n_gals:
        t_warmup = results[n_gals_str]["t_warmup"]
        t_sampling = results[n_gals_str]["t_sampling"]

        n_chains = int(n_gals_str)  # new fmt

        # (avg.) time to warmup 1 object
        t_per_obj_warmup = t_warmup / n_chains

        # (avg.) time to produce 1 effective sample for 1 object (ignoring warmup)
        t_per_obj_per_sample_sampling = t_sampling / (n_chains * n_samples) / avg_ess
        t_300 = t_per_obj_per_sample_sampling * 300 + t_per_obj_warmup

        # save
        warmup_times_per_obj[n_chains] = t_per_obj_warmup
        inference_times[n_chains] = t_per_obj_per_sample_sampling
        t_300_dict[n_chains] = t_300

        if n_gals_str == max_n_gal_str:
            print(f"Global best efficiency: {t_per_obj_per_sample_sampling:.3g} sec")
            print(f"Global best warmup: {t_per_obj_warmup:.3g} sec")

    # create latex table with rows for n_chains and columns for t_per_obj_warmup, t_per_obj_per_sample_sampling,
    # and eff_samples_per_sec
    table_str = "\\begin{tabular}{|c|c|c|c|}\n"
    table_str += "\\hline\n"
    table_str += "\\# of Galaxies \\newline in Parallel & Warmup time (sec) & Inference time / eff. sample (sec) & Time to produce \\newline 300 eff. samples (sec)\\\\\n"
    table_str += "\\hline\n"
    for n_chains in sorted(t_300_dict.keys()):
        t_per_obj_warmup = warmup_times_per_obj[n_chains]
        table_str += f"{n_chains} & {t_per_obj_warmup:.2g} & {inference_times[n_chains]:.2g} & {t_300_dict[n_chains]:.2g} \\\\\n"
    table_str += "\\hline\n"
    table_str += "\\end{tabular}"

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(table_str)


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
