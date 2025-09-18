import numpy as np
import pandas as pd
from chainconsumer import Chain, ChainConsumer, Truth
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from numpy import ndarray
from numpyro.diagnostics import hpdi
from scipy import stats


def get_contour_plot(
    samples_list: list[dict[str, ndarray]],
    names: list[str],
    truth: dict[str, float],
    figsize: tuple[float, float] = (7, 7),
    kde: bool | int | float = False,
) -> Figure:
    c = ChainConsumer()
    for name, samples in zip(names, samples_list, strict=False):
        df = pd.DataFrame.from_dict(samples)
        c.add_chain(Chain(samples=df, name=name, kde=kde))
    c.add_truth(Truth(location=truth))
    return c.plotter.plot(figsize=figsize)


def get_gauss_pc_fig(ax: Axes, samples: ndarray, truth: float | ndarray) -> None:
    """Get a marginal pc figure assuming Gaussian distribution of samples."""
    assert samples.ndim == 2  # (n_chains, n_samples)
    cis = np.linspace(0.05, 1, 20)
    cis[-1] = 0.99
    sigmas = stats.norm.interval(cis)[1]
    fractions = []
    for s in sigmas:
        counts = truth < samples.mean(axis=1) + s * samples.std(axis=1)
        counts &= truth > samples.mean(axis=1) - s * samples.std(axis=1)
        fractions.append(counts.mean().item())
    fractions = np.array(fractions)
    ax.plot(cis, fractions, "-", color="C0", lw=2)
    tick_labels = [0, 0.25, 0.5, 0.75, 1.0]
    ax.plot(cis, cis, "--k", label="calibrated")
    ax.set_xticks(tick_labels)
    ax.set_yticks(tick_labels)


def get_pc_fig(
    ax: Axes,
    samples: ndarray,
    truth: float | ndarray,
    n_bins: int = 20,
) -> None:
    """Get a marginal probability calibration figure using `hpdi` from `numpyro`."""
    assert samples.ndim == 2  # (n_chains, n_samples)
    ci_bins = np.linspace(0.05, 1, n_bins)  # confidence intervals
    ci_bins[-1] = 0.99  # prevent weird behavior at 1
    fractions = []
    for c in ci_bins:
        ci1, ci2 = hpdi(samples, prob=c, axis=1).T
        counts = (truth > ci1) & (truth < ci2)
        fractions.append(counts.mean().item())
    fractions = np.array(fractions)
    ax.plot(ci_bins, fractions, "-", color="C0", lw=2)
    tick_labels = [0, 0.25, 0.5, 0.75, 1.0]
    ax.plot(ci_bins, ci_bins, "--k", label="calibrated")
    ax.set_xticks(tick_labels)
    ax.set_yticks(tick_labels)


def get_rank_pc_figs(
    samples: ndarray,
    truth: ndarray,
    *,
    axes,
    n_bins: int = 21,
    n_boots: int = 10_000,
    alpha: float = 0.99,  # confidence interval
):
    # we check the uniformity of ranks in various ways
    assert samples.ndim == 2 and truth.ndim == 1
    assert samples.shape[0] == truth.shape[0]
    ranks = np.sum((samples < truth.reshape(-1, 1)), axis=1)

    _, bins = np.histogram(ranks, bins=n_bins)

    # use simulations to get confidence interval for ranks
    high = samples.shape[-1]
    boot_ranks = np.random.randint(low=0, high=high, size=(n_boots, ranks.shape[0]))

    # get histogram and cdfs for comparison on same bins

    hists = []
    ecdfs = []
    for ii in range(len(boot_ranks)):
        hist, _ = np.histogram(boot_ranks[ii], bins=bins)

        hist_normed, _ = np.histogram(boot_ranks[ii], bins=bins, density=True)
        ecdf = np.cumsum(hist_normed)
        ecdf /= ecdf[-1]

        hists.append(hist)
        ecdfs.append(ecdf)

    hists = np.stack(hists, axis=0)
    ecdfs = np.stack(ecdfs, axis=0)

    low = np.quantile(hists, (1 - alpha) / 2, axis=0)
    med = np.quantile(hists, 0.5, axis=0)
    high = np.quantile(hists, (1 + alpha) / 2, axis=0)

    low_ecdf = np.quantile(ecdfs, (1 - alpha) / 2, axis=0)
    median_ecdf = np.quantile(ecdfs, 0.5, axis=0)
    high_ecdf = np.quantile(ecdfs, (1 + alpha) / 2, axis=0)

    # figure 1
    ax = axes[0]
    ax.hist(ranks, bins=bins, histtype="step")
    ax.axhline(low.mean(), c="k", ls="--")
    ax.axhline(high.mean(), c="k", ls="--")
    ax.axhline(med.mean(), c="r", ls="--")

    # figure 2
    ax = axes[1]
    mids = (bins[1:] + bins[:-1]) / 2
    vals, _, _ = ax.hist(
        ranks, bins=bins, histtype="step", cumulative=True, density=True
    )
    ax.step(mids, low_ecdf, where="mid", ls="--")
    ax.step(mids, high_ecdf, where="mid", ls="--")
    ax.step(mids, median_ecdf, where="mid", ls="--")

    # figure 3
    ax = axes[2]
    ax.plot(mids, vals - median_ecdf)
    ax.fill_between(
        mids, low_ecdf - median_ecdf, high_ecdf - median_ecdf, color="k", alpha=0.1
    )
    ax.axhline(0.0, c="k", ls="--")
