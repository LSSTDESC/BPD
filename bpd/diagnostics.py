import numpy as np
import pandas as pd
from chainconsumer import Chain, ChainConsumer, Truth
from jax import Array
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from numpyro.diagnostics import hpdi
from scipy import stats


def get_contour_plot(
    samples_list: list[dict[str, Array]],
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


def get_gauss_pc_fig(
    ax: Axes, samples: np.ndarray, truth: float, param_name: str | None = None
) -> None:
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
    ax.plot(cis, fractions, "-x", color="C0")
    tick_labels = [0, 0.25, 0.5, 0.75, 1.0]
    ax.plot(cis, cis, "--k", label="calibrated")
    ax.legend(loc="best", prop={"size": 14})
    ax.set_xlabel(r"Target coverage", fontsize=14)
    ax.set_ylabel(r"Realized coverage", fontsize=14)
    ax.set_xticks(tick_labels)
    ax.set_yticks(tick_labels)
    ax.set_title(param_name, fontsize=14)


def get_pc_fig(
    ax: Axes, samples: np.ndarray, truth: float, param_name: str | None = None
) -> None:
    """Get a marginal probability calibration figure using `hpdi` from `numpyro`."""
    assert samples.ndim == 2  # (n_chains, n_samples)
    ci_bins = np.linspace(0.05, 1, 20)  # confidence intervals
    ci_bins[-1] = 0.99  # prevent weird behavior at 1
    fractions = []
    for c in ci_bins:
        ci1, ci2 = hpdi(samples, prob=c, axis=1).T
        counts = (truth > ci1) & (truth < ci2)
        fractions.append(counts.mean().item())
    fractions = np.array(fractions)
    ax.plot(ci_bins, fractions, "-x", color="C0")
    tick_labels = [0, 0.25, 0.5, 0.75, 1.0]
    ax.plot(ci_bins, ci_bins, "--k", label="calibrated")
    ax.legend(loc="best", prop={"size": 14})
    ax.set_xlabel(r"Target coverage", fontsize=14)
    ax.set_ylabel(r"Realized coverage", fontsize=14)
    ax.set_xticks(tick_labels)
    ax.set_yticks(tick_labels)
    ax.set_title(param_name, fontsize=14)
