import numpy as np
from matplotlib.pyplot import Axes
from numpyro.diagnostics import hpdi, summary
from scipy import stats


def my_gelman_rubin(x):
    # from https://bookdown.org/rdpeng/advstatcomp/monitoring-convergence.html
    assert x.ndim == 2  # (n_chains, n_samples)
    assert x.shape[0] >= 2
    assert x.shape[1] >= 2

    J = x.shape[0]  # noqa: N806
    L = x.shape[1]  # noqa: N806

    chain_mean = x.mean(axis=1)
    grand_mean = chain_mean.mean()

    # between chain variance
    B = (L / (J - 1)) * np.sum((chain_mean - grand_mean) ** 2)  # noqa: N806

    # within chain variance
    sj_sq = (1 / (L - 1)) * np.sum((x - chain_mean.reshape(-1, 1)) ** 2, axis=1)
    W = (1 / J) * np.sum(sj_sq, axis=0)  # noqa: N806

    return ((L - 1) / L * W + B / L) / W


def get_gauss_pc_fig(
    ax: Axes, samples: np.ndarray, truth: float, param_name: str = None
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
    ax: Axes, samples: np.ndarray, truth: float, param_name: str = None
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


def get_summary_from_samples(samples: dict, prob=0.9):
    """Given output of `run_chains`, get summary stats for each chain.

    Assumes each chain was run on a different noise realization of the same data.
    So summaries are kept separate.

    Includes: 'mean', 'std', 'median', 'X - prob.0%', 'X + prob.0%', 'n_eff', 'r_hat'.
    """
    for _, v in samples.items():
        assert v.ndim == 2  # (n_chains, n_samples)

    # compute summary per chain, per parameter
    full_summary = {k: {} for k in samples}
    for k, v in samples.items():
        n_chains = v.shape[0]
        for ii in range(n_chains):
            v_ii = v[ii, None, :, None]
            summary_v_ii = summary({k: v_ii}, prob=prob, group_by_chain=True)
            for k in summary_v_ii:
                for kk in summary_v_ii[k]:
                    if kk not in full_summary[k]:
                        full_summary[k][kk] = []
                    full_summary[k][kk].append(summary_v_ii[k][kk])

    # convert to numpy arrays
    for k in full_summary:
        for kk in full_summary[k]:
            full_summary[k][kk] = np.array(full_summary[k][kk])

    return full_summary
