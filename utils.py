import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from numpyro.diagnostics import hpdi, summary, print_summary

from matplotlib.pyplot import Axes


def get_gauss_pc_fig(ax:Axes, samples:np.ndarray, truth:float, param_name:str=None) -> None:
    """Get a probability calibration figure assuming Gaussian calibration"""
    assert samples.ndim == 2 # (n_chains, n_samples)
    cis = np.linspace(0.05, 1, 20)
    cis[-1] = 0.99
    sigmas = stats.norm.interval(cis)[1]
    fractions = []
    for s in sigmas:
        counts = truth < samples.mean(axis=1) + s * samples.std(axis=1)
        counts &= truth > samples.mean(axis=1) - s * samples.std(axis=1)
        fractions.append(counts.mean().item())
    fractions = np.array(fractions)
    ax.plot(cis, fractions, "-x", color='C0')
    tick_labels = [0, 0.25, 0.5, 0.75, 1.0]
    ax.plot(cis, cis, "--k", label="calibrated")
    ax.legend(loc="best", prop={"size": 14})
    ax.set_xlabel(r"Target coverage", fontsize=14) 
    ax.set_ylabel(r"Realized coverage", fontsize=14)
    ax.set_xticks(tick_labels)
    ax.set_yticks(tick_labels)
    ax.set_title(param_name, fontsize=14)
    
def get_pc_fig(ax:Axes, samples:np.ndarray, truth:float, param_name:str=None) -> None:
    """Get a probability calibration figure assuming Gaussian calibration"""
    assert samples.ndim == 2 # (n_chains, n_samples)
    ci_bins = np.linspace(0.05, 1, 20) # confidence intervals
    ci_bins[-1] = 0.99 # prevent weird behavior with 1
    fractions = []
    for c in ci_bins:
        ci1, ci2 = hpdi(samples, prob=c, axis=1).T
        counts = (truth > ci1) & (truth < ci2)
        fractions.append(counts.mean().item())
    fractions = np.array(fractions)
    ax.plot(ci_bins, fractions, "-x", color='C0')
    tick_labels = [0, 0.25, 0.5, 0.75, 1.0]
    ax.plot(ci_bins, ci_bins, "--k", label="calibrated")
    ax.legend(loc="best", prop={"size": 14})
    ax.set_xlabel(r"Target coverage", fontsize=14) 
    ax.set_ylabel(r"Realized coverage", fontsize=14)
    ax.set_xticks(tick_labels)
    ax.set_yticks(tick_labels)
    ax.set_title(param_name, fontsize=14)
    
def run_chains(data, kernel, rng_key, n_vec=1, num_warmup=100, num_samples=1000):
    """Run chains on subsets of data as iid samples and collect samples.
    
    This function is particularly useful when we want to run on multiple noise
    realizations of the same galaxy image.
    """
    n = len(data)
    all_samples = {}
    for ii in range(0, n, n_vec):
        data_ii = data[ii:ii+n_vec]
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(rng_key, data=data_ii)
        samples = mcmc.get_samples()
        for k in samples: 
            v2 = samples.get(k)
            assert v2.shape == (num_samples, n_vec)
            if k not in all_samples: 
                all_samples[k] = v2
            else:
                v1 = all_samples[k]
                assert v1.shape[0] == num_samples
                assert v1.ndim == 2
                all_samples[k] = jnp.concatenate([v1,v2], axis=-1)
    all_samples = {k:v.T for k in all_samples}
    
    for k,v in all_samples.items(): 
        assert v.shape == (n, num_samples)
        
    return all_samples # (n_chains, n_samples)


# TODO: Need to modify, the above are not actually `n_chains` as they are run on different data i.e. noise realizations.
# could compute the summary statistics `per-chain` but combinbing them does not make sense.
def get_summary_from_samples(samples: dict, prob=0.9):
    """Given output of `run_chains`, get summary stats for each chain.
    
    Includes: 'mean', 'std', 'median', 'X - prob.0%', 'X + prob.0%', 'n_eff', 'r_hat'.
    """ 
    for k,v in samples.items(): 
        assert v.ndim == 2 # (n_chains, n_samples)
    
    d = {k:v[...,None] for k,v in samples.items()}
    return summary(d, prob=prob, group_by_chain=True) # assumes (n_chains, n_samples, sample_shape)


def print_summary_from_samples(samples: dict, prob=0.9):
    """Given output of `run_chains`, get summary stats for each chain.
    
    Includes: 'mean', 'std', 'median', '5.0%', '95.0%', 'n_eff', 'r_hat'.
    
    """ 
    for k,v in samples.items(): 
        assert v.ndim == 2 # (n_chains, n_samples)
    
    d = {k:v[..., None] for k,v in samples.items()}
    return print_summary(d, prob=prob, group_by_chain=True)