from functools import partial
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import uniform
from jax.tree_util import tree_map
from jax.typing import ArrayLike
from tqdm import tqdm

DEFAULT_HYPERPARAMS = {
    "shape_noise": 0.2,
    "a_logflux": 14.0,
    "mean_logflux": 2.45,
    "sigma_logflux": 0.4,
    "mean_loghlr": -0.4,
    "sigma_loghlr": 0.05,
}

MAX_N_GALS_PER_GPU = 5000


def get_snr(im: ArrayLike, background: float) -> float:
    """Calculate the signal-to-noise ratio of an image.

    Args:
        im: 2D image array with no background.
        background: Background level.

    Returns:
        float: The signal-to-noise ratio.
    """
    assert im.ndim == 2
    assert isinstance(background, float) or background.shape == ()
    return jnp.sqrt(jnp.sum(im * im / (background + im)))


def uniform_logpdf(x: ArrayLike, a: float, b: float):
    return uniform.logpdf(x, loc=a, scale=b - a)


def process_in_batches(
    fnc: Callable, *args, n_points: int, batch_size: int, no_bar: bool = False
):
    """Process a function in batches to avoid memory issues. Works with dicts and array outputs.

    Args:
        fnc: Function to be applied to the arguments.
        *args: Arguments to the function (Arrays or dict of Arrays).
        n_points: Size of the dataset (e.g. number of galaxies).
        batch_size: Size of each batch.
    """
    assert callable(fnc)
    assert len(args) > 0
    assert n_points > 0
    assert batch_size > 0
    assert n_points % batch_size == 0
    for x in args:
        if isinstance(x, dict):
            for _, v in x.items():
                assert v.shape[0] == n_points
        elif isinstance(x, ArrayLike):
            assert x.shape[0] == n_points
        else:
            raise ValueError("Arguments must be dicts or arrays.")

    n_batches = n_points // batch_size
    assert batch_size * n_batches == n_points
    assert n_batches > 0

    results = []

    for ii in tqdm(range(n_batches), disable=no_bar, desc="Processing batches"):
        start = ii * batch_size
        end = (ii + 1) * batch_size
        _slice_fnc = lambda x, start, end: x[start:end]
        slice_fnc = partial(_slice_fnc, start=start, end=end)
        args_batch = tree_map(slice_fnc, args)
        results.append(fnc(*args_batch))

    return tree_map(lambda *x: jnp.concatenate(x, axis=0), *results)


def combine_subposts_gaussian(sub_samples: ArrayLike, n_dim: int = 2):
    # assume subposteriors and final full posterior approximates a Gaussian distribution
    # (Bernstein-Von Mises Theorem) and return mean and Sigma of full posterior.
    assert sub_samples.ndim == 3
    assert sub_samples.shape[2] == n_dim
    n_posts = sub_samples.shape[0]
    ss = sub_samples

    # compute mean of each subposterior
    mus = ss.mean(axis=1)

    # compute covariances of each subposterior
    covs = []
    for ii in range(n_posts):
        samples_ii = ss[ii]
        cov_ii = np.cov(samples_ii, rowvar=False)
        covs.append(cov_ii)
    covs = np.stack(covs, axis=0)
    assert covs.shape == (n_posts, n_dim, n_dim)

    # get full covariance
    fcov = 0.0
    for ii in range(n_posts):
        fcov += np.linalg.inv(covs[ii])
    fcov = np.linalg.inv(fcov)

    # get full mean
    fmu = 0.0
    for ii in range(n_posts):
        mu_ii = mus[ii].reshape(n_dim, 1)
        fmu += np.linalg.inv(covs[ii]).dot(mu_ii)

    assert fmu.shape == (n_dim, 1)
    fmu = fcov.dot(fmu)

    return fmu, fcov
