"""Functions for initializing MCMC sampling algorithms."""

from typing import Callable

from jax import Array, random
from jax._src.prng import PRNGKeyArray


def init_with_truth(
    rng_key: PRNGKeyArray, true_params: dict[str, float], *, data: Array = None
):
    """Seems useless but we it can actually be helpful when vmapping this function."""
    new = {}
    for p in true_params:
        new[p] = true_params[p]
    return new


def init_with_ball(
    rng_key: PRNGKeyArray,
    true_params: dict[str, float],
    *,
    offset_dict: dict[str, float],
    data: Array = None,
):
    """Sample ball given offset of each parameter."""
    new = {}
    keys = random.split(rng_key, len(true_params.keys()))
    rng_key_dict = {p: k for p, k in zip(true_params, keys, strict=False)}

    for p, centr in true_params.items():
        offset = offset_dict[p]
        new[p] = random.uniform(
            rng_key_dict[p], shape=(), minval=centr - offset, maxval=centr + offset
        )
    return new


def init_with_prior(
    rng_key: PRNGKeyArray,
    true_params: dict[str, float],
    *,
    prior: Callable,
    data: Array = None,
):
    """Sample ball given offset of each parameter."""
    return prior(rng_key)
