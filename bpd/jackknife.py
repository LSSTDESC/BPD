from math import ceil
from typing import Callable

import jax.numpy as jnp
from jax import Array, random
from tqdm import tqdm


def run_jackknife_shear_pipeline(
    rng_key,
    init_g: Array,
    post_params_pos: dict,
    post_params_neg: dict,
    shear_pipeline: Callable,
    n_jacks: int = 10,
    disable_bar: bool = True,
):
    """Use jackknife+shape noise cancellation to estimate the mean and std of the shear posterior.

    Args:
        rng_key: Random jax key.
        init_g: Initial value for shear `g`.
        post_params_pos: Interim posterior galaxy parameters estimated using positive shear.
        post_params_neg: Interim posterior galaxy parameters estimated using negative shear,
            and otherwise same conditions and random seed as `post_params_pos`.
        shear_pipeline: Function that outputs shear posterior samples from `post_params` with all
            keyword arguments pre-specified.
        n_jacks: Number of jackknife batches.

    Returns:
        Jackknife

    """
    N, _ = post_params_pos["e1"].shape  # N = n_gals, K = n_samples_per_gal
    batch_size = ceil(N / n_jacks)

    g_best_list = []
    keys = random.split(rng_key, n_jacks)

    for ii in tqdm(range(n_jacks), desc="Jackknife #", disable=disable_bar):
        k_ii = keys[ii]
        start, end = ii * batch_size, (ii + 1) * batch_size

        _params_jack_pos = {
            k: jnp.concatenate([v[:start], v[end:]]) for k, v in post_params_pos.items()
        }
        _params_jack_neg = {
            k: jnp.concatenate([v[:start], v[end:]]) for k, v in post_params_neg.items()
        }

        g_pos_ii = shear_pipeline(k_ii, _params_jack_pos, init_g)
        g_neg_ii = shear_pipeline(k_ii, _params_jack_neg, -init_g)
        g_best_ii = (g_pos_ii - g_neg_ii) * 0.5
        g_best_mean_ii = g_best_ii.mean(axis=0)

        g_best_list.append(g_best_mean_ii)

    g_best_means = jnp.array(g_best_list)
    return g_best_means
