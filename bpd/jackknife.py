from math import ceil
from typing import Callable

import jax.numpy as jnp
from jax import Array, jit, random, vmap
from tqdm import tqdm


def run_jackknife_shear_pipeline(
    rng_key,
    *,
    init_g: Array,
    post_params_pos: dict,
    post_params_neg: dict,
    shear_pipeline: Callable,
    n_gals: int,
    n_jacks: int = 100,
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
        Jackknife samples of shear posterior mean combined with shape noise cancellation trick.
    """
    batch_size = ceil(n_gals / n_jacks)

    results_plus = []
    results_minus = []
    keys = random.split(rng_key, n_jacks)

    pipe = jit(shear_pipeline)

    for ii in tqdm(range(n_jacks), desc="Jackknife #", disable=disable_bar):
        k_ii = keys[ii]
        start, end = ii * batch_size, (ii + 1) * batch_size

        _params_jack_pos = {
            k: jnp.concatenate([v[:start], v[end:]]) for k, v in post_params_pos.items()
        }
        _params_jack_neg = {
            k: jnp.concatenate([v[:start], v[end:]]) for k, v in post_params_neg.items()
        }

        g_pos_ii = pipe(k_ii, _params_jack_pos, init_g)
        g_neg_ii = pipe(k_ii, _params_jack_neg, -init_g)

        results_plus.append(g_pos_ii)
        results_minus.append(g_neg_ii)

    g_pos_samples = jnp.stack(results_plus, axis=0)
    g_neg_samples = jnp.stack(results_minus, axis=0)
    assert g_pos_samples.shape[0] == n_jacks and g_pos_samples.shape[-1] == 2
    assert g_neg_samples.shape[0] == n_jacks and g_neg_samples.shape[-1] == 2

    return g_pos_samples, g_neg_samples


def run_jackknife_vectorized(
    rng_key,
    *,
    init_g: Array,
    post_params_pos: dict,
    post_params_neg: dict,
    shear_pipeline: Callable,
    n_gals: int,
    n_jacks: int = 100,
    n_splits: int = 2,
):
    """Same as previous function, but vectorized for speed."""
    assert n_gals % n_jacks == 0, "# of galaxies needs to be divisible by # jackknives."
    batch_size = int(n_gals / n_jacks)
    keys = random.split(rng_key, n_jacks)

    # prepare dictionaries of jackknife samples
    params_jack_pos = {}
    params_jack_neg = {}
    for k in post_params_pos:
        v1 = post_params_pos[k]
        v2 = post_params_neg[k]
        all_jack_params_pos = []
        all_jack_params_neg = []
        for ii in range(n_jacks):
            start, end = ii * batch_size, (ii + 1) * batch_size
            all_jack_params_pos.append(jnp.concatenate([v1[:start], v1[end:]]))
            all_jack_params_neg.append(jnp.concatenate([v2[:start], v2[end:]]))

        params_jack_pos[k] = jnp.stack(all_jack_params_pos, axis=0)
        params_jack_neg[k] = jnp.stack(all_jack_params_neg, axis=0)

        assert params_jack_pos[k].shape[0] == n_jacks
        assert params_jack_neg[k].shape[0] == n_jacks

    # run on a single example for compilation purposes
    vec_shear_pipeline = jit(vmap(shear_pipeline, in_axes=(0, 0, None)))
    _ = vec_shear_pipeline(
        keys[0, None], {k: v[0, None] for k, v in params_jack_pos.items()}, init_g
    )

    # run on full dataset
    results_plus = []
    results_minus = []
    batch_size2 = int(n_jacks / n_splits)
    assert n_jacks % n_splits == 0
    for jj in range(n_splits):
        start, end = jj * batch_size2, (jj + 1) * batch_size2
        params_pos_jj = {k: v[start:end] for k, v in params_jack_pos.items()}
        params_neg_jj = {k: v[start:end] for k, v in params_jack_neg.items()}

        gp_jj = vec_shear_pipeline(keys[start:end], params_pos_jj, init_g)
        gn_jj = vec_shear_pipeline(keys[start:end], params_neg_jj, -init_g)

        results_plus.append(gp_jj)
        results_minus.append(gn_jj)

    g_pos_samples = jnp.concatenate(results_plus)
    g_neg_samples = jnp.concatenate(results_minus)

    assert g_pos_samples.shape[0] == n_jacks
    assert g_pos_samples.shape[-1] == 2
    assert g_neg_samples.shape[0] == n_jacks
    assert g_neg_samples.shape[-1] == 2

    return g_pos_samples, g_neg_samples
