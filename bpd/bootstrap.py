from typing import Callable

import jax.numpy as jnp
from jax import device_put, jit, random, vmap
from tqdm import tqdm

from bpd.utils import process_in_batches


def run_bootstrap(
    rng_key,
    *args,  # like init positions, shared across + and -
    post_params: dict,
    pipeline: Callable,
    n_gals: int,
    n_boots: int,
    no_bar: bool = True,
    cpu=None,  # device_put does nothing by default
    gpu=None,
):
    """Obtain boostrap samples of shear posteriors. Has memory friendly option."""
    for p in post_params:
        assert n_gals == post_params[p].shape[0]

    results = []
    keys = random.split(rng_key, n_boots)
    pipe = jit(pipeline)

    post_params = device_put(post_params, cpu)
    args = (device_put(x, cpu) for x in args)

    for ii in tqdm(range(n_boots), desc="Bootstrap #", disable=no_bar):
        k1, k2 = random.split(keys[ii])
        args_ii = (device_put(x[ii], gpu) for x in args)
        indices = random.randint(k1, shape=(n_gals,), minval=0, maxval=n_gals)
        _params = device_put({k: v[indices] for k, v in post_params.items()}, gpu)
        samples_ii = device_put(pipe(k2, _params, *args_ii), cpu)
        results.append(samples_ii)

    samples = {}
    for k in results[0]:
        samples[k] = jnp.stack([rs[k] for rs in results], axis=0)

    return samples


def run_bootstrap_shear_vectorized(
    rng_key,
    *args,  # arguments to vectorize like "init_positions".
    post_params_plus: dict,
    post_params_minus: dict,
    shear_pipeline: Callable,
    n_gals: int,
    n_boots: int = 100,
    n_splits: int = 10,
    no_bar: bool = True,
):
    for p in post_params_plus:
        assert n_gals == post_params_plus[p].shape[0]

    k1, k2 = random.split(rng_key)
    k2s = random.split(k2, n_boots)
    indices = random.randint(k1, shape=(n_boots, n_gals), minval=0, maxval=n_gals)
    boot_ppp = {k: v[indices] for k, v in post_params_plus.items()}
    boot_ppm = {k: v[indices] for k, v in post_params_minus.items()}

    batch_size = n_boots // n_splits
    samples_plus = process_in_batches(
        vmap(shear_pipeline),
        k2s,
        boot_ppp,
        *args,
        n_points=n_boots,
        batch_size=batch_size,
        no_bar=no_bar,
    )
    samples_minus = process_in_batches(
        vmap(shear_pipeline),
        k2s,
        boot_ppm,
        *args,
        n_points=n_boots,
        batch_size=batch_size,
        no_bar=no_bar,
    )

    return samples_plus, samples_minus
