from typing import Callable

import jax.numpy as jnp
from jax import device_put, jit, random
from tqdm import tqdm


def run_bootstrap(
    rng_key,
    *args,  # like init positions, shared across + and -
    post_params: dict,
    pipeline: Callable,
    n_gals: int,
    n_boots: int = 10,
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
