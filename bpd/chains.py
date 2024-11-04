from functools import partial
from typing import Callable

import blackjax
import jax
from jax import random
from jax._src.prng import PRNGKeyArray
from jax.typing import ArrayLike


def inference_loop(rng_key, initial_state, kernel, n_samples: int):
    """Function to run a single chain with a given kernel and obtain `n_samples`."""

    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, n_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return (states, infos)


def run_inference_nuts(
    rng_key: PRNGKeyArray,
    init_positions: ArrayLike,
    data: ArrayLike,
    *,
    logtarget: Callable,
    n_samples: int,
    initial_step_size: float,
    n_warmup_steps: int = 500,
    is_mass_matrix_diagonal: bool = True,
    max_num_doublings: int = 5,
    target_acceptance_rate: float = 0.80,
):
    key1, key2 = random.split(rng_key)

    _logtarget = partial(logtarget, data=data)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        _logtarget,
        progress_bar=False,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        max_num_doublings=max_num_doublings,
        initial_step_size=initial_step_size,
        target_acceptance_rate=target_acceptance_rate,
    )

    (init_states, tuned_params), _ = warmup.run(key1, init_positions, n_warmup_steps)
    kernel = blackjax.nuts(_logtarget, **tuned_params).step
    states, _ = inference_loop(key2, init_states, kernel=kernel, n_samples=n_samples)
    return states.position
