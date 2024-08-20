from functools import partial

import blackjax
import jax
from jax import jit as jjit


def inference_loop(rng_key, kernel, initial_state, n_samples):

    @jjit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, n_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return (states, infos)


def inference_loop_multiple_chains(
    rng_key, kernel, initial_states, n_samples, n_chains
):

    @jjit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, n_chains)
        states, infos = jax.vmap(kernel)(keys, states)
        return states, (states, infos)

    keys = jax.random.split(rng_key, n_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

    return states, infos


def inference_loop_multiple_chains_with_data(
    rng_key,
    initial_states,
    tuned_params,
    all_data,
    kernel,
    log_prob_fn,
    n_samples,
    n_chains,
):
    """Source: https://blackjax-devs.github.io/sampling-book/models/change_of_variable_hmc.html"""
    # e.g.  kernel = blackjax.nuts.build_kernel()

    @jjit
    def step_fn(key, state, data, **params):
        logdensity = partial(log_prob_fn, data=data)
        return kernel(key, state, logdensity, **params)

    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, n_chains)
        states, infos = jax.vmap(step_fn)(keys, states, all_data, **tuned_params)
        return states, (states, infos)

    keys = jax.random.split(rng_key, n_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

    return (states, infos)
