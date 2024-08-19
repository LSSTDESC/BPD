from functools import partial

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit as jjit


def run_warmup(seed, data, init_states, n_steps=300, logdensity_fn=None):
    logdensity = jjit(partial(logdensity_fn, data=data))
    warmup = blackjax.window_adaptation(
        blackjax.nuts, logdensity, progress_bar=False, is_mass_matrix_diagonal=True
    )
    (initial_states, tuned_params), adapt_info = warmup.run(seed, init_states, n_steps)
    return initial_states, tuned_params, adapt_info


def inference_loop(rng_key, kernel, initial_state, n_samples):
    """Should ensure kernel is jitted."""

    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, n_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return (states, infos)


def inference_loop_multiple_chains(rng_key, kernel, initial_state, n_samples, n_chains):

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, n_chains)
        states, infos = jax.vmap(kernel)(keys, states)
        return states, (states, infos)

    keys = jax.random.split(rng_key, n_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos


def inference_loop_multiple_chains_with_data(
    rng_key, initial_states, tuned_params, all_data, log_prob_fn, n_samples, n_chains
):
    """Source: https://blackjax-devs.github.io/sampling-book/models/change_of_variable_hmc.html"""
    kernel = blackjax.nuts.build_kernel()

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


def run_one_chain(
    rng_key, data, initial_position, logdensity_fn, n_warmup=300, n_samples=1000
):
    """Do warmup followed by inference"""

    # keys
    _, warmup_key, sample_key = jax.random.split(rng_key, 3)

    # trick that allows for data to be vmapped later
    logdensity = jjit(partial(logdensity_fn, data=data))

    # adaptation
    warmup = blackjax.window_adaptation(
        blackjax.nuts, logdensity, progress_bar=False, is_mass_matrix_diagonal=False
    )
    (state, parameters), adapt_info = warmup.run(
        warmup_key, initial_position, num_steps=n_warmup
    )

    # sampling
    kernel = jjit(blackjax.nuts(logdensity, **parameters).step)  # not jitted by default
    states, sample_info = inference_loop(sample_key, kernel, state, n_samples)

    return states.position, sample_info, adapt_info
