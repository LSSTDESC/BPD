import jax


def inference_loop(rng_key, initial_state, kernel, n_samples: int):
    """Function to run a single chain with a given kernel and obtain `n_samples`."""

    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, n_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return (states, infos)
