from functools import partial

import blackjax
import jax
from jax import jit as jjit

from bpd.chains import inference_loop


def run_warmup_nuts(seed, data, init_states, n_steps=300, logdensity_fn=None):
    logdensity = jjit(partial(logdensity_fn, data=data))
    warmup = blackjax.window_adaptation(
        blackjax.nuts, logdensity, progress_bar=False, is_mass_matrix_diagonal=True
    )
    (initial_states, tuned_params), adapt_info = warmup.run(seed, init_states, n_steps)
    return initial_states, tuned_params, adapt_info


def run_one_nuts_chain(
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
