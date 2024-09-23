#!/usr/bin/env python3

import os
from typing import Callable

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from functools import partial

import blackjax
import h5py
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
from jax import jit as jjit
from jax import random, vmap

from bpd.chains import inference_loop

jax.config.update("jax_enable_x64", True)


###### INTERIM POSTERIOR
def log_target(e_sheared, e_obs, sigma_m: float = 0.1, interim_prior: Callable = None):
    assert e_sheared.shape == (2,) and e_obs.shape == (2,)

    # ignore angle because flat
    # prior enforces magnitude < 1.0 for posterior samples
    e_sheared_mag = jnp.sqrt(e_sheared[0] ** 2 + e_sheared[1] ** 2)
    prior = jnp.log(interim_prior(e_sheared_mag))

    likelihood = jnp.sum(jsp.stats.norm.logpdf(e_obs, loc=e_sheared, scale=sigma_m))
    return prior + likelihood


def do_inference(rng_key, init_positions, e_obs, m=10, sigma_m=0.1, sigma_e=0.3):
    interim_prior = partial(ellip_mag_prior, sigma=sigma_e * 2)
    _logtarget = partial(
        log_target, e_obs=e_obs, sigma_m=sigma_m, interim_prior=interim_prior
    )

    key1, key2 = random.split(rng_key)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        _logtarget,
        progress_bar=False,
        is_mass_matrix_diagonal=True,
        max_num_doublings=5,
        initial_step_size=0.01,
        target_acceptance_rate=0.80,
    )

    (init_states, tuned_params), _ = warmup.run(key1, init_positions, 500)
    kernel = blackjax.nuts(_logtarget, **tuned_params).step
    states, _ = inference_loop(key2, init_states, kernel=kernel, n_samples=m)
    return states.position


# now implement sheared inference procedure assuming K observed samples per galaxy
# ie imagine running mcmc and getting 10 samples per noisy galaxy


SEED = 42


def main():
    # for all experiments we produce too many samples to fit in GPU,
    # we do it in baches and save intermediate to h5py files

    # sanity check 1: low noise, shape noise OK
    g = 0.05, 0.0
    n_gals = int(1e7)  # target
    sigma_m = 0.0001
    sigma_e = 0.3
    key = random.key(42)
    batch_size = int(1e5)
    n_batches = n_gals // batch_size

    keys = random.split(key, n_batches)

    _run_inference1 = partial(do_inference, sigma_e=sigma_e, sigma_m=sigma_m, m=10)
    _run_inference = jjit(vmap(_run_inference1, in_axes=(0, 0, 0)))

    for ii in range(n_batches):

        k1, k2 = random.split(keys[ii])

        e_obs, e_sheared, _ = sample_synthetic_sheared_ellips_unclipped(
            k1, g, n=batch_size, sigma_m=sigma_m, sigma_e=sigma_e
        )

        _ks = random.split(k2, batch_size)
        e_post = _run_inference(_ks, e_sheared, e_obs)  # init positions = truth
        with h5py.File("int_post.hdf5", "w") as f:
            pass
    e_obs.shape, e_sheared.shape


if __name__ == "__main__":
    main()
