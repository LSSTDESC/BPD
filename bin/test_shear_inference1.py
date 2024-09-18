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


def ellip_mag_prior(e, sigma: float = 0.3):
    """Unnormalized Prior for the magnitude of the ellipticity, domain is (0, 1)

    This distribution is taken from Gary's 2013 paper on Bayesian shear inference.
    The additional factor on the truncated Gaussian guarantees differentiability
    at e = 0 and e = 1.

    Gary uses 0.3 as a default level of shape noise.
    """
    return (1 - e**2) ** 2 * jnp.exp(-(e**2) / (2 * sigma**2))


def sample_mag_ellip_prior(rng_key, n=1, n_bins=100000, sigma=0.3):
    """Sample n points from Gary's ellipticity magnitude prior."""
    # this part could be cached
    e_array = jnp.linspace(0, 1, n_bins)
    p_array = ellip_mag_prior(e_array, sigma=sigma)
    p_array /= jnp.sum(p_array)

    return random.choice(rng_key, e_array, shape=(n,), p=p_array)


def sample_ellip_prior(rng_key, n=1, sigma=0.3):
    """Sample n ellipticities isotropic components with Gary's prior from magnitude."""
    key1, key2 = random.split(rng_key, 2)
    e_mag = sample_mag_ellip_prior(key1, n, sigma=sigma)
    e_phi = random.uniform(key2, shape=(n,), minval=0, maxval=2 * jnp.pi)
    e1 = e_mag * jnp.cos(2 * e_phi)
    e2 = e_mag * jnp.sin(2 * e_phi)
    return jnp.stack((e1, e2), axis=1)


def scalar_shear_transformation(e: tuple[float, float], g: tuple[float, float]):
    """Transform elliptiticies by a fixed shear (scalar version).

    The transformation we used is equation 3.4b in Seitz & Schneider (1997).

    NOTE: This function is meant to be vmapped later.
    """
    e1, e2 = e
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp + g_comp) / (1 + g_comp.conjugate() * e_comp)
    return e_prime.real, e_prime.imag


def scalar_inv_shear_transformation(e: tuple[float, float], g: tuple[float, float]):
    """Same as above but the inverse."""
    e1, e2 = e
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp - g_comp) / (1 - g_comp.conjugate() * e_comp)
    return e_prime.real, e_prime.imag


# useful for jacobian later, only need 2 grads really
inv_shear_func1 = lambda e, g: scalar_inv_shear_transformation(e, g)[0]
inv_shear_func2 = lambda e, g: scalar_inv_shear_transformation(e, g)[1]


def shear_transformation(e, g: tuple[float, float]):
    """Transform elliptiticies by a fixed shear.

    The transformation we used is equation 3.4b in Seitz & Schneider (1997).
    """
    e1, e2 = e[..., 0], e[..., 1]
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp + g_comp) / (1 + g_comp.conjugate() * e_comp)
    return jnp.stack([e_prime.real, e_prime.imag], axis=-1)


def inv_shear_transformation(e, g: tuple[float, float]):
    """Same as above but the inverse."""
    e1, e2 = e[..., 0], e[..., 1]
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp - g_comp) / (1 - g_comp.conjugate() * e_comp)
    return jnp.stack([e_prime.real, e_prime.imag], axis=-1)


# get synthetic measured sheared ellipticities
def sample_synthetic_sheared_ellips_unclipped(
    rng_key,
    g: tuple[float, float],
    n: int = 1,
    sigma_m: float = 0.1,
    sigma_e: float = 0.3,
):
    """We sample sheared ellipticities from N(e_int + g, sigma_m^2)"""
    key1, key2 = random.split(rng_key, 2)

    e_int = sample_ellip_prior(key1, n, sigma=sigma_e)
    e_sheared = shear_transformation(e_int, g)
    e_obs = random.normal(key2, shape=(n, 2)) * sigma_m + e_sheared.reshape(n, 2)
    return e_obs, e_sheared, e_int


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


# first we need a likelihood which is given by the inference procedure in Schneider et al. 2014
def loglikelihood_fn(
    g: tuple[float, float], e_obs, prior: Callable, interim_prior: Callable
):
    # assume single shear g
    # assume e_obs.shape == (N, K, 2) where N is number of galaxies, K is samples per galaxy
    # the priors are callables for now on only ellipticities
    # the interim_prior should have been used when obtaining e_obs from the chain (i.e. for now same sigma)
    # normalizatoin in priors can be ignored for now as alpha is fixed.
    N, K, _ = e_obs.shape

    e_obs_mag = jnp.sqrt(e_obs[..., 0] ** 2 + e_obs[..., 1] ** 2)
    denom = interim_prior(e_obs_mag)  # (N, K), can ignore angle in prior as uniform

    # for num we do trick p(w_n' | g, alpha )  = p(w_n' \cross^{-1} g | alpha ) = p(w_n | alpha) * |jac(w_n / w_n')|

    # shape = (N, K, 2)
    jac1 = jax.vmap(
        jax.vmap(jax.grad(inv_shear_func1, argnums=0), in_axes=(0, None)),
        in_axes=(0, None),
    )(e_obs, g)

    jac2 = jax.vmap(
        jax.vmap(jax.grad(inv_shear_func2, argnums=0), in_axes=(0, None)),
        in_axes=(0, None),
    )(e_obs, g)

    jac = jnp.stack([jac1, jac2], axis=-1)  # shape = (N, K, 2, 2)
    assert jac.shape == (N, K, 2, 2)
    jacdet = jnp.linalg.det(jac)  # shape = (N, K)

    e_obs_unsheared = inv_shear_transformation(e_obs, g)
    e_obs_unsheared_mag = jnp.sqrt(
        e_obs_unsheared[..., 0] ** 2 + e_obs_unsheared[..., 1] ** 2
    )
    num = prior(e_obs_unsheared_mag) * jacdet  # (N, K)

    ratio = jnp.log((1 / K)) + jsp.special.logsumexp(
        jnp.log(num) - jnp.log(denom), axis=-1
    )
    return jnp.sum(ratio)


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
