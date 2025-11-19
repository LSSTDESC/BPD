#!/usr/bin/env python3
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
import typer
from jax import Array, grad, jit, random, vmap
from jax.random import PRNGKey
from jax.scipy.stats import norm

from bpd import DATA_DIR
from bpd.chains import run_inference_nuts
from bpd.io import save_dataset
from bpd.prior import ellip_prior_e1e2
from bpd.sample import sample_ellip_prior
from bpd.shear import (
    inv_shear_func1,
    inv_shear_func2,
    inv_shear_transformation,
    shear_transformation,
)
from bpd.utils import process_in_batches, uniform_logpdf

eta_inv_fnc1 = lambda eta: eta2g(eta)[0]
eta_inv_fnc2 = lambda eta: eta2g(eta)[1]

grad_fnc1 = grad(eta_inv_fnc1)
grad_fnc2 = grad(eta_inv_fnc2)


def eta2g(eta: Array):
    assert eta.shape == (2,)
    eta1 = eta[0]
    eta2 = eta[1]
    abseta = jnp.sqrt(eta1**2 + eta2**2)
    g1 = eta1 * jnp.tanh(0.5 * abseta) / abseta
    g2 = eta2 * jnp.tanh(0.5 * abseta) / abseta
    return jnp.array([g1, g2])


def g2eta(g: Array):
    assert g.shape == (2,)
    g1 = g[0]
    g2 = g[1]
    absg = jnp.sqrt(g1**2 + g2**2)
    eta1 = g1 * jnp.arctanh(absg) * 2 / absg
    eta2 = g2 * jnp.arctanh(absg) * 2 / absg
    return jnp.array([eta1, eta2])


def sample_noisy_eta(rng_key, *, g, sigma_e: float, sigma_m: float, n: int = 1):
    k1, k2 = random.split(rng_key)
    es = sample_ellip_prior(k1, sigma_e, n)
    essh = shear_transformation(es, g)
    etas = vmap(g2eta)(essh)
    noisy_etas = etas.reshape(n, 2) + random.normal(k2, shape=(n, 2)) * sigma_m
    return noisy_etas, etas


def eta_target(eta, *, data, sigma_e: float, sigma_m: float):
    neta = data
    llike = norm.logpdf(neta, loc=eta, scale=sigma_m).sum()

    e1e2 = eta2g(eta)
    lprior1 = jnp.log(ellip_prior_e1e2(e1e2, sigma_e))

    grad1 = grad_fnc1(eta)
    grad2 = grad_fnc2(eta)
    prior2 = jnp.abs(grad1[..., 0] * grad2[..., 1] - grad1[..., 1] * grad2[..., 0])
    lprior2 = jnp.log(prior2)

    return llike + lprior1 + lprior2


_grad_shear_fnc1 = vmap(
    vmap(grad(inv_shear_func1), in_axes=(0, None)), in_axes=(0, None)
)
_grad_shear_fnc2 = vmap(
    vmap(grad(inv_shear_func2), in_axes=(0, None)), in_axes=(0, None)
)

# already vmapped once
_inv_shear_trans = vmap(inv_shear_transformation, in_axes=(0, None))


def shear_eta_target(g, *, data, sigma_e: float, sigma_e_int: float):
    assert g.shape == (2,)
    assert data.ndim == 3 and data.shape[2] == 2
    etas = data

    # P(eta' | alpha, g) = P(eps | alpha) * (del eps' / del eta') * (del eps / del eps')
    # jacobian on eta cancels between num and denom so we ignore it.
    eps_sheared = vmap(vmap(eta2g))(etas)
    eps = _inv_shear_trans(eps_sheared, g)
    num1 = jnp.log(ellip_prior_e1e2(eps, sigma_e))

    grad_eps1 = _grad_shear_fnc1(eps_sheared, g)
    grad_eps2 = _grad_shear_fnc2(eps_sheared, g)
    jac = jnp.abs(
        grad_eps1[..., 0] * grad_eps2[..., 1] - grad_eps1[..., 1] * grad_eps2[..., 0]
    )
    num2 = jnp.log(jac)

    num = num1 + num2

    # now denom P0(eta') = P0(eps') * (del eps' / del eta')
    denom = jnp.log(ellip_prior_e1e2(eps_sheared, sigma_e_int))

    # ratio
    ratio = jsp.special.logsumexp(num - denom, axis=-1)
    loglike = ratio.sum()

    # prior on shear
    g_mag = jnp.sqrt(g[0] ** 2 + g[1] ** 2)
    logprior = uniform_logpdf(g_mag, 0.0, 1.0) + jnp.log(1 / (2 * jnp.pi))

    return logprior + loglike


def run_one_eta_experiment(
    rng_key: PRNGKey,
    true_g,
    *,
    n_gals: int,
    n_samples_per_gal: int,
    n_samples_shear: int,
    sigma_e: float,
    sigma_e_int: float,
    sigma_m: float,
):
    k1, k2, k3 = random.split(rng_key, 3)

    # generate data
    netas, etas = sample_noisy_eta(
        k1, g=true_g, sigma_e=sigma_e, sigma_m=sigma_m, n=n_gals
    )

    # run phase 1
    target1 = partial(eta_target, sigma_e=sigma_e_int, sigma_m=sigma_m)
    pipe1 = partial(
        run_inference_nuts,
        logtarget=target1,
        n_samples=n_samples_per_gal,
        initial_step_size=0.01,
        max_num_doublings=2,
    )
    pipe1 = vmap(jit(pipe1))
    k2s = random.split(k2, n_gals)
    eta_samples = pipe1(k2s, netas, etas)

    # run phase 2
    target2 = partial(shear_eta_target, sigma_e=sigma_e, sigma_e_int=sigma_e_int)
    pipe2 = partial(
        run_inference_nuts,
        logtarget=target2,
        n_samples=n_samples_shear,
        initial_step_size=0.01,
        max_num_doublings=2,
    )
    pipe2 = jit(pipe2)

    g_samples = pipe2(k3, eta_samples, true_g)
    return g_samples


# draw shear samples from prior (unit disk)
def sample_shear_prior(rng_key):
    k1, k2 = random.split(rng_key)
    g_mag = random.uniform(k1, minval=0.0, maxval=1.0)
    theta = random.uniform(k2, minval=0.0, maxval=jnp.pi)
    g1 = g_mag * jnp.cos(2 * theta)
    g2 = g_mag * jnp.sin(2 * theta)
    return jnp.array([g1, g2])


def main(seed: int = 53, n_exps: int = 1000):
    save_dir = DATA_DIR / "cache_chains" / f"exp81_{seed}"
    save_dir.mkdir(exist_ok=True)

    k = random.key(seed)
    k1, k2 = random.split(k)
    k1s = random.split(k1, n_exps)
    k2s = random.split(k2, n_exps)

    pipe = partial(
        run_one_eta_experiment,
        n_gals=1000,
        n_samples_per_gal=300,
        n_samples_shear=2000,
        sigma_e=0.2,
        sigma_e_int=0.3,
        sigma_m=0.1,  # compared to distribution of ellipticity samples in experiment 70
    )
    pipe = jit(pipe)

    # run once (jitting)
    _ = pipe(k1s[0], jnp.array([0.02, 0.0]))

    true_gs = vmap(sample_shear_prior)(k1s)
    g_samples = process_in_batches(
        vmap(pipe),
        k2s,
        true_gs,
        n_points=n_exps,
        batch_size=50,
    )

    save_dataset(
        {"g_samples": g_samples, "true_g": true_gs}, save_dir / "eta_shear_samples.npz"
    )


if __name__ == "__main__":
    typer.run(main)
