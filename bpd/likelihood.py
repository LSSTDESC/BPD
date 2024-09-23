from typing import Callable

import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, vmap
from jax.typing import ArrayLike

from bpd.prior import inv_shear_func1, inv_shear_func2, inv_shear_transformation


def shear_loglikelihood_unreduced(
    g: tuple[float, float], e_obs, prior: Callable, interim_prior: Callable
) -> ArrayLike:
    # Given by the inference procedure in Schneider et al. 2014
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
    jac1 = vmap(
        vmap(grad(inv_shear_func1, argnums=0), in_axes=(0, None)),
        in_axes=(0, None),
    )(e_obs, g)

    jac2 = vmap(
        vmap(grad(inv_shear_func2, argnums=0), in_axes=(0, None)),
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
    return ratio


def shear_loglikelihood(
    g: tuple[float, float], e_obs, prior: Callable, interim_prior: Callable
) -> float:
    """Reduce with sum"""
    return shear_loglikelihood_unreduced(g, e_obs, prior, interim_prior).sum()
