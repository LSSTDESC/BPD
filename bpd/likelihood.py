from typing import Callable

import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, vmap
from jax.typing import ArrayLike

from bpd.prior import inv_shear_func1, inv_shear_func2, inv_shear_transformation


def shear_loglikelihood_unreduced(
    g: tuple[float, float], e_post, prior: Callable, interim_prior: Callable
) -> ArrayLike:
    # Given by the inference procedure in Schneider et al. 2014
    # assume single shear g
    # assume e_obs.shape == (N, K, 2) where N is number of galaxies, K is samples per galaxy
    # the priors are callables for now on only ellipticities
    # the interim_prior should have been used when obtaining e_obs from the chain (i.e. for now same sigma)
    # normalizatoin in priors can be ignored for now as alpha is fixed.
    N, K, _ = e_post.shape

    e_post_mag = jnp.sqrt(e_post[..., 0] ** 2 + e_post[..., 1] ** 2)
    denom = interim_prior(e_post_mag)  # (N, K), can ignore angle in prior as uniform

    # for num we do trick p(w_n' | g, alpha )  = p(w_n' \cross^{-1} g | alpha ) = p(w_n | alpha) * |jac(w_n / w_n')|

    # shape = (N, K, 2)
    grad1 = vmap(
        vmap(grad(inv_shear_func1, argnums=0), in_axes=(0, None)),
        in_axes=(0, None),
    )(e_post, g)

    grad2 = vmap(
        vmap(grad(inv_shear_func2, argnums=0), in_axes=(0, None)),
        in_axes=(0, None),
    )(e_post, g)

    absjacdet = jnp.abs(grad1[..., 0] * grad2[..., 1] - grad1[..., 1] * grad2[..., 0])

    e_post_unsheared = inv_shear_transformation(e_post, g)
    e_obs_unsheared_mag = jnp.sqrt(
        e_post_unsheared[..., 0] ** 2 + e_post_unsheared[..., 1] ** 2
    )
    num = prior(e_obs_unsheared_mag) * absjacdet  # (N, K)

    ratio = jnp.log((1 / K)) + jsp.special.logsumexp(
        jnp.log(num) - jnp.log(denom), axis=-1
    )
    return ratio


def shear_loglikelihood(
    g: tuple[float, float], e_post, prior: Callable, interim_prior: Callable
) -> float:
    """Reduce with sum"""
    return shear_loglikelihood_unreduced(g, e_post, prior, interim_prior).sum()
