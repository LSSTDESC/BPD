from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, grad, vmap
from jax.numpy.linalg import norm
from jax.typing import ArrayLike

from bpd.prior import inv_shear_func1, inv_shear_func2, inv_shear_transformation

_grad_fnc1 = vmap(vmap(grad(inv_shear_func1), in_axes=(0, None)), in_axes=(0, None))
_grad_fnc2 = vmap(vmap(grad(inv_shear_func2), in_axes=(0, None)), in_axes=(0, None))
_inv_shear_trans = vmap(inv_shear_transformation, in_axes=(0, None))


def shear_loglikelihood(
    g: Array,
    e_post: Array,
    sigma_e: float,
    *,
    sigma_e_int: float,
    prior: Callable,
    interim_prior: Callable,
) -> ArrayLike:
    # Given by the inference procedure in Schneider et al. 2014
    # assume single shear g
    # assume e_obs.shape == (N, K, 2) where N is number of galaxies, K is samples per galaxy
    # the priors are callables for now on only ellipticities
    # the interim_prior should have been used when obtaining e_obs from the chain (i.e. for now same sigma)
    _, _, _ = e_post.shape  # (N, K, 2)
    _prior = partial(prior, sigma=sigma_e)
    _interim_prior = partial(interim_prior, sigma=sigma_e_int)

    e_post_mag = norm(e_post, axis=-1)
    denom = _interim_prior(e_post_mag)  # (N, K), can ignore angle in prior as uniform

    # for num, use trick
    # p(w_n' | g, alpha )  = p(w_n' \cross^{-1} g | alpha ) = p(w_n | alpha) * |jac(w_n / w_n')|

    # shape = (N, K, 2)
    grad1 = _grad_fnc1(e_post, g)
    grad2 = _grad_fnc2(e_post, g)
    absjacdet = jnp.abs(grad1[..., 0] * grad2[..., 1] - grad1[..., 1] * grad2[..., 0])

    e_post_unsheared = _inv_shear_trans(e_post, g)
    e_post_unsheared_mag = norm(e_post_unsheared, axis=-1)
    num = _prior(e_post_unsheared_mag) * absjacdet  # (N, K)

    ratio = jsp.special.logsumexp(jnp.log(num) - jnp.log(denom), axis=-1)
    return ratio.sum()
