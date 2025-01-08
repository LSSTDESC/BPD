from typing import Callable

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, grad, vmap
from jax.typing import ArrayLike

from bpd.prior import (
    ellip_prior_e1e2,
    inv_shear_func1,
    inv_shear_func2,
    inv_shear_transformation,
)

_grad_fnc1 = vmap(vmap(grad(inv_shear_func1), in_axes=(0, None)), in_axes=(0, None))
_grad_fnc2 = vmap(vmap(grad(inv_shear_func2), in_axes=(0, None)), in_axes=(0, None))
_inv_shear_trans = vmap(inv_shear_transformation, in_axes=(0, None))


def true_ellip_logprior(e_post: Array, g: Array, *, sigma_e: float):
    """Implementation of GB's true prior on interim posterior samples of ellipticities."""

    # jacobian of inverse shear transformation
    grad1 = _grad_fnc1(e_post, g)
    grad2 = _grad_fnc2(e_post, g)
    absjacdet = jnp.abs(grad1[..., 0] * grad2[..., 1] - grad1[..., 1] * grad2[..., 0])

    # true prior on unsheared ellipticity
    e_post_unsheared = _inv_shear_trans(e_post, g)
    prior_val = ellip_prior_e1e2(e_post_unsheared, sigma=sigma_e)

    return jnp.log(prior_val) + jnp.log(absjacdet)


def shear_loglikelihood(
    g: Array,
    post_params: dict[str, Array],
    *,
    logprior: Callable,
    interim_logprior: Callable,  # fixed
) -> ArrayLike:
    """Shear Likelihood implementation of Schneider et al. 2014."""
    denom = interim_logprior(post_params)
    num = logprior(post_params, g)
    ratio = jsp.special.logsumexp(num - denom, axis=-1)
    return ratio.sum()
