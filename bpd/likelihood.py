from typing import Callable

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax.scipy import stats
from jax.typing import ArrayLike


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


def gaussian_image_loglikelihood(
    params: dict[str, Array],
    data: Array,
    fixed_params: dict[str, Array],
    *,
    draw_fnc: Callable,
    background: float,
    free_flux_hlr: bool = True,
    free_dxdy: bool = True,
):
    _draw_params = {}

    if free_dxdy:
        _draw_params["x"] = params["dx"] + fixed_params["x"]
        _draw_params["y"] = params["dy"] + fixed_params["y"]

    else:
        _draw_params["x"] = fixed_params["x"]
        _draw_params["y"] = fixed_params["y"]

    if free_flux_hlr:
        _draw_params["f"] = 10 ** params["lf"]
        _draw_params["hlr"] = 10 ** params["lhlr"]

    else:
        _draw_params["f"] = fixed_params["f"]
        _draw_params["hlr"] = fixed_params["hlr"]

    _draw_params["e1"] = params["e1"]
    _draw_params["e2"] = params["e2"]

    model = draw_fnc(**_draw_params)
    likelihood_pp = stats.norm.logpdf(data, loc=model, scale=jnp.sqrt(background))
    return jnp.sum(likelihood_pp)
