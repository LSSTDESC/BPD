import jax.numpy as jnp
from jax import Array, grad, vmap
from jax.numpy.linalg import norm
from jax.scipy import stats
from jax.typing import ArrayLike

from bpd.shear import (
    inv_shear_func1,
    inv_shear_func2,
    inv_shear_transformation,
)


def ellip_mag_prior(e_mag: ArrayLike, sigma: float) -> ArrayLike:
    """Prior for the magnitude of the ellipticity with domain (0, 1).

    This distribution corresponds to Gary's 2013 paper on Bayesian shear inference.

    Importantly, the paper did not include an additional factor of |e| that is needed
    make this the correct expression for a Gaussian in `polar` coordinates. We include
    this factor in this equation. This blog post is helpful: https://andrewcharlesjones.github.io/journal/rayleigh.html.

    The additional factor of (1-e^2)^2 introduced by Gary guarantees differentiability
    at e = 0 and e = 1.
    """

    # norm from Mathematica
    _norm = -4 * sigma**4 + sigma**2 + 8 * sigma**6 * (1 - jnp.exp(-1 / (2 * sigma**2)))
    return (1 - e_mag**2) ** 2 * e_mag * jnp.exp(-(e_mag**2) / (2 * sigma**2)) / _norm


def ellip_prior_e1e2(e1e2: Array, sigma: float) -> ArrayLike:
    """Prior on e1, e2 using Gary's prior for magnitude. Includes Jacobian factor: `|e|`"""
    e_mag = norm(e1e2, axis=-1)

    _norm1 = (
        -4 * sigma**4 + sigma**2 + 8 * sigma**6 * (1 - jnp.exp(-1 / (2 * sigma**2)))
    )
    _norm2 = 2 * jnp.pi  # from f(\beta) and Jacobian
    _norm = _norm1 * _norm2

    # jacobian factor also cancels `e_mag` term below
    return (1 - e_mag**2) ** 2 * jnp.exp(-(e_mag**2) / (2 * sigma**2)) / _norm


_grad_fnc1 = vmap(vmap(grad(inv_shear_func1), in_axes=(0, None)), in_axes=(0, None))
_grad_fnc2 = vmap(vmap(grad(inv_shear_func2), in_axes=(0, None)), in_axes=(0, None))
_inv_shear_trans = vmap(inv_shear_transformation, in_axes=(0, None))


def interim_gprops_logprior(
    params: dict[str, Array],
    *,
    sigma_e: float,
    sigma_x: float = 0.5,  # pixels
    flux_bds: tuple = (-1.0, 9.0),
    hlr_bds: tuple = (-2.0, 1.0),
    free_flux_hlr: bool = True,
    free_dxdy: bool = True,
) -> Array:
    prior = jnp.array(0.0)

    if free_flux_hlr:
        f1, f2 = flux_bds
        prior += stats.uniform.logpdf(params["lf"], f1, f2 - f1)

        h1, h2 = hlr_bds
        prior += stats.uniform.logpdf(params["lhlr"], h1, h2 - h1)

    if free_dxdy:
        prior += stats.norm.logpdf(params["dx"], loc=0.0, scale=sigma_x)
        prior += stats.norm.logpdf(params["dy"], loc=0.0, scale=sigma_x)

    e1e2 = jnp.stack((params["e1"], params["e2"]), axis=-1)
    prior += jnp.log(ellip_prior_e1e2(e1e2, sigma=sigma_e))

    return prior


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


def true_all_params_logprior(
    post_params: dict[str, Array],
    g: Array,
    *,
    sigma_e: float,
    mean_logflux: float,
    sigma_logflux: float,
    mean_loghlr: float,
    sigma_loghlr: float,
):
    lf = post_params["lf"]
    lhlr = post_params["lhlr"]
    e1e2 = jnp.stack((post_params["e1"], post_params["e2"]), axis=-1)

    prior = jnp.array(0.0)

    # we also pickup a jacobian in the corresponding probability densities
    prior += stats.norm.logpdf(lf, loc=mean_logflux, scale=sigma_logflux)
    prior += stats.norm.logpdf(lhlr, loc=mean_loghlr, scale=sigma_loghlr)

    # elliptcity
    prior += true_ellip_logprior(e1e2, g, sigma_e=sigma_e)

    return prior


def true_all_params_trunc_logprior(
    post_params: dict[str, Array],
    g: Array,
    *,
    sigma_e: float,
    mean_logflux: float,
    sigma_logflux: float,
    min_logflux: float,
    mean_loghlr: float,
    sigma_loghlr: float,
):
    lf = post_params["lf"]
    lhlr = post_params["lhlr"]
    e1e2 = jnp.stack((post_params["e1"], post_params["e2"]), axis=-1)

    prior = jnp.array(0.0)

    # log flux uses a truncted normal distribution
    a = (min_logflux - mean_logflux) / sigma_logflux
    b = jnp.inf
    prior += stats.truncnorm.logpdf(lf, a=a, b=b, loc=mean_logflux, scale=sigma_logflux)

    # hlr
    prior += stats.norm.logpdf(lhlr, loc=mean_loghlr, scale=sigma_loghlr)

    # elliptcity
    prior += true_ellip_logprior(e1e2, g, sigma_e=sigma_e)

    return prior


def true_all_params_skew_logprior(
    post_params: dict[str, Array],
    g: Array,
    *,
    sigma_e: float,
    a: float,  # skewness
    mean_logflux: float,
    sigma_logflux: float,
    mean_loghlr: float,
    sigma_loghlr: float,
):
    lf = post_params["lf"]
    lhlr = post_params["lhlr"]
    e1e2 = jnp.stack((post_params["e1"], post_params["e2"]), axis=-1)

    prior = jnp.array(0.0)

    # log flux uses a skew normal distribution
    # jax does not have an implementation of skew normal so we use the equation in:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html
    x = (lf - mean_logflux) / sigma_logflux
    prior += (
        jnp.log(2) + stats.norm.logpdf(x) + stats.norm.logcdf(a * x) / sigma_logflux
    )

    # hlr
    prior += stats.norm.logpdf(lhlr, loc=mean_loghlr, scale=sigma_loghlr)

    # elliptcity
    prior += true_ellip_logprior(e1e2, g, sigma_e=sigma_e)

    return prior
