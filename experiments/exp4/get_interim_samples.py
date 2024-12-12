#!/usr/bin/env python3
"""Check chains ran on a variety of galaxies with different SNR, initialization from the prior."""

import time
from functools import partial

import jax.numpy as jnp
import typer
from jax import jit, random, vmap
from jax._src.prng import PRNGKeyArray

from bpd import DATA_DIR
from bpd.chains import run_sampling_nuts, run_warmup_nuts
from bpd.draw import draw_gaussian
from bpd.initialization import init_with_prior
from bpd.pipelines.image_samples import (
    get_target_images,
    get_true_params_from_galaxy_params,
    loglikelihood,
    logprior,
    logtarget,
    sample_target_galaxy_params_simple,
)


def sample_prior(
    rng_key: PRNGKeyArray,
    *,
    mean_logflux: float = 6.0,
    sigma_logflux: float = 0.1,
    mean_loghlr: float = 1.0,
    sigma_loghlr: float = 0.05,
    shape_noise: float = 0.3,
    g1: float = 0.02,
    g2: float = 0.0,
) -> dict[str, float]:
    k1, k2, k3 = random.split(rng_key, 3)

    lf = random.normal(k1) * sigma_logflux + mean_logflux
    hlr = random.normal(k2) * sigma_loghlr + mean_loghlr

    other_params = sample_target_galaxy_params_simple(
        k3, shape_noise=shape_noise, g1=g1, g2=g2
    )

    return {"lf": lf, "hlr": hlr, **other_params}
