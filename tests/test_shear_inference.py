"""Test shear inference reaches desired accuracy for low-noise regime."""

import jax.numpy as jnp
import pytest
from jax import random

from bpd.pipelines.shear_inference import pipeline_shear_inference
from bpd.pipelines.toy_ellips import pipeline_toy_ellips_samples
from bpd.prior import sample_ellip_prior, shear_transformation


@pytest.mark.parametrize("seed", [1234, 4567])
def test_shear_inference_toy_ellipticities(seed):
    key = random.key(seed)
    k1, k2 = random.split(key)

    g1 = 0.02
    g2 = 0.0
    sigma_e = 1e-3
    sigma_m = 1e-4
    sigma_e_int = 3e-2

    e_post = pipeline_toy_ellips_samples(
        k1,
        g1=g1,
        g2=g2,
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        sigma_m=sigma_m,
        n_gals=1000,
        n_samples_per_gal=100,
    )[0]
    assert e_post.shape == (1000, 100, 2)
    e_post_trimmed = e_post[:, ::10, :]

    shear_samples = pipeline_shear_inference(
        k2,
        e_post_trimmed,
        true_g=jnp.array([g1, g2]),
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        n_samples=1000,
        initial_step_size=1e-3,
    )
    assert shear_samples.shape == (1000, 2)
    assert jnp.abs((jnp.mean(shear_samples[:, 0]) - g1) / g1) <= 3e-3
    assert jnp.abs(jnp.mean(shear_samples[:, 1])) <= 3e-3


@pytest.mark.parametrize("seed", [1234, 4567])
def test_shape_noise_scaling(seed):
    """No measure noise is added so scaling of shear scatter should be ~`shape_noise / sqrt(N)`."""
    key = random.key(seed)
    k1, k2 = random.split(key)

    g1 = 0.02
    g2 = 0.0
    sigma_e = 1e-3
    sigma_e_int = 3e-2
    n_gals = 1000

    # no observation noise, 1 sample.
    e_unsheared = sample_ellip_prior(k1, sigma=sigma_e, n=n_gals)
    e_post = shear_transformation(e_unsheared, (g1, g2)).reshape(n_gals, 1, 2)

    shear_samples = pipeline_shear_inference(
        k2,
        e_post,
        true_g=jnp.array([g1, g2]),
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        n_samples=2000,
        initial_step_size=1e-3,
    )
    assert shear_samples.shape == (2000, 2)
    g1_mean = jnp.mean(shear_samples[:, 0])
    g2_mean = jnp.mean(shear_samples[:, 1])
    g1_std = jnp.std(shear_samples[:, 0])
    g2_std = jnp.std(shear_samples[:, 1])

    # mean check
    assert jnp.abs((g1_mean - g1) / g1) <= 3e-3
    assert jnp.abs(g2_mean) <= 3e-3

    # scatter
    # NOTE: sqrt(2) factor or no??
    assert jnp.allclose(g1_std, sigma_e / jnp.sqrt(n_gals), atol=0, rtol=2e-2)
    assert jnp.allclose(g2_std, sigma_e / jnp.sqrt(n_gals), atol=0, rtol=2e-2)
