"""Test shear inference reaches desired accuracy for low-noise regime."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from bpd.pipelines.shear_inference import pipeline_shear_inference
from bpd.pipelines.toy_ellips import pipeline_toy_ellips_samples


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
    assert np.testing.assert_allclose(
        jnp.std(shear_samples[:, 0]), sigma_e / jnp.sqrt(1000), rtol=0.1, atol=0
    )
    assert np.testing.assert_allclose(
        jnp.std(shear_samples[:, 1]), sigma_e / jnp.sqrt(1000), rtol=0.1, atol=0
    )
