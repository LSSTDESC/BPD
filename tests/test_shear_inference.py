"""Test shear inference reaches desired accuracy for low-noise regime."""

import jax.numpy as jnp
import pytest

from scripts.get_shear_from_post_ellips import pipeline_shear_inference
from scripts.get_toy_ellip_samples import pipeline_toy_ellips_samples


@pytest.mark.parametrize("seed", [1234, 4567])
def test_shear_inference_toy_ellipticities(seed):

    g1 = 0.02
    g2 = 0.0
    sigma_e = 1e-3
    sigma_m = 1e-4

    e_post = pipeline_toy_ellips_samples(
        seed,
        g1=g1,
        g2=g2,
        sigma_e=sigma_e,
        sigma_e_int=2 * sigma_e,
        sigma_m=sigma_m,
        n_samples=1000,
        k=100,
    )[0]
    assert e_post.shape == (1000, 100, 2)
    e_post_trimmed = e_post[:, ::10, :]

    shear_samples = pipeline_shear_inference(
        seed,
        e_post_trimmed,
        jnp.array([g1, g2]),
        sigma_e=sigma_e,
        sigma_e_int=2 * sigma_e,
        n_samples=1000,
    )
    assert shear_samples.shape == (1000, 2)
    assert jnp.abs((jnp.mean(shear_samples[:, 0]) - g1) / g1) <= 3e-3
    assert jnp.abs(jnp.mean(shear_samples[:, 1])) <= 3e-3
