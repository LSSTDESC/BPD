"""Scripts to test the convergence of chains for various shear inference cases."""

from functools import partial

import jax.numpy as jnp
import pytest
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
from jax import jit, random, vmap

from bpd.chains import run_inference_nuts
from bpd.pipelines.shear_inference import pipeline_shear_inference_ellipticities
from bpd.pipelines.toy_ellips import logtarget as logtarget_toy_ellips
from bpd.pipelines.toy_ellips import pipeline_toy_ellips_samples
from bpd.prior import sample_noisy_ellipticities_unclipped


@pytest.mark.parametrize("seed", [1234, 4567])
def test_interim_toy_convergence(seed):
    """Check efficiency and convergence of chains for 100 galaxies."""
    g1, g2 = 0.02, 0.0
    sigma_m = 1e-4
    sigma_e = 1e-3
    sigma_e_int = 4e-2
    n_gals = 100
    n_samples_per_galaxy = 1_000  # enough to test convergence
    max_num_doublings = 2

    true_g = jnp.array([g1, g2])

    key = random.key(seed)
    k1, k2 = random.split(key)

    e_obs, e_sheared, _ = sample_noisy_ellipticities_unclipped(
        k1, g=true_g, sigma_m=sigma_m, sigma_e=sigma_e, n=n_gals
    )

    # now we vectorize and run 4 chains over each observed ellipticity sample
    keys2 = random.split(k2, (n_gals, 4))
    _logtarget = partial(logtarget_toy_ellips, sigma_m=sigma_m, sigma_e_int=sigma_e_int)

    _do_inference_jitted = jit(
        partial(
            run_inference_nuts,
            logtarget=_logtarget,
            initial_step_size=sigma_e,
            n_samples=n_samples_per_galaxy,
            max_num_doublings=max_num_doublings,
        )
    )
    _do_inference_vmapped = vmap(_do_inference_jitted, in_axes=(0, None, None))
    _run_inference = vmap(_do_inference_vmapped, in_axes=(0, 0, 0))

    e_post = _run_inference(keys2, e_sheared, e_obs)
    assert e_post.shape == (n_gals, 4, n_samples_per_galaxy, 2)

    for ii in (0, 1):
        ess_list = []
        rhat_list = []
        e_ii = e_post[..., ii]

        for jj in range(n_gals):
            ess_list.append(effective_sample_size(e_ii[jj]))
            rhat_list.append(potential_scale_reduction(e_ii[jj]))

        ess = jnp.array(ess_list)
        rhat = jnp.array(rhat_list)

        assert ess.min() > 0.7 * n_samples_per_galaxy * 4
        assert jnp.abs(rhat - 1).max() < 0.01


@pytest.mark.parametrize("seed", [1234, 4567])
def test_toy_shear_convergence(seed):
    g1, g2 = 0.02, 0.0
    sigma_m = 1e-4
    sigma_e = 1e-3
    sigma_e_int = 4e-2
    n_gals = 1000
    n_samples = 1000
    k = 10  # enough to test convergence

    true_g = jnp.array([g1, g2])

    key = random.key(seed)
    k1, k2 = random.split(key)

    e_post, _, _ = pipeline_toy_ellips_samples(
        k1,
        g1=g1,
        g2=g2,
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        sigma_m=sigma_m,
        n_gals=n_gals,
        n_samples_per_gal=k,
    )

    # run 4 shear chains over the given e_post
    _pipeline_shear1 = partial(
        pipeline_shear_inference_ellipticities,
        init_g=true_g,
        sigma_e=sigma_e,
        sigma_e_int=sigma_e_int,
        n_samples=n_samples,
        initial_step_size=1e-2,
    )
    _pipeline_shear1_jitted = jit(_pipeline_shear1)
    _pipeline_shear = vmap(_pipeline_shear1_jitted, in_axes=(0, None))

    keys2 = random.split(k2, 4)

    g_samples = _pipeline_shear(keys2, e_post)

    assert g_samples.shape == (4, 1000, 2)

    # check each component
    for ii in (0, 1):
        ess = effective_sample_size(g_samples[..., ii])
        rhat = potential_scale_reduction(g_samples[..., ii])

        assert ess > 0.7 * 4000
        assert jnp.abs(rhat - 1) < 0.01
