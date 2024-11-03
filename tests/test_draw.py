"""Minimal amount of code that checks jax-galsim is working correctly."""

from functools import partial

import jax.numpy as jnp
from jax import jit as jjit
from jax import vmap

from bpd.draw import draw_gaussian


def test_jax_galsim():
    _draw_fnc1 = partial(draw_gaussian, slen=101, fft_size=512)
    _draw_fnc = vmap(jjit(_draw_fnc1))

    f = jnp.array([1000, 2000])
    hlr = jnp.array([0.9, 1.0])
    e1 = jnp.array([0.2, -0.1])
    e2 = jnp.array([0.0, 0.2])
    g1 = jnp.array([0.02, 0.0])
    g2 = jnp.array([0.0, 0.02])
    x = jnp.array([1.0, 0.0])
    y = jnp.array([0.0, 1.0])

    a = _draw_fnc(f=f, hlr=hlr, e1=e1, e2=e2, g1=g1, g2=g2, x=x, y=y)
    assert a.ndim == 3
    assert a.shape == (2, 101, 101)
