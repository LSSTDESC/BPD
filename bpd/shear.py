import jax.numpy as jnp
from jax import Array, vmap


def scalar_shear_transformation(e: Array, g: Array):
    """Transform elliptiticies by a fixed shear (scalar version).

    The transformation we used is equation 3.4b in Seitz & Schneider (1997).

    NOTE: This function is meant to be vmapped later.
    """
    assert e.shape == (2,) and g.shape == (2,)

    e1, e2 = e
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp + g_comp) / (1 + g_comp.conjugate() * e_comp)
    return jnp.array([e_prime.real, e_prime.imag])


def scalar_inv_shear_transformation(e: Array, g: Array):
    """Same as above but the inverse."""
    assert e.shape == (2,) and g.shape == (2,)
    e1, e2 = e
    g1, g2 = g

    e_comp = e1 + e2 * 1j
    g_comp = g1 + g2 * 1j

    e_prime = (e_comp - g_comp) / (1 - g_comp.conjugate() * e_comp)
    return jnp.array([e_prime.real, e_prime.imag])


# batched
shear_transformation = vmap(scalar_shear_transformation, in_axes=(0, None))
inv_shear_transformation = vmap(scalar_inv_shear_transformation, in_axes=(0, None))

# useful for jacobian later
inv_shear_func1 = lambda e, g: scalar_inv_shear_transformation(e, g)[0]
inv_shear_func2 = lambda e, g: scalar_inv_shear_transformation(e, g)[1]
