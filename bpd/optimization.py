from functools import Callable

import optax
from jax import Array, value_and_grad
from jax.lax import scan


def find_likelihood_peak_scan(
    data: Array,
    params_init: dict,
    *,
    lr: float = 1e-3,
    n_steps: int = 1000,
    likelihood_fnc: Callable,
) -> tuple[float, float]:
    """Find the peak of the likelihood using gradient descent."""

    def loss(params):
        return -likelihood_fnc(params, data)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params_init)
    params = params_init

    def step(state, _):
        params, opt_state = state
        loss_val, grads = value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss_val

    (params, _), loss_vals = scan(step, (params, opt_state), length=n_steps)

    return params, loss_vals
