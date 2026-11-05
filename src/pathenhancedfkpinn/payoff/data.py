"""Data payoff Y. Offline, float64, no gradient.

    Y = u0(X_t) + int_0^t f(X_s) ds      (left-point sum on the simulation grid)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def data_payoff(problem, X: jax.Array, XT: jax.Array, ds: jax.Array) -> jax.Array:
    """X (M, n, d) left-point states, XT (M, d) terminal states, ds (n,) or (M, n). Returns (M,)."""
    Y = jax.vmap(problem.u0)(XT)
    fX = jax.vmap(jax.vmap(problem.f))(X)  # (M, n); zeros when there is no source
    return Y + jnp.sum(fX * jnp.broadcast_to(ds, fX.shape), axis=1)
