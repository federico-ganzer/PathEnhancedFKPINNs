"""Point losses: A* (exact labels) and A0 (MC labels, FK-PINN).

R_data = mean_i (u_theta(x_i, t_i) - label_i)^2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def point_values(u, params, x, t) -> jax.Array:
    return jax.vmap(lambda xi, ti: u(params, xi, ti))(x, t)


def point_loss(u, params, x, t, labels) -> jax.Array:
    return jnp.mean((point_values(u, params, x, t) - labels) ** 2)


def point_residual_vector(u, params, x, t, labels) -> jax.Array:
    """r with mean-loss = sum r^2 (for Gauss-Newton spectra)."""
    return (point_values(u, params, x, t) - labels) / jnp.sqrt(x.shape[0])
