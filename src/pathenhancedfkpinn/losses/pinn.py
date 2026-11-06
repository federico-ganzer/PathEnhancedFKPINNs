"""PINN terms shared by every arm (arm P, draft eq. 1).

R_res = mean_j r_theta(x_j, t_j)^2,   r = d_t u - L u - F - f.
R_bc  = mean (u - g)^2 on box faces + mean (u(., 0) - u0)^2 on initial points.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def residuals(problem, u, params, x, t) -> jax.Array:
    """r_theta at points x (n, d), t (n,). Returns (n,)."""
    return jax.vmap(lambda xi, ti: problem.residual(lambda y, tau: u(params, y, tau), xi, ti))(x, t)


def residual_loss(problem, u, params, x, t) -> jax.Array:
    return jnp.mean(residuals(problem, u, params, x, t) ** 2)


def bc_loss(problem, u, params, x_bc, t_bc, x_ic) -> tuple[jax.Array, dict]:
    ub = jax.vmap(lambda xi, ti: u(params, xi, ti))(x_bc, t_bc)
    gb = jax.vmap(problem.boundary_value)(x_bc, t_bc)
    ui = jax.vmap(lambda xi: u(params, xi, jnp.zeros((), xi.dtype)))(x_ic)
    gi = jax.vmap(problem.u0)(x_ic)
    l_bc, l_ic = jnp.mean((ub - gb) ** 2), jnp.mean((ui - gi) ** 2)
    return l_bc + l_ic, {"bc_faces": l_bc, "bc_initial": l_ic}
