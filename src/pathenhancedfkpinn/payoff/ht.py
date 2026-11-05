"""Horvitz-Thompson time-step subsampling (draft eq. 16).

Steps are split into J blocks of m steps. One step r_j per block is drawn uniformly at
simulation time. The estimator

    sum_j [ g(s_j) . (W_{s_{j+1}} - W_{s_j}) + m (g(r_j) - g(s_j)) . dW_{r_j} ]
    + sum_j [ h(s_j) (s_{j+1} - s_j) + m (h(r_j) - h(s_j)) ds_{r_j} ]

is a weighted sum over 2J nodes per path:
    node s_j: state X_{s_j}, increment W_{s_{j+1}} - W_{s_j} - m dW_{r_j}, ds (s_{j+1}-s_j) - m ds_{r_j}
    node r_j: state X_{r_j}, increment m dW_{r_j},                          ds m ds_{r_j}
so `model_payoff` evaluates it unchanged. Conditional on the path, its mean is the full Euler sum.

Warning (verified): the HT variance grows like m^2. Never use it on a configuration
that fails D0.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Nodes(NamedTuple):
    X: jax.Array  # (M, 2J, d)
    dW: jax.Array  # (M, 2J, d) effective increments
    s: jax.Array  # (M, 2J) node times
    ds: jax.Array  # (M, 2J) effective time weights


def draw_offsets(key: jax.Array, M: int, J: int, m: int) -> jax.Array:
    """r_j - s_j for each path and block, uniform in {0, ..., m-1}. Shape (M, J)."""
    return jax.random.randint(key, (M, J), 0, m)


def ht_nodes(X: jax.Array, dW: jax.Array, s_grid: jax.Array, J: int, offsets: jax.Array) -> Nodes:
    """Build 2J nodes per path from full stored paths.

    X, dW (M, n, d); s_grid (n+1,); offsets (M, J) in [0, m). n must be divisible by J."""
    M, n, d = X.shape
    if n % J:
        raise ValueError(f"n_steps={n} must be divisible by J={J}")
    m = n // J
    starts = jnp.arange(J) * m  # (J,)
    r = starts[None, :] + offsets  # (M, J)
    ds_all = jnp.diff(s_grid)  # (n,)
    # coarse increments W_{s_{j+1}} - W_{s_j}
    dW_coarse = dW.reshape(M, J, m, d).sum(axis=2)  # (M, J, d)
    take = jax.vmap(lambda A, idx: A[idx])  # per path gather
    X_s = X[:, starts]  # (M, J, d)
    X_r = take(X, r)
    dW_r = take(dW, r)
    s_s = jnp.broadcast_to(s_grid[starts], (M, J))
    s_r = s_grid[r]
    ds_r = ds_all[r]
    ds_coarse = jnp.broadcast_to(s_grid[starts + m] - s_grid[starts], (M, J))
    return Nodes(
        X=jnp.concatenate([X_s, X_r], axis=1),
        dW=jnp.concatenate([dW_coarse - m * dW_r, m * dW_r], axis=1),
        s=jnp.concatenate([s_s, s_r], axis=1),
        ds=jnp.concatenate([ds_coarse - m * ds_r, m * ds_r], axis=1),
    )
