"""Euler-Maruyama simulation of FK paths. Offline, float64.

    dX_s = b(X_s, s) ds + sigma dW_s,  X_0 = x,  s in [0, t]

The network's time argument along a path is t - s (remaining time); that is the caller's job.

Random numbers: paths come in blocks of `block` paths. Block j of the point with key k uses
key fold(k, j), and step i of that block uses fold(fold(k, j), i). Any block can be regenerated
alone, and arms that share (seed, point) share paths (common random numbers).

`euler_scan` is generic: an `accumulate(acc, X, s, ds, dW)` callback sees left-point states and
increments and can stream Ito sums or signatures without storing paths; `emit(X, s, ds, dW)` can
return per-step arrays to store (stacked along a leading step axis).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from pathenhancedfkpinn.keys import fold


def uniform_grid(t, n: int, dtype=jnp.float64) -> jax.Array:
    """s_k = k t / n, k = 0..n. Shape (n+1,)."""
    return jnp.asarray(t, dtype) * jnp.arange(n + 1, dtype=dtype) / n


def graded_grid(t, n: int, dtype=jnp.float64) -> jax.Array:
    """s_k = t (1 - (1 - k/n)^2): steps shrink towards s = t (non-smooth payoffs)."""
    r = jnp.arange(n + 1, dtype=dtype) / n
    return jnp.asarray(t, dtype) * (1.0 - (1.0 - r) ** 2)


def make_grid(t, n: int, mesh: str = "uniform", dtype=jnp.float64) -> jax.Array:
    if mesh == "uniform":
        return uniform_grid(t, n, dtype)
    if mesh == "graded":
        return graded_grid(t, n, dtype)
    raise ValueError(f"unknown mesh {mesh!r}")


def apply_sigma(sigma, dW: jax.Array) -> jax.Array:
    """sigma dW for scalar, diagonal (d,) or full (d, d) sigma. dW (..., d)."""
    sigma = jnp.asarray(sigma, dW.dtype)
    if sigma.ndim == 2:
        return dW @ sigma.T
    return sigma * dW


def brownian_step(key: jax.Array, step, n_blocks: int, block: int, d: int, ds, dtype=jnp.float64):
    """Increments (n_blocks * block, d) for one step; block j uses fold(key, j, step)."""
    def one(j):
        return jax.random.normal(fold(key, j, step), (block, d), dtype)

    z = jax.vmap(one)(jnp.arange(n_blocks)).reshape(n_blocks * block, d)
    return z * jnp.sqrt(jnp.asarray(ds, dtype))


class Paths(NamedTuple):
    """Stored left-point paths at one FK point.

    X  (M, n, d)  states X_{s_k}, k = 0..n-1 (left points)
    dW (M, n, d)  Brownian increments over [s_k, s_{k+1}]
    s  (n+1,)     time grid (shared by all paths)
    XT (M, d)     terminal state X_t
    """

    X: jax.Array
    dW: jax.Array
    s: jax.Array
    XT: jax.Array


def euler_scan(
    drift: Callable,
    sigma,
    x0: jax.Array,
    s_grid: jax.Array,
    key: jax.Array,
    n_paths: int,
    accumulate: Callable | None = None,
    acc0=None,
    emit: Callable | None = None,
    block: int | None = None,
):
    """Euler-Maruyama over the grid `s_grid` (n+1,) from x0 (d,), n_paths paths.

    drift(X, s): X (M, d) -> (M, d). Returns (X_T (M, d), acc, emitted or None)."""
    d = x0.shape[-1]
    dtype = s_grid.dtype
    block = block or n_paths
    if n_paths % block:
        raise ValueError(f"n_paths={n_paths} must be a multiple of block={block}")
    n_blocks = n_paths // block
    X0 = jnp.broadcast_to(jnp.asarray(x0, dtype), (n_paths, d))
    n = s_grid.shape[0] - 1

    def step(carry, k):
        X, acc = carry
        s, ds = s_grid[k], s_grid[k + 1] - s_grid[k]
        dW = brownian_step(key, k, n_blocks, block, d, ds, dtype)
        out = emit(X, s, ds, dW) if emit is not None else None
        if accumulate is not None:
            acc = accumulate(acc, X, s, ds, dW)
        X = X + drift(X, s) * ds + apply_sigma(sigma, dW)
        return (X, acc), out

    (XT, acc), ys = jax.lax.scan(step, (X0, acc0), jnp.arange(n))
    return XT, acc, ys


def simulate_paths(drift, sigma, x0, t, n_steps: int, key, n_paths: int, mesh="uniform", block=None) -> Paths:
    """Store full left-point paths at one FK point."""
    grid = make_grid(t, n_steps, mesh)
    XT, _, (X, dW) = euler_scan(
        drift, sigma, x0, grid, key, n_paths, emit=lambda X, s, ds, dW: (X, dW), block=block
    )
    return Paths(X=jnp.swapaxes(X, 0, 1), dW=jnp.swapaxes(dW, 0, 1), s=grid, XT=XT)


def zero_drift(X, s):
    return jnp.zeros_like(X)
