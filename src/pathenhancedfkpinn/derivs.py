"""Derivatives of scalar fields f(x), x (d,).

- First derivatives and Hessian-vector products: jax.grad, jvp-of-grad.
- Higher directional derivatives: Taylor mode, `jax.experimental.jet`. The input series
  (v, 0, ...) returns f'(x)[v], f''(x)[v,v], ... in derivative convention (not divided by
  factorials). `jet` is experimental: unit test U15 must pass before any arm uses it.
- Laplacian: exact (sum over coordinates) for d <= 10; Hutchinson with Rademacher probes for
  large d, in arm P only (say so in results).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import jet


def directional(f: Callable, x: jax.Array, v: jax.Array, order: int) -> list[jax.Array]:
    """[D f(x)[v], D^2 f(x)[v,v], ..., D^order f(x)[v,...,v]] via jet."""
    series = [v] + [jnp.zeros_like(v)] * (order - 1)
    _, coeffs = jet.jet(f, (x,), (series,))
    return list(coeffs)


def hvp(f: Callable, x: jax.Array, v: jax.Array) -> jax.Array:
    """Hessian-vector product H(x) v, shape (d,)."""
    return jax.jvp(jax.grad(f), (x,), (v,))[1]


def second_directional(f: Callable, x: jax.Array, v: jax.Array) -> jax.Array:
    return jnp.vdot(v, hvp(f, x, v))


def laplacian(f: Callable, x: jax.Array) -> jax.Array:
    """Exact Laplacian: sum_i e_i^T H e_i (d jvp-of-grad passes, vmapped)."""
    eye = jnp.eye(x.shape[-1], dtype=x.dtype)
    return jnp.sum(jax.vmap(lambda e: second_directional(f, x, e))(eye))


def weighted_laplacian(f: Callable, x: jax.Array, a: jax.Array) -> jax.Array:
    """sum_i a_i d^2 f / dx_i^2 (diagonal diffusion, a = sigma^2 / 2 per coordinate)."""
    eye = jnp.eye(x.shape[-1], dtype=x.dtype)
    return jnp.sum(a * jax.vmap(lambda e: second_directional(f, x, e))(eye))


def laplacian_hutchinson(f: Callable, x: jax.Array, key: jax.Array, n_probes: int = 16) -> jax.Array:
    """Unbiased stochastic Laplacian with Rademacher probes (arm P at d = 50 only)."""
    z = jax.random.rademacher(key, (n_probes, x.shape[-1]), dtype=x.dtype)
    return jnp.mean(jax.vmap(lambda v: second_directional(f, x, v))(z))
