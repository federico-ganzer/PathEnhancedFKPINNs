"""Gauss-Newton condition number (draft §3.6).

For a data term with residual vector r(theta) (loss = sum r^2), K = J J^T with J = dr/dtheta.
kappa = lambda_max / lambda_min over eigenvalues above cutoff * lambda_max (the resolved band).
Always float64: parameters are cast before differentiating.

Two ways to build K, same eigenvalues:
  gn_spectrum         residual_fn(params) -> (n,); K built a chunk of rows at a time. Each chunk
                      differentiates the whole residual, so use it when r is cheap (point and
                      derivative arms).
  gn_spectrum_blocks  Blocks(fn, args): r = concat_b fn(params, *args_b), block b depending only on
                      its own args (path arms: one block per point, its paths inside). J is built one
                      block at a time and K = J J^T. Peak memory is one block's reverse pass, not
                      chunk x the whole residual (250 GiB for arm B at 128 points x 64 paths x 1000
                      steps through gn_spectrum).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from pathenhancedfkpinn.precision import tree_f64


def _jjt_rows(f: Callable, flat, E):
    """Rows E J J^T for cotangent rows E (c, n): one reverse pass (J^T e) then one forward pass."""
    _, vjp = jax.vjp(f, flat)
    G = jax.vmap(lambda e: vjp(e)[0])(E)  # (c, P): rows of J
    return jax.vmap(lambda g: jax.jvp(f, (flat,), (g,))[1])(G)  # (c, n)


def gn_spectrum(residual_fn: Callable, params, chunk: int = 32) -> np.ndarray:
    """Eigenvalues (descending) of J J^T for residual_fn(params) -> (n,) at float64 params.

    J J^T is built `chunk` rows at a time, so J itself (n x P) is never held in memory and only
    `chunk` reverse passes run at once. Peak memory scales with chunk, not n."""
    p64 = tree_f64(params)
    flat, unravel = ravel_pytree(p64)

    def f(v):
        return residual_fn(unravel(v))

    out = jax.eval_shape(f, flat)
    n, dtype = out.shape[0], out.dtype
    rows = jax.jit(lambda v, E: _jjt_rows(f, v, E))
    K = np.empty((n, n), dtype=np.float64)
    for i0 in range(0, n, chunk):
        idx = np.arange(i0, i0 + chunk)
        valid = idx < n  # last chunk padded with zero rows so every call has the same shape
        E = jax.nn.one_hot(jnp.asarray(np.minimum(idx, n - 1)), n, dtype=dtype) * jnp.asarray(valid, dtype)[:, None]
        K[i0 : i0 + int(valid.sum())] = np.asarray(rows(flat, E), dtype=np.float64)[valid]
    K = 0.5 * (K + K.T)  # exact symmetry; rows agree to rounding
    return np.linalg.eigvalsh(K)[::-1]


@dataclass(frozen=True)
class Blocks:
    """Residual vector concat_b fn(params, *[a[b] for a in args]); fn returns (r,) per block."""

    fn: Callable
    args: tuple


def _block_jacobian(fn, unravel, row_chunk, flat, args):
    """(r, P) Jacobian of one block: one forward pass, reverse passes row_chunk rows at a time."""
    out, vjp = jax.vjp(lambda v: fn(unravel(v), *args), flat)
    E = jnp.eye(out.shape[0], dtype=out.dtype)
    return jax.lax.map(lambda e: vjp(e)[0], E, batch_size=row_chunk)


def gn_spectrum_blocks(spec: Blocks, params, row_chunk: int = 8) -> np.ndarray:
    """Eigenvalues (descending) of J J^T for a block residual, J built block by block."""
    p64 = tree_f64(params)
    flat, unravel = ravel_pytree(p64)
    jac = jax.jit(lambda v, a: jax.lax.map(partial(_block_jacobian, spec.fn, unravel, row_chunk, v), a))
    J = jac(flat, tuple(spec.args))  # (n_blocks, r, P)
    J = J.reshape(-1, J.shape[-1])
    K = np.asarray(jax.jit(lambda A: A @ A.T)(J), dtype=np.float64)
    K = 0.5 * (K + K.T)
    return np.linalg.eigvalsh(K)[::-1]


def condition_from_spectrum(ev: np.ndarray, cutoff: float = 1e-8) -> tuple[float, int]:
    """(kappa, number of eigenvalues kept)."""
    lmax = float(ev[0])
    if lmax <= 0:
        return float("inf"), 0
    kept = ev[ev > cutoff * lmax]
    return lmax / float(kept[-1]), int(kept.size)


def gn_condition(residual_fn: Callable, params, cutoff: float = 1e-8) -> float:
    return condition_from_spectrum(gn_spectrum(residual_fn, params), cutoff)[0]


def jnp_f64(x):
    return jnp.asarray(x, jnp.float64)
