"""Precision policy.

float64: path accumulators (Ito sums, time integrals), signature features, basis coefficients
and norms, labels, per-path residuals R, Gauss-Newton spectra.
float32: network parameters and network inputs, with full-precision float32 matrix products
(`jax_default_matmul_precision = "highest"`; the GPU default would be TF32).

The package enables `jax_enable_x64` on import, so jnp defaults to float64. Always pass a dtype.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

F32 = jnp.float32
F64 = jnp.float64


def f64(x) -> jax.Array:
    return jnp.asarray(x, dtype=F64)


def f32(x) -> jax.Array:
    return jnp.asarray(x, dtype=F32)


def tree_f64(tree):
    """Cast every floating leaf of a pytree to float64 (used for kappa checkpoints)."""
    return jax.tree.map(
        lambda a: a.astype(F64) if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating) else a,
        tree,
    )
