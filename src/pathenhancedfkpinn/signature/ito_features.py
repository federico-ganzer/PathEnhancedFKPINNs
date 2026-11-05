"""Ito signature features of the time-augmented Brownian path.

Left-point Ito recursion on the simulation grid:   S^{va}_{k+1} = S^{va}_k + S^v_k dW_hat^a_k,
which is the tensor-algebra product  S_{k+1} = S_k (x) (1 + dW_hat_k).

Two backends give identical numbers (unit test U18):
- "signax": dense recursion with `signax.tensor_ops.addcmul` on all (d+1)^k words up to `depth`,
  then the closure words are gathered. Default while the dense size is moderate.
- "sparse": gather recursion over the prefix closure only. Needed for large d
  (d = 50, N = 2, N0 = 2 has ~6.8M dense terms per path) and for word subsampling.

Both return S with shape (M, K+1): column 0 = empty word (= 1), column j+1 = tables.closure[j].
Features are float64 and are never differentiated (offline data side).
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from signax import tensor_ops

from pathenhancedfkpinn.signature.words import WordTables, dense_size

Backend = Literal["auto", "signax", "sparse"]

# Above this many dense terms per path, "auto" uses the sparse recursion.
DENSE_LIMIT = 100_000


def choose_backend(tables: WordTables, backend: Backend = "auto") -> str:
    if backend != "auto":
        return backend
    return "signax" if dense_size(tables.d, tables.depth) <= DENSE_LIMIT else "sparse"


# --- sparse backend ----------------------------------------------------------------------------


def sparse_init(M: int, n_cols: int, dtype=jnp.float64) -> jax.Array:
    return jnp.zeros((M, n_cols), dtype).at[:, 0].set(1.0)


def sparse_step(S: jax.Array, inc: jax.Array, parent, letter) -> jax.Array:
    """One left-point step. S (M, K+1), inc (M, d+1) with column 0 = ds."""
    return S + (S[:, parent] * inc[:, letter]).at[:, 0].set(0.0)


def ito_signature_sparse(dWhat: jax.Array, tables: WordTables) -> jax.Array:
    """dWhat (n_steps, M, d+1), column 0 = ds. Returns S (M, K+1)."""
    parent, letter = jnp.asarray(tables.parent), jnp.asarray(tables.letter)
    S0 = sparse_init(dWhat.shape[1], parent.shape[0], dWhat.dtype)
    return jax.lax.scan(lambda S, inc: (sparse_step(S, inc, parent, letter), None), S0, dWhat)[0]


# --- signax backend ---------------------------------------------------------------------------


def dense_init(n: int, depth: int, dtype=jnp.float64) -> list[jax.Array]:
    """Levels 1..depth of the identity element (the level-0 term 1 is implicit)."""
    return [jnp.zeros((n,) * k, dtype) for k in range(1, depth + 1)]


def dense_step(levels: list[jax.Array], inc: jax.Array) -> list[jax.Array]:
    """levels (x) (1 + inc), one path. Every level updates from the previous step's values."""
    out = [levels[0] + inc]
    for k in range(1, len(levels)):
        out.append(tensor_ops.addcmul(levels[k], levels[k - 1], inc))
    return out


def ito_signature_dense_flat(dWhat_path: jax.Array, depth: int) -> jax.Array:
    """One path: dWhat_path (n_steps, d+1) -> flat Signax layout (sum_k (d+1)^k,)."""
    n = dWhat_path.shape[-1]
    levels = jax.lax.scan(
        lambda L, inc: (dense_step(L, inc), None), dense_init(n, depth, dWhat_path.dtype), dWhat_path
    )[0]
    return jnp.concatenate([lv.ravel() for lv in levels])


def ito_signature_signax(dWhat: jax.Array, tables: WordTables) -> jax.Array:
    """dWhat (n_steps, M, d+1). Returns S (M, K+1) in closure order, via the dense recursion."""
    flat = jax.vmap(ito_signature_dense_flat, in_axes=(1, None))(dWhat, tables.depth)
    S = flat[:, jnp.asarray(tables.closure_dense_idx)]
    return jnp.concatenate([jnp.ones((S.shape[0], 1), S.dtype), S], axis=1)


# --- streaming API (no stored increments) ----------------------------------------------------


def stream_init(tables: WordTables, M: int, backend: Backend = "auto", dtype=jnp.float64):
    """Signature state for M paths, updated one step at a time inside a simulation scan."""
    if not tables.closure:
        return jnp.ones((M, 1), dtype)
    if choose_backend(tables, backend) == "signax":
        return [jnp.zeros((M,) + (tables.d + 1,) * k, dtype) for k in range(1, tables.depth + 1)]
    return sparse_init(M, len(tables.closure) + 1, dtype)


def stream_step(state, inc: jax.Array, tables: WordTables, backend: Backend = "auto"):
    """inc (M, d+1), column 0 = ds."""
    if not tables.closure:
        return state
    if choose_backend(tables, backend) == "signax":
        return jax.vmap(dense_step)(state, inc)
    return sparse_step(state, inc, jnp.asarray(tables.parent), jnp.asarray(tables.letter))


def stream_finish(state, tables: WordTables, backend: Backend = "auto") -> jax.Array:
    """State -> S (M, K+1) in closure order."""
    if not tables.closure or choose_backend(tables, backend) == "sparse":
        return state
    flat = jnp.concatenate([lv.reshape(lv.shape[0], -1) for lv in state], axis=1)
    S = flat[:, jnp.asarray(tables.closure_dense_idx)]
    return jnp.concatenate([jnp.ones((S.shape[0], 1), S.dtype), S], axis=1)


# --- public API -------------------------------------------------------------------------------


def ito_signature(dWhat: jax.Array, tables: WordTables, backend: Backend = "auto") -> jax.Array:
    """Ito signature on the closure words. dWhat (n_steps, M, d+1) float64; column 0 = ds."""
    if not tables.closure:  # N = 0: only the empty word
        return jnp.ones((dWhat.shape[1], 1), dWhat.dtype)
    b = choose_backend(tables, backend)
    if b == "signax":
        return ito_signature_signax(dWhat, tables)
    if b == "sparse":
        return ito_signature_sparse(dWhat, tables)
    raise ValueError(f"unknown backend {backend!r}")


def features(
    S: jax.Array, tables: WordTables, t, include_empty: bool = True
) -> tuple[jax.Array, jax.Array]:
    """Orthogonal features Phi and norms <p_w, p_w>(t) at payoff time t.

    include_empty=True (default) prepends the empty word: Phi[:, 0] = 1, norm 1. Its error
    coefficient is delta beta_empty = E[R], the level-0 term (FK point error with the martingale
    control variate, arm B_0). Shapes: Phi (M, W+1), norms (W+1,); without it (M, W), (W,)."""
    C, zc = jnp.asarray(tables.C), jnp.asarray(tables.zc)
    nrm1, tpow = jnp.asarray(tables.nrm1), jnp.asarray(tables.tpow)
    t = jnp.asarray(t, S.dtype)
    Phi, norms = S @ (C * t**zc), nrm1 * t**tpow
    if include_empty:
        Phi = jnp.concatenate([jnp.ones((S.shape[0], 1), S.dtype), Phi], axis=1)
        norms = jnp.concatenate([jnp.ones((1,), S.dtype), norms])
    return Phi, norms


def time_augmented_increments(dW: jax.Array, ds) -> jax.Array:
    """dW (n_steps, M, d) and ds scalar, (n_steps,) or (n_steps, M) -> dWhat (n_steps, M, d+1)."""
    ds = jnp.asarray(ds, dW.dtype)
    if ds.ndim == 1:
        ds = ds[:, None]  # per-step, shared by all paths
    ds = jnp.broadcast_to(ds, dW.shape[:-1])
    return jnp.concatenate([ds[..., None], dW], axis=-1)


def stratonovich_signature(path: jax.Array, depth: int) -> jax.Array:
    """Piecewise-linear (Stratonovich) signature via `signax.signature`; validation only.
    path (..., n_points, channels) -> (..., sum_k channels^k), Signax flat layout."""
    import signax

    return signax.signature(path, depth)


def dense_layout_check(tables: WordTables) -> np.ndarray:
    """Closure words' dense indices (for tests against signax.utils.term_at)."""
    return np.asarray(tables.closure_dense_idx)
