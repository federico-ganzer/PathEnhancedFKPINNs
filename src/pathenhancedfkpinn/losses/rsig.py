"""Path-signature residual loss R_sig at one FK point (draft eq. 15).

    a_{w,i} = R_i Phi_w^{(i)} / <p_w, p_w>
    R_sig   = sum_w <p_w, p_w> delta_beta_w^2,   delta_beta_w = E[R Phi_w] / <p_w, p_w>

Estimators over the M paths at one point:
- U-statistic (default): sum_w <p_w,p_w> [(sum_i a)^2 - sum_i a^2] / (M (M-1)). Unbiased at
  every theta. Can be negative; never clamp.
- plug-in: sum_w <p_w,p_w> (mean_i a)^2. Biased upward by (1/M) sum_w <p_w,p_w> Var(a_w).

`word_weights` are the 1/pi_w inclusion weights of word subsampling; None = all words.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _a(R: jax.Array, Phi: jax.Array, norms: jax.Array) -> jax.Array:
    return R[:, None] * Phi / norms


def per_word_ustat(R: jax.Array, Phi: jax.Array, norms: jax.Array) -> jax.Array:
    """Unbiased per-word energies <p_w,p_w> delta_beta_w^2, shape (W,). R (M,), Phi (M, W)."""
    a = _a(R, Phi, norms)
    M = R.shape[0]
    return norms * (a.sum(0) ** 2 - (a**2).sum(0)) / (M * (M - 1))


def per_word_plugin(R: jax.Array, Phi: jax.Array, norms: jax.Array) -> jax.Array:
    return norms * _a(R, Phi, norms).mean(0) ** 2


def rsig_ustat(R, Phi, norms, word_weights=None) -> jax.Array:
    e = per_word_ustat(R, Phi, norms)
    return jnp.sum(e if word_weights is None else e * word_weights)


def rsig_plugin(R, Phi, norms, word_weights=None) -> jax.Array:
    e = per_word_plugin(R, Phi, norms)
    return jnp.sum(e if word_weights is None else e * word_weights)
