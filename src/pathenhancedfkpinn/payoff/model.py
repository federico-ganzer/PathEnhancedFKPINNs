"""Model payoff Y^theta. Online, differentiable in theta.

    Y^theta = u(x, t) + sum_k [ h(X_k, t - s_k) ds_k + grad_sigma u(X_k, t - s_k) . dW_k ]
    h = b_bar . grad u - F(., u, grad_sigma u)          (zero for linear problems with b_bar = 0)

Stored paths are data (they do not depend on theta), so every (path, step) term is independent:
vmap over paths and steps, no scan. grad u . sigma dW is one forward-mode JVP with tangent
sigma dW, so its cost does not grow with d.

Horvitz-Thompson nodes use the same call (the HT estimator as a weighted sum over 2J nodes per path):
pass node states, effective increments and effective ds (see payoff/ht.py). `s` may then differ
per path, shape (M, n).

path_chunk: process paths in chunks under jax.checkpoint so reverse mode stores one chunk at a
time (needed for full sums: 1024 paths x 500 steps ran out of memory unchunked on CPU).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def _sigma_dw(sigma, dWk):
    sigma = jnp.asarray(sigma, dWk.dtype)
    return sigma @ dWk if sigma.ndim == 2 else sigma * dWk


def model_payoff(
    u: Callable,
    params,
    x: jax.Array,
    t,
    X: jax.Array,
    dW: jax.Array,
    s: jax.Array,
    sigma,
    ds: jax.Array | None = None,
    h: Callable | None = None,
    path_chunk: int | None = None,
) -> jax.Array:
    """Y^theta per path, shape (M,).

    u(params, y, tau) -> scalar. x (d,), t scalar. X, dW (M, n, d) left-point states and
    increments; s (n,) or (M, n) left-point times; ds like s (needed only with h).
    h(params, y, tau) -> scalar generator term (None for linear problems with b_bar = 0)."""
    M, n = X.shape[0], X.shape[1]
    s = jnp.broadcast_to(s, (M, n))
    if h is not None:
        if ds is None:
            raise ValueError("generator term h needs ds")
        ds = jnp.broadcast_to(ds, (M, n))
    else:
        ds = jnp.zeros((M, n), X.dtype)

    def term(Xk, dWk, sk, dsk):
        tau = t - sk
        ito = jax.jvp(lambda y: u(params, y, tau), (Xk,), (_sigma_dw(sigma, dWk),))[1]
        if h is None:
            return ito
        return ito + h(params, Xk, tau) * dsk

    def per_path(Xp, dWp, sp, dsp):
        return jax.vmap(term)(Xp, dWp, sp, dsp).sum()

    if path_chunk is None or path_chunk >= M:
        path_sum = jax.vmap(per_path)(X, dW, s, ds)
    else:
        if M % path_chunk:
            raise ValueError(f"M={M} must be a multiple of path_chunk={path_chunk}")
        body = jax.checkpoint(jax.vmap(per_path))

        def split(A):
            return A.reshape(M // path_chunk, path_chunk, *A.shape[1:])

        path_sum = jax.lax.map(lambda a: body(*a), (split(X), split(dW), split(s), split(ds))).reshape(M)
    return u(params, x, t) + path_sum
