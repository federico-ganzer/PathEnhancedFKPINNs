"""Derivative (Sobolev-type) losses at FK points.

    R = mean_i sum_{n=0..N} w_n(t_i) || grad^n u_theta(x_i, t_i) - label_n,i ||_F^2

Arm D : labels = exact derivatives, Parseval weights w_n = tau(t)^n / n!  (R_loc, draft eq. 12).
Arm D': labels exact, tuned weights: family "const" w_n = lambda_n, family "tpow" w_n = lambda_n t^n.
Arm C : labels = MC pathwise derivatives (fkdata.pathwise_labels), level 0 = ybar,
        Parseval weights of the driving noise (problem.tau if defined, else sigma^2 t).

Derivative tensors use nested forward mode (jax.jacfwd), not jet, so U15 does not gate them.
"""

from __future__ import annotations

from math import factorial

import jax
import jax.numpy as jnp


def deriv_tensors(f, x: jax.Array, N: int) -> list[jax.Array]:
    """[f(x), grad f, Hess f, ...] up to order N; order n has shape (d,)*n."""
    out, g = [f(x)], f
    for _ in range(N):
        g = jax.jacfwd(g)
        out.append(g(x))
    return out


def net_derivs(u, params, x, t, N: int) -> list[jax.Array]:
    """Network derivative tensors at points x (P, d), t (P,): list of (P, d^n...) arrays."""
    return jax.vmap(lambda xi, ti: deriv_tensors(lambda y: u(params, y, ti), xi, N))(x, t)


def exact_derivs(problem, x, t, N: int) -> list[jax.Array]:
    return jax.vmap(lambda xi, ti: deriv_tensors(lambda y: problem.exact(y, ti), xi, N))(x, t)


def parseval_weights(tau_t: jax.Array, N: int) -> jax.Array:
    """(P, N+1): tau^n / n!."""
    return jnp.stack([tau_t**n / factorial(n) for n in range(N + 1)], axis=1)


def tuned_weights(t: jax.Array, lambdas, family: str) -> jax.Array:
    lam = jnp.asarray(lambdas, t.dtype)
    if family == "const":
        return jnp.broadcast_to(lam, (t.shape[0], lam.shape[0]))
    if family == "tpow":
        return lam[None, :] * t[:, None] ** jnp.arange(lam.shape[0])[None, :]
    raise ValueError(f"unknown D' family {family!r}")


def noise_tau(problem, t: jax.Array) -> jax.Array:
    """tau(t) for the Parseval weights: the problem's own (heat, OU) or sigma^2 t otherwise."""
    if hasattr(problem, "tau"):
        return problem.tau(t)
    s = jnp.asarray(problem.sigma(), t.dtype)
    s2 = s**2 if s.ndim == 0 else jnp.mean(jnp.diag(s) ** 2 if s.ndim == 2 else s**2)
    return s2 * t


def per_level(u, params, x, t, labels: list[jax.Array], weights: jax.Array) -> jax.Array:
    """(N+1,) mean over points of w_n ||grad^n u - label_n||_F^2."""
    N = len(labels) - 1
    nd = net_derivs(u, params, x, t, N)
    out = []
    for n in range(N + 1):
        diff = (nd[n] - labels[n]).reshape(x.shape[0], -1)
        out.append(jnp.mean(weights[:, n] * jnp.sum(diff**2, axis=1)))
    return jnp.stack(out)


def residual_vector(u, params, x, t, labels, weights) -> jax.Array:
    N = len(labels) - 1
    nd = net_derivs(u, params, x, t, N)
    P = x.shape[0]
    parts = [
        (jnp.sqrt(weights[:, n] / P)[:, None] * (nd[n] - labels[n]).reshape(P, -1)).ravel()
        for n in range(N + 1)
    ]
    return jnp.concatenate(parts)
