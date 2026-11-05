"""MC orthogonality of the basis, and end-to-end R_sig on heat (zero-residual subspace).

Expected numbers are closed-form word energies. Fast variants use fewer paths and looser tolerances; the full
path counts run with `pytest -m slow`.
"""

from math import factorial, pi

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pathenhancedfkpinn.keys import derive, fold
from pathenhancedfkpinn.losses.rsig import per_word_ustat
from pathenhancedfkpinn.sde.simulate import euler_scan, uniform_grid, zero_drift
from pathenhancedfkpinn.signature.ito_features import features, stream_finish, stream_init, stream_step
from pathenhancedfkpinn.signature.words import word_tables


def _ito_features_heat(tables, t, dt, M, chunk, key, backend, extra=None):
    """Stream Brownian paths (b = 0, sigma = 1 here; callers scale) and return S plus any
    extra accumulator. extra = (acc0_fn, accumulate) for additional streamed quantities."""
    n = int(round(t / dt))
    grid = uniform_grid(t, n)

    @jax.jit
    def run_chunk(k):
        acc0 = (stream_init(tables, chunk, backend), extra[0](chunk) if extra else None)

        def accumulate(acc, X, s, ds, dW):
            S, ex = acc
            inc = jnp.concatenate([jnp.broadcast_to(ds, (chunk, 1)), dW], axis=1)
            S = stream_step(S, inc, tables, backend)
            ex = extra[1](ex, X, s, ds, dW) if extra else None
            return S, ex

        XT, (S, ex), _ = euler_scan(zero_drift, 1.0, jnp.zeros(tables.d), grid, k, chunk, accumulate, acc0)
        return stream_finish(S, tables, backend), XT, ex

    outs = [run_chunk(fold(key, c)) for c in range(M // chunk)]
    return [jnp.concatenate([o[i] for o in outs]) if outs[0][i] is not None else None for i in range(3)]


# --- MC orthogonality --------------------------------------------------------------------------


def _mc_orthogonality(M, chunk, diag_tol, corr_tol, backend):
    T = word_tables(1, 3, 2)
    t, dt = 0.7, 7e-4
    S, _, _ = _ito_features_heat(T, t, dt, M, chunk, derive(0, "test", 2), backend)
    Phi, norms = features(S, T, t, include_empty=False)
    G = (Phi.T @ Phi) / M
    diag = jnp.diag(G)
    rel = jnp.abs(diag / norms - 1)
    corr = G / jnp.sqrt(jnp.outer(diag, diag))
    off = jnp.abs(corr - jnp.diag(jnp.diag(corr))).max()
    assert float(rel.max()) < diag_tol, f"diag rel err {float(rel.max()):.3f}"
    assert float(off) < corr_tol, f"max off-diag corr {float(off):.3f}"


@pytest.mark.parametrize("backend", ["signax", "sparse"])
def test_mc_orthogonality_fast(backend):
    _mc_orthogonality(M=40_000, chunk=20_000, diag_tol=0.10, corr_tol=0.06, backend=backend)


@pytest.mark.slow
def test_mc_orthogonality_full():
    _mc_orthogonality(M=400_000, chunk=50_000, diag_tol=0.03, corr_tol=0.03, backend="signax")


# --- end-to-end R_sig on heat ------------------------------------------------------------------

NU, T_, X_, EPS, K = 0.05, 0.5, 0.3, 0.1, 3
SIG = np.sqrt(2 * NU)


def _u(y, s):
    return jnp.exp(-NU * pi**2 * s) * jnp.sin(pi * y)


def _e(y, s):
    return EPS * jnp.exp(-NU * (K * pi) ** 2 * s) * jnp.sin(K * pi * y)


def _ux(y, s):
    return pi * jnp.exp(-NU * pi**2 * s) * jnp.cos(pi * y) + EPS * K * pi * jnp.exp(
        -NU * (K * pi) ** 2 * s
    ) * jnp.cos(K * pi * y)


def _exact_energy(w):
    if 0 in w:
        return 0.0
    n = len(w)
    c = EPS * (K * pi) ** n * np.exp(-NU * (K * pi) ** 2 * T_) * np.sin(K * pi * X_ + n * pi / 2)
    return (2 * NU * T_) ** n / factorial(n) * c**2


def _heat_energies(M, chunk, backend):
    """u_theta = exact heat solution of a perturbed IC, so R = e0(X_t) + Euler defect."""
    T = word_tables(1, 3, 2)
    # Paths are standard BM W; X = x + SIG W. Ito sum of grad u_theta . sigma dW streamed alongside.
    extra = (
        lambda m: jnp.zeros(m, jnp.float64),
        lambda ito, Wk, s, ds, dW: ito + SIG * _ux(X_ + SIG * Wk[:, 0], T_ - s) * dW[:, 0],
    )
    S, WT, ito = _ito_features_heat(T, T_, 1e-3, M, chunk, derive(0, "test", 3), backend, extra)
    R = (_u(X_, T_) + _e(X_, T_)) + ito - _u(X_ + SIG * WT[:, 0], 0.0)
    Phi, norms = features(S, T, T_)  # includes the empty word at column 0
    E = np.asarray(per_word_ustat(R, Phi, norms))
    words = [()] + list(T.feats)
    return dict(zip(words, E)), float(jnp.std(R))


def _check_heat_energies(E, stdR, rtol, time_tol):
    for w, ref in [((1,), 4.732e-4), ((1, 1), 1.109e-4), ((1, 1, 1), 1.556e-3)]:
        assert abs(_exact_energy(w) - ref) / ref < 2e-3  # closed form matches reference table
        assert abs(E[w] - ref) / ref < rtol, f"word {w}: {E[w]:.4e} vs {ref:.4e}"
    time_words = [w for w in E if 0 in w]
    assert len(time_words) == 16
    assert max(abs(E[w]) for w in time_words) < time_tol
    assert abs(E[()] - 1.125e-5) / 1.125e-5 < 0.25  # level 0 = e(x,t)^2
    assert abs(stdR - 0.071) < 0.004


@pytest.mark.parametrize("backend", ["signax", "sparse"])
def test_heat_rsig_fast(backend):
    E, stdR = _heat_energies(M=40_000, chunk=20_000, backend=backend)
    _check_heat_energies(E, stdR, rtol=0.2, time_tol=2e-5)


@pytest.mark.slow
def test_heat_rsig_full():
    E, stdR = _heat_energies(M=200_000, chunk=50_000, backend="signax")
    # Time words have expectation 0; at 2e5 paths the max is 2.55e-6 (pure MC noise).
    # Threshold 5e-6; logged in notes/session_log.md.
    _check_heat_energies(E, stdR, rtol=0.05, time_tol=5e-6)
