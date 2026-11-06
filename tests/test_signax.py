"""Signax-based Ito features vs the sparse recursion, Signax layout, legacy Stratonovich
reference, and the exact discrete Ito-Stratonovich identity."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import signax
from signax import utils as sxu

from pathenhancedfkpinn.signature import (
    dense_index,
    ito_signature,
    stratonovich_signature,
    time_augmented_increments,
    word_tables,
)

REF = Path(__file__).resolve().parents[1] / "legacy" / "torch" / "reference" / "strat_signature.json"


def _increments(d, n=300, M=32, dt=1e-3, seed=0):
    dW = jax.random.normal(jax.random.key(seed), (n, M, d), jnp.float64) * np.sqrt(dt)
    return time_augmented_increments(dW, dt)


@pytest.mark.parametrize("d,N,N0", [(1, 3, 2), (2, 2, 2), (1, 5, 2), (3, 2, 1)])
def test_signax_equals_sparse(d, N, N0):
    T = word_tables(d, N, N0)
    dWh = _increments(d)
    S_dense = ito_signature(dWh, T, "signax")
    S_sparse = ito_signature(dWh, T, "sparse")
    assert S_dense.dtype == jnp.float64
    np.testing.assert_allclose(S_dense, S_sparse, rtol=0, atol=1e-12)


def test_n0_empty_word_only():
    T = word_tables(2, 0, 2)
    S = ito_signature(_increments(2), T)
    assert S.shape == (32, 1) and bool(jnp.all(S == 1.0))


@pytest.mark.parametrize("d,depth", [(1, 4), (2, 3)])
def test_dense_index_matches_signax_layout(d, depth):
    n = d + 1
    L = sum(n**k for k in range(1, depth + 1))
    flat = jnp.arange(L, dtype=jnp.float64)
    levels = sxu.unravel_signature(flat, n, depth)
    import itertools

    for k in range(1, depth + 1):
        for w in itertools.product(range(n), repeat=k):
            assert int(levels[k - 1][w]) == dense_index(w, d)


def test_stratonovich_matches_legacy_torch():
    if not REF.exists():
        pytest.skip("run legacy/torch/scripts/make_strat_reference.py first")
    ref = json.loads(REF.read_text())
    path = jnp.asarray(ref["path"], jnp.float64)
    sig = stratonovich_signature(path, ref["depth"])
    np.testing.assert_allclose(sig, np.asarray(ref["signature"]), rtol=1e-10, atol=1e-12)


def test_ito_stratonovich_identity_word_11():
    """Strat S^{11} = W_t^2 / 2, left-point Ito S^{11} = (W_t^2 - sum dW^2) / 2, per path."""
    d, n, M, dt = 1, 200, 16, 1e-3
    dWh = _increments(d, n=n, M=M, dt=dt, seed=3)
    T = word_tables(1, 2, 0)  # words 1, 11
    S_ito = ito_signature(dWh, T, "signax")
    j11 = 1 + T.closure.index((1, 1))
    path = jnp.concatenate([jnp.zeros((M, 1, 2)), jnp.cumsum(jnp.moveaxis(dWh, 0, 1), axis=1)], axis=1)
    S_str = signax.signature(path, 2)
    strat11 = S_str[:, dense_index((1, 1), 1)]
    qv = 0.5 * jnp.sum(dWh[..., 1] ** 2, axis=0)
    np.testing.assert_allclose(strat11 - S_ito[:, j11], qv, rtol=1e-10, atol=1e-13)


def test_iisignature_if_available():
    iisig = pytest.importorskip("iisignature")
    dWh = _increments(2, n=100, M=4, seed=5)
    path = np.concatenate([np.zeros((4, 1, 3)), np.cumsum(np.moveaxis(np.asarray(dWh), 0, 1), axis=1)], axis=1)
    ref = np.stack([iisig.sig(p, 3) for p in path])
    np.testing.assert_allclose(stratonovich_signature(jnp.asarray(path), 3), ref, rtol=1e-9, atol=1e-12)
