"""jax.experimental.jet against nested jax.grad in float64 (gate for any jet use)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pathenhancedfkpinn import derivs
from pathenhancedfkpinn.nets import FieldNet, make_mlp


def _f64_mlp(in_size, key):
    mlp = make_mlp(in_size, width=32, depth=3, key=key)
    return jax.tree.map(lambda a: a.astype(jnp.float64) if eqx.is_inexact_array(a) else a, mlp)


def test_jet_vs_nested_grad_1d():
    mlp = _f64_mlp(1, jax.random.key(0))

    def f(x):
        return mlp(jnp.atleast_1d(x))

    g1 = jax.grad(f)
    g2 = jax.grad(g1)
    g3 = jax.grad(g2)
    for x in [-0.7, 0.1, 0.55]:
        x = jnp.float64(x)
        jets = derivs.directional(f, x, jnp.float64(1.0), 3)
        for a, b in zip(jets, [g1(x), g2(x), g3(x)]):
            np.testing.assert_allclose(float(a), float(b), rtol=1e-10, atol=1e-12)


def test_jet_closed_form():
    def f(x):
        return jnp.sin(3.0 * x) * jnp.exp(0.5 * x)

    got = derivs.directional(f, jnp.float64(0.3), jnp.float64(1.0), 3)
    s, c, e = np.sin(0.9), np.cos(0.9), np.exp(0.15)
    # f = Im exp((0.5 + 3i) x): derivatives are Im((0.5 + 3i)^n exp(...)).
    exact = [e * (3 * c + 0.5 * s), e * (3 * c - 8.75 * s), e * (-24.75 * c - 13.375 * s)]
    np.testing.assert_allclose([float(v) for v in got], exact, rtol=1e-12)


def test_laplacian_d10_matches_coordinatewise():
    mlp = _f64_mlp(10, jax.random.key(1))
    x = jax.random.normal(jax.random.key(2), (10,), jnp.float64) * 0.3
    tr = float(jnp.trace(jax.hessian(mlp)(x)))
    eye = jnp.eye(10, dtype=jnp.float64)
    lap_jet = sum(float(derivs.directional(mlp, x, eye[i], 2)[1]) for i in range(10))
    np.testing.assert_allclose(float(derivs.laplacian(mlp, x)), tr, rtol=1e-10)
    np.testing.assert_allclose(lap_jet, tr, rtol=1e-10)


def test_hutchinson_exact_for_diagonal_hessian():
    def f(x):
        return jnp.sum(jnp.arange(1.0, 6.0) * x**2)  # Laplacian = 2 * 15 = 30

    est = derivs.laplacian_hutchinson(f, jnp.zeros(5, jnp.float64), jax.random.key(0), n_probes=4)
    np.testing.assert_allclose(float(est), 30.0, rtol=1e-12)


def test_fieldnet_dtypes_and_scaling():
    net = FieldNet.create(d=2, time=True, lo=[-1, -1, 0], hi=[1, 1, 1], key=jax.random.key(0))
    leaves = [a for a in jax.tree.leaves(net) if eqx.is_inexact_array(a)]
    assert all(a.dtype == jnp.float32 for a in leaves)
    out = net(jnp.array([0.1, -0.2], jnp.float64), 0.5)
    assert out.dtype == jnp.float64 and out.shape == ()
