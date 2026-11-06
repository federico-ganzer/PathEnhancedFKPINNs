"""gn_condition on a linear model with known Jacobian."""

import jax.numpy as jnp
import numpy as np

from pathenhancedfkpinn.diagnostics.gn_spectrum import condition_from_spectrum, gn_condition, gn_spectrum


def test_linear_model_exact_kappa():
    # r(theta) = A theta - y  ->  J = A,  K = A A^T
    A = np.diag([3.0, 1.0, 0.5])
    A = np.vstack([A, np.zeros((1, 3))])  # 4 residuals, rank 3 -> one zero eigenvalue
    params = {"w": jnp.zeros(3, jnp.float32)}

    def r(p):
        return jnp.asarray(A) @ p["w"] - 1.0

    ev = gn_spectrum(r, params)
    np.testing.assert_allclose(ev[:3], [9.0, 1.0, 0.25], rtol=1e-12)
    assert abs(ev[3]) < 1e-12
    np.testing.assert_allclose(gn_condition(r, params), 36.0, rtol=1e-12)


def test_cutoff_drops_small_eigenvalues():
    ev = np.array([1.0, 1e-3, 1e-9, 0.0])
    k, n = condition_from_spectrum(ev, cutoff=1e-8)
    assert n == 2 and k == 1e3
    k, n = condition_from_spectrum(ev, cutoff=1e-10)
    assert n == 3 and np.isclose(k, 1e9)


def test_chunked_matches_dense_jacobian():
    """Chunked J J^T rows (28 residuals, chunk 5: last chunk padded) equal the dense jacrev spectrum."""
    import jax
    from jax.flatten_util import ravel_pytree

    from pathenhancedfkpinn.precision import tree_f64

    params = {"a": jax.random.normal(jax.random.key(0), (5, 3), jnp.float32), "b": jnp.ones(3, jnp.float32)}
    x = jnp.linspace(-1.0, 1.0, 7, dtype=jnp.float64)

    def r(p):
        h = jnp.tanh(x[:, None] * p["a"].sum(0)[None, :] + p["b"])
        return jnp.concatenate([h.ravel(), jnp.sin(h).sum(1)])

    flat, unravel = ravel_pytree(tree_f64(params))
    J = jax.jacrev(lambda f: r(unravel(f)))(flat)
    want = np.linalg.eigvalsh(np.asarray(J @ J.T))[::-1]
    np.testing.assert_allclose(gn_spectrum(r, params, chunk=5), want, rtol=1e-10, atol=1e-12)
