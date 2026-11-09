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


def test_blocks_match_dense():
    """Block-built J J^T (5 blocks x 3 rows, row_chunk 2: last row chunk short) equals gn_spectrum."""
    import jax

    from pathenhancedfkpinn.diagnostics.gn_spectrum import Blocks, gn_spectrum_blocks

    params = {"a": jax.random.normal(jax.random.key(1), (5, 3), jnp.float32), "b": jnp.ones(3, jnp.float32)}
    xs = jnp.linspace(-1.0, 1.0, 20, dtype=jnp.float64).reshape(5, 4)

    def fn(p, xb):
        h = jnp.tanh(xb[:, None] * p["a"].sum(0)[None, :] + p["b"])
        return jnp.stack([h.sum(), jnp.sin(h).sum(), (h**2).mean()])

    def r(p):
        return jnp.concatenate([fn(p, xs[b]) for b in range(xs.shape[0])])

    want = gn_spectrum(r, params)
    np.testing.assert_allclose(gn_spectrum_blocks(Blocks(fn, (xs,)), params, row_chunk=2), want, rtol=1e-10, atol=1e-12)


def test_path_arm_kappa_residual_unchanged():
    """Arm B kappa blocks, concatenated, equal the whole-vector residual (vmap over points)."""
    import jax

    from pathenhancedfkpinn.config import resolve
    from pathenhancedfkpinn.diagnostics.gn_spectrum import Blocks, gn_spectrum_blocks
    from pathenhancedfkpinn.keys import derive
    from pathenhancedfkpinn.nets import FieldNet, apply_fn, split
    from pathenhancedfkpinn.payoff.model import model_payoff
    from pathenhancedfkpinn.precision import tree_f64
    from pathenhancedfkpinn.train import build_arm_data, build_kappa, build_points, build_problem

    cfg = resolve(["configs/base.yaml"], ["arm=B", "data.n_coll=200", "data.M=16", "data.dt=0.05",
                                          "data.path_block=16", "net.width=8", "net.depth=1",
                                          "kappa.n_c=6", "kappa.M=8", "kappa.slices=[1.0]"])  # fmt: skip
    problem = build_problem(cfg)
    pts = build_points(problem, cfg, 0)
    lo, hi = problem.lohi_with_time()
    params, static = split(FieldNet.create(problem.d, True, lo, hi, width=8, depth=1, key=derive(0, "init")))
    u = apply_fn(static)
    (spec,) = build_kappa(problem, cfg, build_arm_data(problem, cfg, pts, 0), u, 0).values()
    assert isinstance(spec, Blocks)
    x, t, X, dW, s, Phi, norms = spec.args
    n_c, sigma = x.shape[0], problem.sigma()

    def r_whole(p):
        Yth = jax.vmap(lambda xi, ti, Xi, dWi, si: model_payoff(u, p, xi, ti, Xi, dWi, si, sigma))(x, t, X, dW, s)
        beta = jnp.mean(Yth[..., None] * Phi, axis=1) / norms
        return (jnp.sqrt(norms / n_c) * beta).ravel()

    p64 = tree_f64(params)  # kappa differentiates at float64 parameters
    r_blocks = jnp.concatenate([spec.fn(p64, *[a[b] for a in spec.args]) for b in range(n_c)])
    np.testing.assert_allclose(np.asarray(r_blocks), np.asarray(r_whole(p64)), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(gn_spectrum_blocks(spec, params), gn_spectrum(r_whole, params), rtol=1e-8, atol=1e-14)
