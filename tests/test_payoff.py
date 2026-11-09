"""Same-path cancellation at the exact solution, Horvitz-Thompson unbiasedness through
`model_payoff`, and residual sanity checks for the Stage 1 problems."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pathenhancedfkpinn.keys import derive, fold
from pathenhancedfkpinn.losses.rsig import rsig_plugin, rsig_ustat
from pathenhancedfkpinn.payoff import data_payoff, draw_offsets, ht_nodes, model_payoff
from pathenhancedfkpinn.problems import OU, Heat, Manufactured
from pathenhancedfkpinn.sde.simulate import simulate_paths
from pathenhancedfkpinn.signature import features, ito_signature, time_augmented_increments, word_tables

HEAT = Heat(d=1, nu=0.05, k=1.0)
X0, T0 = jnp.array([0.3]), 0.5


def _exact_u(problem):
    return lambda params, y, tau: problem.exact(y, tau)


def _perturbed_u(params, y, tau):
    """Exact heat solution of a perturbed IC (as in tests/test_heat_mc.py): solves the PDE, wrong at t = 0."""
    e = 0.1 * jnp.exp(-0.05 * (3 * jnp.pi) ** 2 * tau) * jnp.sin(3 * jnp.pi * y[0])
    return HEAT.exact(y, tau) + e


def _paths(problem, dt, M, seed=0, x=X0, t=T0):
    n = int(round(t / dt))
    return simulate_paths(lambda X, s: jax.vmap(problem.b)(X), problem.sigma(), x, t, n,
                          derive(seed, "test", 4), M, block=min(M, 1000))  # fmt: skip


# --- problems ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "problem",
    [HEAT, Manufactured(k=4), Manufactured(k=4, variant="two_mode_heat"), OU(), OU(d=2, box=((-1.5, 1.5),) * 2)],
)
def test_exact_solutions_have_zero_residual(problem):
    xs, ts = problem.sample_interior(jax.random.key(0), 64)
    r = jax.vmap(lambda x, t: problem.residual(problem.exact, x, t))(xs, ts)
    assert float(jnp.abs(r).max()) < 1e-9


# --- same-path cancellation --------------------------------------------------------------------


def test_cancellation_scales_as_sqrt_dt():
    u = _exact_u(HEAT)
    T = word_tables(1, 2, 1)
    stds, plugins = [], []
    for dt in [4e-3, 1e-3, 2.5e-4]:
        P = _paths(HEAT, dt, M=2000)
        ds = jnp.diff(P.s)
        Y = data_payoff(HEAT, P.X, P.XT, ds)
        Yth = model_payoff(u, None, X0, T0, P.X, P.dW, P.s[:-1], HEAT.sigma())
        R = Yth - Y
        S = ito_signature(time_augmented_increments(jnp.swapaxes(P.dW, 0, 1), ds), T)
        Phi, norms = features(S, T, T0)
        stds.append(float(jnp.std(R)))
        plugins.append(float(rsig_plugin(R, Phi, norms)))
        assert float(jnp.std(R)) < 0.1 * float(jnp.std(Y))
    ratios = np.array(stds[:-1]) / np.array(stds[1:])
    np.testing.assert_allclose(ratios, 2.0, rtol=0.15)  # sqrt(4)
    assert plugins[2] < 0.4 * plugins[1] < 0.16 * plugins[0]  # plug-in R_sig -> 0 like dt


def test_path_chunk_same_values_and_grads():
    """path_chunk (checkpointed chunks of paths) changes memory only, not Y^theta or its gradient."""
    P = _paths(HEAT, 0.05, M=16)
    params = {"w": jnp.array([1.3, -0.7], jnp.float64), "c": jnp.array(0.2, jnp.float64)}

    def u(p, y, tau):
        return jnp.tanh(p["w"][0] * y[0] + p["w"][1] * tau) + p["c"] * y[0] ** 2

    def Y(p, chunk):
        return model_payoff(u, p, X0, T0, P.X, P.dW, P.s[:-1], HEAT.sigma(), path_chunk=chunk)

    np.testing.assert_allclose(Y(params, 4), Y(params, None), rtol=1e-13)

    def g(chunk):
        return jax.grad(lambda p: jnp.sum(Y(p, chunk) ** 2))(params)

    for a, b in zip(jax.tree.leaves(g(4)), jax.tree.leaves(g(None))):
        np.testing.assert_allclose(a, b, rtol=1e-12)


# --- Horvitz-Thompson unbiasedness -------------------------------------------------------------


def test_ht_unbiased_single_path():
    """Mean of HT draws of Y^theta equals the full Euler sum (one path, J = 20)."""
    P = _paths(HEAT, 1e-3, M=1, seed=1)
    full = float(model_payoff(_perturbed_u, None, X0, T0, P.X, P.dW, P.s[:-1], HEAT.sigma())[0])
    n_draws, J = 20_000, 20
    m = P.X.shape[1] // J
    offs = draw_offsets(derive(1, "ht", 0), n_draws, J, m)
    rep = lambda A: jnp.broadcast_to(A, (n_draws,) + A.shape[1:])  # noqa: E731
    nodes = ht_nodes(rep(P.X), rep(P.dW), P.s, J, offs)
    draws = model_payoff(_perturbed_u, None, X0, T0, nodes.X, nodes.dW, nodes.s, HEAT.sigma())
    mean, se = float(draws.mean()), float(draws.std() / np.sqrt(n_draws))
    assert abs(mean - full) < 3 * se, f"HT mean {mean:.6f} +- {se:.6f} vs full {full:.6f}"
    assert se > 0


def test_ustat_unbiased_over_ht_seeds():
    """Mean over HT seeds of the U-statistic equals the full-sum U-statistic on the same paths."""
    M, J, dt = 400, 10, 1e-3
    P = _paths(HEAT, dt, M=M, seed=2)
    ds = jnp.diff(P.s)
    T = word_tables(1, 2, 1)
    S = ito_signature(time_augmented_increments(jnp.swapaxes(P.dW, 0, 1), ds), T)
    Phi, norms = features(S, T, T0)
    Y = data_payoff(HEAT, P.X, P.XT, ds)
    R_full = model_payoff(_perturbed_u, None, X0, T0, P.X, P.dW, P.s[:-1], HEAT.sigma()) - Y
    target = float(rsig_ustat(R_full, Phi, norms))
    m = P.X.shape[1] // J

    @jax.jit
    def one(seed):
        nodes = ht_nodes(P.X, P.dW, P.s, J, draw_offsets(fold(derive(2, "ht"), seed), M, J, m))
        R = model_payoff(_perturbed_u, None, X0, T0, nodes.X, nodes.dW, nodes.s, HEAT.sigma()) - Y
        return rsig_ustat(R, Phi, norms)

    vals = np.array([float(one(i)) for i in range(300)])
    se = vals.std() / np.sqrt(len(vals))
    assert abs(vals.mean() - target) < 3 * se, f"{vals.mean():.4e} +- {se:.1e} vs {target:.4e}"
