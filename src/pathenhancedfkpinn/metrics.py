"""Metrics beyond the held-out errors.

Label quality (once, at setup): Monte Carlo labels against exact values.
    ybar      point labels vs u(x, t): rmse, relative rmse, mean error (bias), rms MC standard error
    C labels  pathwise derivatives vs exact grad^n u: relative rmse per order n
    beta_hat  signature coefficients vs exact beta_w: error energy sum_w <p_w,p_w> (beta_hat - beta)^2
              per level, next to the exact energy sum_w <p_w,p_w> beta_w^2 (problems with exact beta)

Per-word error energies (at eval checkpoints):
    E_w = <p_w,p_w> delta_beta_w^2,   delta_beta_w = E[(Y^theta - Y^u) Phi_w] / <p_w,p_w>
averaged over evaluation points. Y^u is the model payoff of the exact solution on the same paths,
u(x, t) + sum_k grad_sigma u(X_k, t - s_k) . dW_k (the discrete martingale representation of the data
payoff), so E[Y^u Phi_w] / <p_w,p_w> = beta_w up to O(dt) and E_w is the error against the exact
coefficients. Differencing on the same paths makes the estimator's noise shrink with the error;
comparing mean(Y^theta Phi_w) against closed-form beta_w instead leaves noise of order
Var(Y^theta Phi_w) / M, which swamps late-training energies. U-statistic over paths: unbiased, can
be slightly negative; never clamp. Needs problem.exact (heat, OU, manufactured all have it).
"""

from __future__ import annotations

from functools import partial
from math import ceil

import jax
import jax.numpy as jnp
import numpy as np

from pathenhancedfkpinn.keys import derive
from pathenhancedfkpinn.losses.local import exact_derivs
from pathenhancedfkpinn.payoff.model import model_payoff
from pathenhancedfkpinn.problems.heat import Heat
from pathenhancedfkpinn.signature.words import WordTables


def exact_beta(problem, x: jax.Array, t: jax.Array, tables: WordTables) -> jax.Array | None:
    """Exact coefficients (P, W+1) at points x (P, d), t (P,); column 0 = empty word. None if unknown.

    Heat: beta_empty = u; beta_w = sigma^n d_{i1}...d_{in} u(x, t) for a word w = i1...in without
    time letters; 0 for words with a time letter (Y depends on W_t only)."""
    if not isinstance(problem, Heat):
        return None
    N = int(max(tables.level, default=0))
    D = exact_derivs(problem, x, t, N)
    sigma = jnp.asarray(problem.sigma(), jnp.float64)
    P = x.shape[0]
    cols = [D[0]]
    for w, n, z in zip(tables.feats, tables.level, tables.zeros):
        if z > 0:
            cols.append(jnp.zeros((P,), jnp.float64))
        else:
            idx = (slice(None),) + tuple(int(c) - 1 for c in w)
            cols.append(sigma**int(n) * D[int(n)][idx])
    return jnp.stack(cols, axis=1)


def label_quality(problem, x, t, *, Y=None, Phi=None, norms=None, tables: WordTables | None = None,
                  c_labels: list | None = None) -> dict:  # fmt: skip
    """Label errors against the exact solution. Y (P, M) payoffs; Phi (P, M, W+1), norms (P, W+1)
    features of the same paths; c_labels [g_1 (P, d), g_2 (P, d, d), ...] arm-C labels."""
    if not problem.has_exact():
        return {}
    out: dict[str, float] = {}
    u = jax.vmap(problem.exact)(x, t)
    if Y is not None:
        M = Y.shape[1]
        err = Y.mean(axis=1) - u
        se = Y.std(axis=1, ddof=1) / np.sqrt(M)
        out.update(
            ybar_rmse=float(jnp.sqrt(jnp.mean(err**2))),
            ybar_rel_rmse=float(jnp.sqrt(jnp.sum(err**2) / jnp.sum(u**2))),
            ybar_mean_err=float(jnp.mean(err)),
            ybar_se_rms=float(jnp.sqrt(jnp.mean(se**2))),
        )
    if c_labels:
        D = exact_derivs(problem, x, t, len(c_labels))
        P = x.shape[0]
        for n, g in enumerate(c_labels, start=1):
            diff, ref = (jnp.asarray(g) - D[n]).reshape(P, -1), D[n].reshape(P, -1)
            out[f"c_label{n}_rel_rmse"] = float(jnp.sqrt(jnp.sum(diff**2) / jnp.sum(ref**2)))
    if Phi is not None and tables is not None:
        beta = exact_beta(problem, x, t, tables)
        if beta is not None:
            beta_hat = jnp.mean(Y[..., None] * Phi, axis=1) / norms
            e_err, e_ref = norms * (beta_hat - beta) ** 2, norms * beta**2  # (P, W+1)
            level = np.concatenate([[0], tables.level])
            for n in range(int(level.max()) + 1):
                m = jnp.asarray(level == n)
                out[f"beta_err_energy_level{n}"] = float(jnp.mean(jnp.sum(jnp.where(m, e_err, 0.0), axis=1)))
                out[f"beta_energy_level{n}"] = float(jnp.mean(jnp.sum(jnp.where(m, e_ref, 0.0), axis=1)))
    return out


# ------------------------------------------------------------------------------------------------
# per-word error energies
# ------------------------------------------------------------------------------------------------


def word_names(tables: WordTables) -> list[str]:
    """'empty', then each feature word as its letters ('1', '01', '11', ...); 0 = time letter."""
    return ["empty"] + ["".join(str(c) for c in w) for w in tables.feats]


def _simulate(problem, x, t, key, *, M, dt, n_max, mesh, tables, backend, keep_paths):
    from pathenhancedfkpinn.fkdata import simulate_point

    return simulate_point(problem, x, t, key, key, M=M, dt=dt, n_max=n_max, mesh=mesh, tables=tables,
                          backend=backend, J=None, block=None, keep_paths=keep_paths)  # fmt: skip


def eval_paths(problem, x, t, tables: WordTables, *, seed: int, M: int, dt: float, mesh: str = "uniform",
               backend: str = "auto", path_chunk: int | None = None) -> dict:  # fmt: skip
    """Fixed evaluation paths at points x (P, d), t (P,), keyed by (seed, 'energy_paths', i), with
    features and Yref = Y^u, the exact solution's model payoff on each path (P, M)."""
    n_max = int(ceil(problem.T / dt - 1e-9))
    fn = jax.jit(partial(_simulate, problem, M=M, dt=dt, n_max=n_max, mesh=mesh, tables=tables,
                         backend=backend, keep_paths=True))  # fmt: skip
    outs = [fn(x[i], t[i], derive(seed, "energy_paths", i)) for i in range(x.shape[0])]
    stack = lambda k: jnp.stack([o[k] for o in outs])  # noqa: E731
    ev = dict(x=x, t=t, X=stack("X"), dW=stack("dW"), s=stack("s"), Phi=stack("Phi"), norms=stack("norms"))
    u_exact = lambda params, y, tau: problem.exact(y, tau)  # noqa: E731
    yref = jax.jit(lambda a: model_payoff(u_exact, None, *a, problem.sigma(), path_chunk=path_chunk))
    ev["Yref"] = jnp.stack([yref((x[i], t[i], ev["X"][i], ev["dW"][i], ev["s"][i])) for i in range(x.shape[0])])
    return ev


def word_energies(u, params, ev: dict, sigma, path_chunk: int | None = None) -> jax.Array:
    """(W+1,) per-word error energies E_w, mean over evaluation points (U-statistic over paths)."""

    def one(args):
        x, t, X, dW, s, Phi, norms, Yref = args
        R = model_payoff(u, params, x, t, X, dW, s, sigma, path_chunk=path_chunk) - Yref  # (M,)
        a = R[:, None] * Phi / norms
        M = a.shape[0]
        return norms * (a.sum(axis=0) ** 2 - (a**2).sum(axis=0)) / (M * (M - 1))

    E = jax.lax.map(one, (ev["x"], ev["t"], ev["X"], ev["dW"], ev["s"], ev["Phi"], ev["norms"], ev["Yref"]))
    return E.mean(axis=0)
