"""Offline FK dataset for one seed.

For every FK point (x_i, t_i): M Euler paths at uniform step dt (n_i = ceil(t_i / dt) steps,
last step partial), the data payoff Y, the orthogonal features Phi, and the point label
ybar = mean(Y). Shapes are static: every point uses n_max = ceil(T / dt) steps and steps beyond
t_i have ds = 0, so dW = 0 and they contribute nothing to Ito sums or signatures.

Model-side storage:
- mode "full": X, dW at every step (M, n_max, d) per point. d = 1 at default sizes is ~1.6 GB.
- mode "ht":   2J Horvitz-Thompson nodes per path (payoff/ht.py), blocks over each point's own
               n_i steps; block j has m_j steps and weight m_j.

Paths of point i come from keys.derive(seed, "paths", i), so all arms of a seed share them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import jax
import jax.numpy as jnp
import numpy as np

from pathenhancedfkpinn.keys import derive
from pathenhancedfkpinn.payoff.data import data_payoff
from pathenhancedfkpinn.sde.simulate import euler_scan
from pathenhancedfkpinn.signature.ito_features import features, ito_signature, time_augmented_increments
from pathenhancedfkpinn.signature.words import WordTables


@dataclass
class FKData:
    x: jax.Array  # (P, d)
    t: jax.Array  # (P,)
    Y: jax.Array  # (P, M) data payoffs
    ybar: jax.Array  # (P,) point labels (A0)
    Phi: jax.Array  # (P, M, W+1) features, column 0 = empty word
    norms: jax.Array  # (P, W+1)
    X: jax.Array  # (P, M, n, d) model-side states (full steps or HT nodes)
    dW: jax.Array  # (P, M, n, d) model-side increments
    s: jax.Array  # (P, M, n) model-side left times
    ds: jax.Array  # (P, M, n) model-side time weights
    mode: str = "full"
    meta: dict = field(default_factory=dict)

    @property
    def P(self) -> int:
        return self.x.shape[0]

    @property
    def M(self) -> int:
        return self.Y.shape[1]


def point_grid(t, dt: float, n_max: int, mesh: str = "uniform") -> tuple[jax.Array, jax.Array]:
    """(s_grid (n_max+1,), n_eff) for one point; padded with s = t beyond n_eff steps."""
    t = jnp.asarray(t, jnp.float64)
    k = jnp.arange(n_max + 1, dtype=jnp.float64)
    n_eff = jnp.maximum(jnp.ceil(t / dt - 1e-9), 1.0)
    if mesh == "uniform":
        s = jnp.minimum(k * dt, t)
    elif mesh == "graded":
        r = jnp.minimum(k / n_eff, 1.0)
        s = t * (1.0 - (1.0 - r) ** 2)
    else:
        raise ValueError(f"unknown mesh {mesh!r}")
    return s, n_eff.astype(jnp.int32)


def ht_nodes_variable(X, dW, s_grid, n_eff, J: int, key):
    """HT nodes with J blocks over the first n_eff steps (blocks of unequal size allowed).

    X, dW (M, n_max, d), s_grid (n_max+1,). Returns X, dW, s, ds node arrays (M, 2J, ...)."""
    M, n_max, d = X.shape
    j = jnp.arange(J + 1)
    b = (j * n_eff) // J  # block boundaries (J+1,)
    m = b[1:] - b[:-1]  # (J,) block sizes, may be 0 if n_eff < J
    u = jax.random.uniform(key, (M, J), jnp.float64)
    r = b[:-1][None, :] + jnp.floor(u * m[None, :]).astype(jnp.int32)  # (M, J)
    r = jnp.minimum(r, n_max - 1)
    starts = jnp.minimum(b[:-1], n_max - 1)
    Wc = jnp.concatenate([jnp.zeros((M, 1, d), dW.dtype), jnp.cumsum(dW, axis=1)], axis=1)
    coarse = Wc[:, b[1:]] - Wc[:, b[:-1]]  # (M, J, d)
    take = jax.vmap(lambda A, idx: A[idx])
    dW_r = take(dW, r)
    mf = m.astype(dW.dtype)[None, :]
    ds_all = jnp.diff(s_grid)
    ds_r = ds_all[r]
    ds_coarse = (s_grid[b[1:]] - s_grid[b[:-1]])[None, :]
    nodes_X = jnp.concatenate([jnp.broadcast_to(X[:, starts], (M, J, d)), take(X, r)], axis=1)
    nodes_dW = jnp.concatenate([coarse - mf[..., None] * dW_r, mf[..., None] * dW_r], axis=1)
    nodes_s = jnp.concatenate([jnp.broadcast_to(s_grid[starts], (M, J)), s_grid[r]], axis=1)
    nodes_ds = jnp.concatenate([ds_coarse - mf * ds_r, mf * ds_r], axis=1)
    return nodes_X, nodes_dW, nodes_s, nodes_ds


def _drift_fn(problem):
    return lambda X, s: jax.vmap(problem.b)(X)


def simulate_point(problem, x, t, key, ht_key, *, M, dt, n_max, mesh, tables, backend, J, block,
                   keep_paths: bool = True):
    """Everything offline for one FK point. Returns a dict of arrays (no leading point axis).
    key drives the paths, ht_key the HT node draws."""
    s_grid, n_eff = point_grid(t, dt, n_max, mesh)
    XT, _, (X, dW) = euler_scan(
        _drift_fn(problem), problem.sigma(), x, s_grid, key, M,
        emit=lambda X, s, ds, dW: (X, dW), block=block,
    )  # fmt: skip
    ds = jnp.diff(s_grid)
    X, dW = jnp.swapaxes(X, 0, 1), jnp.swapaxes(dW, 0, 1)  # (M, n_max, d)
    Y = data_payoff(problem, X, XT, ds)
    S = ito_signature(time_augmented_increments(jnp.swapaxes(dW, 0, 1), ds), tables, backend)
    Phi, norms = features(S, tables, t)
    out = {"Y": Y, "Phi": Phi, "norms": norms, "XT": XT}
    if not keep_paths:  # point-label arms (A0, C) need no model-side storage
        d = X.shape[-1]
        out.update(X=jnp.zeros((M, 0, d)), dW=jnp.zeros((M, 0, d)), s=jnp.zeros((M, 0)), ds=jnp.zeros((M, 0)))
    elif J is None:
        out.update(X=X, dW=dW, s=jnp.broadcast_to(s_grid[:-1], (M, n_max)), ds=jnp.broadcast_to(ds, (M, n_max)))
    else:
        nX, ndW, ns, nds = ht_nodes_variable(X, dW, s_grid, n_eff, J, ht_key)
        out.update(X=nX, dW=ndW, s=ns, ds=nds)
    return out


def precompute(problem, x_fk, t_fk, *, seed: int, M: int, dt: float, tables: WordTables,
               mesh: str = "uniform", J: int | None = None, backend: str = "auto",
               block: int | None = None, keep_paths: bool = True) -> FKData:  # fmt: skip
    """Offline FK data for all FK points (sequential over points to bound memory)."""
    n_max = int(ceil(problem.T / dt - 1e-9))
    fn = jax.jit(
        lambda x, t, i: simulate_point(
            problem, x, t, derive(seed, "paths", i), derive(seed, "ht", i), M=M, dt=dt,
            n_max=n_max, mesh=mesh, tables=tables, backend=backend, J=J, block=block,
            keep_paths=keep_paths,
        )
    )  # fmt: skip
    P = x_fk.shape[0]
    outs = [fn(x_fk[i], t_fk[i], i) for i in range(P)]
    stack = lambda k: jnp.stack([o[k] for o in outs])  # noqa: E731
    Y = stack("Y")
    return FKData(
        x=x_fk, t=t_fk, Y=Y, ybar=Y.mean(axis=1), Phi=stack("Phi"), norms=stack("norms"),
        X=stack("X"), dW=stack("dW"), s=stack("s"), ds=stack("ds"),
        mode=("full" if J is None else "ht") if keep_paths else "labels_only",
        meta=dict(M=M, dt=dt, n_max=n_max, mesh=mesh, J=J, seed=seed, N=tables.N, N0=tables.N0,
                  words=len(tables.feats), backend=backend),
    )  # fmt: skip


def pathwise_labels(problem, x_fk, t_fk, *, seed: int, M: int, dt: float, n_order: int,
                    mesh: str = "uniform", block: int | None = None) -> list[np.ndarray]:
    """Arm C labels: mean over paths of d^n/dx^n of the discretised payoff Y(x; dW), n = 1..n_order.

    Differentiates the Euler map x -> (X_k) -> Y with nested forward mode, on the same paths as
    `precompute` (same keys). This is the flow-Jacobian pathwise derivative of the discrete
    scheme; for indicator data it is 0 almost surely (intended: tests M3).
    Returns [g_1 (P, d), g_2 (P, d, d), ...]."""  # fmt: skip
    n_max = int(ceil(problem.T / dt - 1e-9))
    d = x_fk.shape[1]

    def payoff_of_x(x0, t, i):
        s_grid, _ = point_grid(t, dt, n_max, mesh)
        ds = jnp.diff(s_grid)

        def acc_f(acc, X, s, dsk, dW):
            return acc + jax.vmap(problem.f)(X) * dsk

        XT, run, _ = euler_scan(_drift_fn(problem), problem.sigma(), x0, s_grid, derive(seed, "paths", i), M,
                                accumulate=acc_f, acc0=jnp.zeros(M, jnp.float64), block=block)  # fmt: skip
        del ds
        return jnp.mean(jax.vmap(problem.u0)(XT) + run)

    labels = []
    for n in range(1, n_order + 1):
        f = payoff_of_x
        for _ in range(n):
            f = jax.jacfwd(f, argnums=0)
        g = jax.jit(f)
        labels.append(np.stack([np.asarray(g(x_fk[i], t_fk[i], i)) for i in range(x_fk.shape[0])]))
    assert all(lab.shape[1:] == (d,) * (k + 1) for k, lab in enumerate(labels))
    return labels
