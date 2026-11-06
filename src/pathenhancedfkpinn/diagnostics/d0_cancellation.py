"""D0: cancellation map. Mandatory before training any MC arm on a benchmark.

At FK points, use the reference solution as a stand-in network and compute the per-path std of
the data payoff Y and of the residual R = Y^theta - Y, for a simulator setting (dt, mesh, HT J).

Pass: median std(R)/std(Y) <= 0.3 and 90th percentile <= 0.5. If it fails, B cannot beat A0 on
noise; fix the simulator first. Never use HT on a configuration that fails D0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pathenhancedfkpinn import fkdata
from pathenhancedfkpinn.payoff.model import model_payoff
from pathenhancedfkpinn.signature.words import word_tables


def d0_map(problem, x, t, *, seed: int, M: int, dt: float, mesh: str = "uniform", J: int | None = None,
           block: int | None = None) -> dict:  # fmt: skip
    """Per-point std(Y), std(R) and their ratio at the exact solution."""
    tables = word_tables(problem.d, 0, 0)
    fk = fkdata.precompute(problem, x, t, seed=seed, M=M, dt=dt, tables=tables, mesh=mesh, J=J, block=block)

    def u(params, y, tau):
        return problem.exact(y, tau)

    sigma = problem.sigma()
    Yth = jax.vmap(lambda xi, ti, X, dW, s: model_payoff(u, None, xi, ti, X, dW, s, sigma))(x, t, fk.X, fk.dW, fk.s)
    R = Yth - fk.Y
    sY, sR = np.asarray(jnp.std(fk.Y, axis=1)), np.asarray(jnp.std(R, axis=1))
    ratio = sR / np.maximum(sY, 1e-300)
    med, p90 = float(np.median(ratio)), float(np.quantile(ratio, 0.9))
    return {
        "setting": {"M": M, "dt": dt, "mesh": mesh, "J": J},
        "n_points": int(x.shape[0]),
        "std_Y_median": float(np.median(sY)),
        "std_R_median": float(np.median(sR)),
        "ratio_median": med,
        "ratio_p90": p90,
        "pass": bool(med <= 0.3 and p90 <= 0.5),
        "per_point": {"t": np.asarray(t).tolist(), "std_Y": sY.tolist(), "std_R": sR.tolist()},
    }
