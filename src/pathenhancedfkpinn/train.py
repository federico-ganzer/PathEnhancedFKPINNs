"""Training: arm registry, offline data, jitted Adam step, evaluation, kappa, logging, resume.

    R = lam_res R_res + lam_bc R_bc + lam_data R_data        (draft eq. 1)

Arms: P, Astar, A0, B (B_0 is B with N = 0), Boff, C, D, Dprime, E.
Not implemented yet: Pplus, Phc, A0-bc, E with HT nodes, E_h (Stage 2 / rehearsal).

Run directory contents: config.yaml, metrics.jsonl, final.json, env.json, ckpt/, and kappa/
(Gauss-Newton spectra, one step<k>.npz per kappa checkpoint, one array per time slice).
Label errors against the exact solution (metrics.label_quality) go in the setup record of
metrics.jsonl and in final.json["labels"]. Time spent in eval, energies and kappa is counted in
`elapsed` and separately in `diag_elapsed` (final.json: diag_seconds), so training-only wall-clock
is elapsed - diag_elapsed. Per-word error energies (metrics.word_energies) are an
"energies" record after every "eval" record (E_w per word, E_level<n> per level; word names in the
setup record under "energies"); final.json holds the final E_level<n>.
Every random draw is keyed by (seed, purpose, index...), so a resumed run continues exactly.
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from pathenhancedfkpinn import fkdata, metrics
from pathenhancedfkpinn.config import ROOT, config_hash
from pathenhancedfkpinn.diagnostics.gn_spectrum import Blocks, condition_from_spectrum, gn_spectrum, gn_spectrum_blocks
from pathenhancedfkpinn.keys import derive
from pathenhancedfkpinn.losses import local, pinn, point
from pathenhancedfkpinn.losses.rsig import per_word_plugin, per_word_ustat
from pathenhancedfkpinn.nets import FieldNet, apply_fn, split
from pathenhancedfkpinn.payoff.model import model_payoff
from pathenhancedfkpinn.problems import make_problem
from pathenhancedfkpinn.signature.words import word_tables

SMOKE_MAX_STEPS = 200
IMPLEMENTED_ARMS = ("P", "Astar", "A0", "B", "Boff", "C", "D", "Dprime", "E")
PATH_ARMS = ("B", "Boff", "E")


# ------------------------------------------------------------------------------------------------
# setup
# ------------------------------------------------------------------------------------------------


def build_problem(cfg):
    p = dict(cfg["problem"])
    return make_problem(p.pop("name"), **p)


def build_points(problem, cfg, seed):
    """Collocation, FK subset (p_data), initial and boundary points. Fixed for the run."""
    D = cfg["data"]
    x, t = problem.sample_interior(derive(seed, "coll"), D["n_coll"])
    n_fk = int(round(D["p_data"] * D["n_coll"]))
    perm = jax.random.permutation(derive(seed, "fk_points"), D["n_coll"])
    fk, res = perm[:n_fk], perm[n_fk:]
    x_ic, _ = problem.sample_initial(derive(seed, "ic"), D["n_ic"])
    x_bc, t_bc = problem.sample_boundary(derive(seed, "bc"), D["n_bc"])
    return dict(x_res=x[res], t_res=t[res], x_fk=x[fk], t_fk=t[fk], x_ic=x_ic, x_bc=x_bc, t_bc=t_bc)


def build_eval(problem, cfg):
    """Held-out evaluation set, identical for every seed and arm (fixed key)."""
    E = cfg["eval"]
    k = derive(E.get("seed", 12345), "eval")
    k1, k2 = jax.random.split(k)
    lo, hi = problem.lo_hi()
    xT = lo + (hi - lo) * jax.random.uniform(k1, (E["n_eval"], problem.d), jnp.float64)
    tT = jnp.full((E["n_eval"],), problem.T, jnp.float64)
    xa, ta = problem.sample_interior(k2, E["n_eval"])

    def ref(x, t):
        u = jax.vmap(problem.exact)(x, t)
        g = jax.vmap(jax.grad(problem.exact))(x, t)
        return u, g

    uT, gT = ref(xT, tT)
    ua, ga = ref(xa, ta)
    return dict(xT=xT, tT=tT, uT=uT, gT=gT, xa=xa, ta=ta, ua=ua, ga=ga)


def errors(u, params, ev) -> dict:
    """Relative L2 and H1 (FK-PINN eqs. E.10-E.11) at t = T and over (0, T]."""
    out = {}
    for tag, x, t, ur, gr in [("T", ev["xT"], ev["tT"], ev["uT"], ev["gT"]),
                              ("all", ev["xa"], ev["ta"], ev["ua"], ev["ga"])]:  # fmt: skip
        up = jax.vmap(lambda xi, ti: u(params, xi, ti))(x, t)
        gp = jax.vmap(lambda xi, ti: jax.grad(lambda y: u(params, y, ti))(xi))(x, t)
        e2, g2 = (up - ur) ** 2, jnp.sum((gp - gr) ** 2, axis=1)
        out[f"rel_l2_{tag}"] = jnp.sqrt(jnp.sum(e2) / jnp.sum(ur**2))
        out[f"rel_h1_{tag}"] = jnp.sqrt(jnp.mean(e2 + g2) / jnp.mean(ur**2 + jnp.sum(gr**2, axis=1)))
    return out


def make_optimizer(cfg):
    O = cfg["optim"]
    if O.get("schedule", "exp") == "const":
        sched = O["lr"]
    elif O["schedule"] == "exp":
        sched = optax.exponential_decay(O["lr"], transition_steps=O["steps"], decay_rate=O["lr_final"] / O["lr"])
    elif O["schedule"] == "exp_staircase":
        k = O["decay_every"]
        sched = optax.exponential_decay(O["lr"], transition_steps=k,
                                        decay_rate=(O["lr_final"] / O["lr"]) ** (k / O["steps"]), staircase=True)  # fmt: skip
    else:
        raise ValueError(f"unknown schedule {O['schedule']!r}")
    return optax.adam(sched), sched


# ------------------------------------------------------------------------------------------------
# arm data (offline) and data terms (online)
# ------------------------------------------------------------------------------------------------


@dataclass
class ArmData:
    arrays: dict = field(default_factory=dict)  # jnp arrays used inside the step
    meta: dict = field(default_factory=dict)


def _tables_for(cfg):
    A = cfg["arm_opts"]
    return word_tables(cfg["problem"].get("d", 1), A.get("N", 0), A.get("N0", 2))


def build_arm_data(problem, cfg, pts, seed) -> ArmData:
    arm, A, D = cfg["arm"], cfg.get("arm_opts", {}), cfg["data"]
    x, t = pts["x_fk"], pts["t_fk"]
    ad = ArmData(meta={"n_fk": int(x.shape[0]), "labels": {}})  # labels: MC label errors (empty if exact)
    if arm == "P":
        return ad
    if arm == "Astar":
        ad.arrays["labels"] = jax.vmap(problem.exact)(x, t)
        return ad

    needs_paths = arm in ("A0", "C") + PATH_ARMS
    if needs_paths:
        tables = _tables_for(cfg) if arm in ("B", "Boff") else word_tables(problem.d, 0, 0)
        J = A.get("J") if arm in PATH_ARMS else None
        fk = fkdata.precompute(problem, x, t, seed=seed, M=D["M"], dt=D["dt"], tables=tables,
                               mesh=D.get("mesh", "uniform"), J=J, backend=A.get("sig_backend", "auto"),
                               block=D.get("path_block"), keep_paths=arm in PATH_ARMS)  # fmt: skip
        ad.meta.update(fk.meta)
        ad.arrays.update(ybar=fk.ybar)
        betas = arm in ("B", "Boff")
        ad.meta["labels"] = metrics.label_quality(problem, x, t, Y=fk.Y, Phi=fk.Phi if betas else None,
                                                  norms=fk.norms if betas else None,
                                                  tables=tables if betas else None)  # fmt: skip
        if arm in PATH_ARMS:
            ad.arrays.update(Y=fk.Y, Phi=fk.Phi, norms=fk.norms, X=fk.X, dW=fk.dW, s=fk.s)
            level = np.concatenate([[0], tables.level])
            ad.meta.update(level=level, tables=tables)
        if arm == "Boff":
            ad.arrays["beta_hat"] = jnp.mean(fk.Y[..., None] * fk.Phi, axis=1) / fk.norms
    if arm in ("C", "D", "Dprime"):
        N = A["N"]
        if arm == "C":
            g = fkdata.pathwise_labels(problem, x, t, seed=seed, M=D["M"], dt=D["dt"], n_order=N,
                                       mesh=D.get("mesh", "uniform"), block=D.get("path_block"))  # fmt: skip
            labels = [ad.arrays["ybar"]] + [jnp.asarray(gi) for gi in g]
            ad.meta["labels"].update(metrics.label_quality(problem, x, t, c_labels=g))
        else:
            labels = local.exact_derivs(problem, x, t, N)
        ad.arrays["labels"] = labels
        if arm == "Dprime":
            ad.arrays["weights"] = local.tuned_weights(t, A["lambdas"], A["family"])
        else:
            ad.arrays["weights"] = local.parseval_weights(local.noise_tau(problem, t), N)
    return ad


def _path_batch(arr, key, B_x, M_p):
    """Random B_x FK points and M_p of their M paths (without replacement)."""
    P, M = arr["Y"].shape
    k1, k2 = jax.random.split(key)
    idx = jax.random.choice(k1, P, (B_x,), replace=False)
    perm = jax.vmap(lambda k: jax.random.permutation(k, M)[:M_p])(jax.random.split(k2, B_x))

    def g(name):
        return jax.vmap(lambda i, pi: arr[name][i][pi])(idx, perm)

    return idx, {n: g(n) for n in ("Y", "Phi", "X", "dW", "s")}


def make_data_term(problem, cfg, ad: ArmData, u):
    """Returns data_term(params, key, data) -> (loss, logs dict).

    `data` = {"pts": build_points(...), "arm": ArmData.arrays}; passed as a jit argument so the
    (possibly GB-sized) path arrays are not baked into the compiled program as constants."""
    arm, A, Bt = cfg["arm"], cfg.get("arm_opts", {}), cfg.get("batch", {})
    sigma = problem.sigma()

    if arm == "P":
        return None
    if arm in ("Astar", "A0"):
        lab = "labels" if arm == "Astar" else "ybar"

        def term(params, key, data):
            x, t = data["pts"]["x_fk"], data["pts"]["t_fk"]
            return point.point_loss(u, params, x, t, data["arm"][lab]), {}

        return term
    if arm in ("C", "D", "Dprime"):

        def term(params, key, data):
            x, t, arr = data["pts"]["x_fk"], data["pts"]["t_fk"], data["arm"]
            lv = local.per_level(u, params, x, t, arr["labels"], arr["weights"])
            return jnp.sum(lv), {f"level{n}": lv[n] for n in range(lv.shape[0])}

        return term

    B_x, M_p = Bt.get("B_x", 32), Bt.get("M_prime", 64)
    chunk = Bt.get("path_chunk")

    def batch_R(params, key, data):
        x, t, arr = data["pts"]["x_fk"], data["pts"]["t_fk"], data["arm"]
        idx, b = _path_batch(arr, key, B_x, M_p)
        Yth = jax.vmap(lambda xi, ti, X, dW, s: model_payoff(u, params, xi, ti, X, dW, s, sigma, path_chunk=chunk))(
            x[idx], t[idx], b["X"], b["dW"], b["s"]
        )
        return idx, b, Yth

    if arm == "E":
        if A.get("J") is not None:
            raise NotImplementedError("arm E with HT nodes (product of two draws) is not implemented yet")

        def term(params, key, data):
            _, b, Yth = batch_R(params, key, data)
            return jnp.mean((Yth - b["Y"]) ** 2), {}

        return term

    level = jnp.asarray(ad.meta["level"])
    n_levels = int(level.max()) + 1
    estimator = per_word_ustat if A.get("estimator", "ustat") == "ustat" else per_word_plugin

    def level_logs(Ew):
        return {f"level{n}": jnp.sum(jnp.where(level == n, Ew, 0.0)) for n in range(n_levels)}

    if arm == "B":

        def term(params, key, data):
            idx, b, Yth = batch_R(params, key, data)
            R = Yth - b["Y"]
            Ew = jax.vmap(estimator)(R, b["Phi"], data["arm"]["norms"][idx]).mean(axis=0)
            return jnp.sum(Ew), level_logs(Ew)

        return term

    if arm == "Boff":
        # beta_hat from the stored data paths; beta^theta on fresh paths each step, split into two
        # independent halves so the product (b1 - beta_hat)(b2 - beta_hat) is unbiased [proposed].
        tables = ad.meta["tables"]
        D = cfg["data"]
        n_max = ad.meta["n_max"]
        if M_p % 2:
            raise ValueError("Boff needs an even M_prime")

        def fresh(params, key, i, data):
            x, t, beta_hat = data["pts"]["x_fk"], data["pts"]["t_fk"], data["arm"]["beta_hat"]
            out = fkdata.simulate_point(
                problem, x[i], t[i], jax.random.fold_in(key, 0), jax.random.fold_in(key, 1),
                M=M_p, dt=D["dt"], n_max=n_max, mesh=D.get("mesh", "uniform"), tables=tables,
                backend=A.get("sig_backend", "auto"), J=A.get("J"), block=None,
            )  # fmt: skip
            Yth = model_payoff(u, params, x[i], t[i], out["X"], out["dW"], out["s"], sigma, path_chunk=chunk)
            a = Yth[:, None] * out["Phi"] / out["norms"]
            h = M_p // 2
            d1, d2 = a[:h].mean(0) - beta_hat[i], a[h:].mean(0) - beta_hat[i]
            return out["norms"] * d1 * d2

        def term(params, key, data):
            k1, k2 = jax.random.split(key)
            idx = jax.random.choice(k1, data["pts"]["x_fk"].shape[0], (B_x,), replace=False)
            Ew = jax.vmap(lambda i, k: fresh(params, k, i, data))(idx, jax.random.split(k2, B_x)).mean(axis=0)
            return jnp.sum(Ew), level_logs(Ew)

        return term
    raise ValueError(f"unknown arm {arm!r}")


# ------------------------------------------------------------------------------------------------
# kappa (Gauss-Newton) at checkpoints, per time slice
# ------------------------------------------------------------------------------------------------


def build_kappa(problem, cfg, ad: ArmData, u, seed):
    """Residual-vector builders for the arm's data term at n_c points per slice t in {T/4, T/2, T}.

    Point and derivative arms: exact cheap Jacobians (labels do not enter J J^T).
    Path arms: a reduced path set (n_c points, M_k paths) simulated once, labelled 'estimate'."""
    K = cfg.get("kappa", {})
    if not K.get("steps"):
        return None
    arm = cfg["arm"]
    if arm == "P":
        return None
    n_c = K.get("n_c") or (512 if arm not in PATH_ARMS else 128)  # null in base.yaml = arm default
    slices = [problem.T * f for f in K.get("slices", [0.25, 0.5, 1.0])]
    lo, hi = problem.lo_hi()
    out = {}
    for j, ts in enumerate(slices):
        xk = lo + (hi - lo) * jax.random.uniform(derive(seed, "eval", 7, j), (n_c, problem.d), jnp.float64)
        tk = jnp.full((n_c,), ts, jnp.float64)
        if arm in ("Astar", "A0"):
            out[ts] = (lambda xk, tk: lambda p: point.point_residual_vector(u, p, xk, tk, 0.0))(xk, tk)
        elif arm in ("C", "D", "Dprime"):
            N = cfg["arm_opts"]["N"]
            if arm == "Dprime":
                w = local.tuned_weights(tk, cfg["arm_opts"]["lambdas"], cfg["arm_opts"]["family"])
            else:
                w = local.parseval_weights(local.noise_tau(problem, tk), N)
            zeros = [jnp.zeros((n_c,) + (problem.d,) * n) for n in range(N + 1)]
            out[ts] = (lambda xk, tk, w, z: lambda p: local.residual_vector(u, p, xk, tk, z, w))(xk, tk, w, zeros)
        else:
            D, A = cfg["data"], cfg.get("arm_opts", {})
            M_k = K.get("M", 64)
            tables = ad.meta["tables"] if arm in ("B", "Boff") else word_tables(problem.d, 0, 0)
            fk = fkdata.precompute(problem, xk, tk, seed=seed + 10_000 * (j + 1), M=M_k, dt=D["dt"],
                                   tables=tables, mesh=D.get("mesh", "uniform"), J=A.get("J"),
                                   backend=A.get("sig_backend", "auto"))  # fmt: skip
            sigma = problem.sigma()

            def block(p, xi, ti, X, dW, s, Phi, norms):  # one point's rows; J built point by point
                Yth = model_payoff(u, p, xi, ti, X, dW, s, sigma)  # (M_k,)
                if arm == "E":
                    return Yth / jnp.sqrt(n_c * Yth.shape[0])
                beta = jnp.mean(Yth[:, None] * Phi, axis=0) / norms  # plug-in
                return jnp.sqrt(norms / n_c) * beta

            out[ts] = Blocks(block, (xk, tk, fk.X, fk.dW, fk.s, fk.Phi, fk.norms))
    return out


# ------------------------------------------------------------------------------------------------
# per-word error energies at eval checkpoints
# ------------------------------------------------------------------------------------------------


def build_energies(problem, cfg, pts, u, seed):
    """(energy_fn(params) -> record dict, meta dict). energy_fn is None if energies.n_points is 0
    or the problem has no exact solution.

    Evaluation points: the first n_points FK points. Word set: arm_opts N, N0 (as arm B would use)."""
    E = cfg.get("energies") or {}
    n_e = min(int(E.get("n_points") or 0), int(pts["x_fk"].shape[0]))
    if n_e == 0:
        return None, {}
    if not problem.has_exact():
        return None, {"reference": "none (no exact solution)"}
    A, D = cfg.get("arm_opts", {}), cfg["data"]
    tables = word_tables(problem.d, A.get("N", 0), A.get("N0", 2))
    M_e, chunk = int(E.get("M", 256)), E.get("path_chunk")
    ev = metrics.eval_paths(problem, pts["x_fk"][:n_e], pts["t_fk"][:n_e], tables, seed=seed, M=M_e, dt=D["dt"],
                            mesh=D.get("mesh", "uniform"), backend=A.get("sig_backend", "auto"), path_chunk=chunk)  # fmt: skip
    level = np.concatenate([[0], tables.level])
    meta = {"words": metrics.word_names(tables), "n_points": n_e, "M": M_e, "reference": "exact_u_payoff"}
    fn = eqx.filter_jit(lambda p, ev: metrics.word_energies(u, p, ev, problem.sigma(), path_chunk=chunk))

    def energy_fn(params):
        Ew = np.asarray(fn(params, ev), dtype=np.float64)
        rec = {f"E_level{n}": float(Ew[level == n].sum()) for n in range(int(level.max()) + 1)}
        rec["E_w"] = [float(v) for v in Ew]
        return rec

    return energy_fn, meta


# ------------------------------------------------------------------------------------------------
# run bookkeeping
# ------------------------------------------------------------------------------------------------


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "branch": run("branch", "--show-current")}


def env_info() -> dict:
    import equinox
    import signax

    return {
        "git": git_info(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "equinox": equinox.__version__,
        "optax": optax.__version__,
        "signax": signax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _floatify(d: dict) -> dict:
    return {k: (float(v) if hasattr(v, "dtype") or isinstance(v, (int, float)) else v) for k, v in d.items()}


class JsonlLog:
    def __init__(self, path: Path, resume_step: int | None):
        """resume_step None: fresh start, any earlier log is discarded. Else keep lines up to resume_step."""
        self.path = path
        if resume_step is None:
            path.write_text("", encoding="utf-8")
        elif path.exists():  # drop lines logged after the checkpoint
            keep = [ln for ln in path.read_text().splitlines() if ln and json.loads(ln).get("step", -1) <= resume_step]
            path.write_text("".join(ln + "\n" for ln in keep))
        self.fh = open(path, "a", encoding="utf-8")

    def write(self, rec: dict):
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# ------------------------------------------------------------------------------------------------
# main entry
# ------------------------------------------------------------------------------------------------


def train(cfg: dict, run_dir: str | Path, smoke: bool = False, resume: bool = True) -> dict:
    """Run one configuration with one seed. Returns the final record (also in final.json)."""
    if cfg["arm"] not in IMPLEMENTED_ARMS:
        raise NotImplementedError(f"arm {cfg['arm']!r} not implemented; have {IMPLEMENTED_ARMS}")
    run_dir = Path(run_dir)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    steps = int(cfg["optim"]["steps"])
    if smoke:
        steps = min(steps, SMOKE_MAX_STEPS)
    O, L = cfg["optim"], cfg.get("log", {})
    lam = cfg.get("weights", {})
    l_res, l_bc, l_data = lam.get("res", 1.0), lam.get("bc", 1.0), lam.get("data", 1.0)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    env = env_info()
    env["config_hash"] = config_hash(cfg)
    env["smoke"] = smoke
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))

    t_setup = time.time()
    problem = build_problem(cfg)
    pts = build_points(problem, cfg, seed)
    ev = build_eval(problem, cfg)
    lo, hi = problem.lohi_with_time()
    N_ = cfg["net"]
    if N_.get("scale_inputs", True) is False:
        lo, hi = [-1.0] * len(lo), [1.0] * len(hi)  # identity map: legacy parity
    net = FieldNet.create(problem.d, True, lo, hi, width=N_["width"], depth=N_["depth"],
                          activation=N_.get("activation", "tanh"), init=N_.get("init", "xavier_normal"),
                          key=derive(seed, "init"))  # fmt: skip
    params, static = split(net)
    u = apply_fn(static)
    ad = build_arm_data(problem, cfg, pts, seed)
    data_term = make_data_term(problem, cfg, ad, u)
    data = {"pts": pts, "arm": ad.arrays}
    kappa_fns = build_kappa(problem, cfg, ad, u, seed)
    energy_fn, energy_meta = build_energies(problem, cfg, pts, u, seed)
    setup_seconds = time.time() - t_setup

    Dc = cfg["data"]
    resample = Dc.get("resample", False)
    Bt = cfg.get("batch", {})

    def loss_fn(p, step, data):
        pts = data["pts"]
        key = derive(seed, "train", step)
        k_res, k_ic, k_bc, k_data = jax.random.split(key, 4)
        if resample:  # fresh collocation each step (legacy notebook protocol)
            xr, tr = problem.sample_interior(k_res, Bt["n_res"])
            xi, _ = problem.sample_initial(k_ic, Bt["n_ic"])
            xb, tb = problem.sample_boundary(k_bc, Bt["n_bc"])
        else:
            xr, tr, xi, xb, tb = pts["x_res"], pts["t_res"], pts["x_ic"], pts["x_bc"], pts["t_bc"]
            if Bt.get("n_res"):
                j = jax.random.choice(k_res, xr.shape[0], (Bt["n_res"],), replace=False)
                xr, tr = xr[j], tr[j]
        r_res = pinn.residual_loss(problem, u, p, xr, tr)
        r_bc, bc_logs = pinn.bc_loss(problem, u, p, xb, tb, xi)
        total = l_res * r_res + l_bc * r_bc
        logs = {"res": r_res, "bc": r_bc, **bc_logs}
        if data_term is not None:
            r_data, dlogs = data_term(p, k_data, data)
            total = total + l_data * r_data
            logs.update(data=r_data, **{f"data_{k}": v for k, v in dlogs.items()})
        return total, logs

    opt, sched = make_optimizer(cfg)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step_fn(p, s, step, data):
        (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, step, data)
        updates, s = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss, logs

    _eval = eqx.filter_jit(lambda p, ev: errors(u, p, ev))

    def eval_fn(p):
        return _eval(p, ev)

    # --- resume ---
    start, elapsed0, diag0 = 0, 0.0, 0.0
    ck = run_dir / "ckpt" / "state.eqx"
    meta_path = run_dir / "ckpt" / "meta.json"
    if resume and ck.exists() and meta_path.exists():
        params, opt_state = eqx.tree_deserialise_leaves(ck, (params, opt_state))
        meta = json.loads(meta_path.read_text())
        start, elapsed0, diag0 = meta["step"] + 1, meta["elapsed"], meta.get("diag_elapsed", 0.0)
    log = JsonlLog(run_dir / "metrics.jsonl", start - 1 if start else None)
    if start == 0:
        log.write({"step": -1, "event": "setup", "setup_seconds": setup_seconds, "energies": energy_meta,
                   **{k: v for k, v in ad.meta.items() if k not in ("tables", "level")}})  # fmt: skip

    log_every, eval_every, ckpt_every = L.get("every", 100), cfg["eval"].get("every", 1000), L.get("ckpt_every", 2000)
    kappa_steps = set(cfg.get("kappa", {}).get("steps", []) or [])
    kappa_final = "final" in kappa_steps
    kappa_steps.discard("final")
    est = cfg.get("budget", {}).get("est_seconds")
    status, reason, loss0 = "done", None, None
    t0 = time.time() - elapsed0
    # Seconds spent in eval, energies and kappa (inside `elapsed`); training time = elapsed - diag_elapsed.
    diag = {"s": diag0}

    def timed(fn, *a):
        t = time.time()
        out = fn(*a)
        diag["s"] += time.time() - t
        return out

    def do_kappa(step):
        rec, spectra = {"step": step, "event": "kappa"}, {}
        for ts, fn in (kappa_fns or {}).items():
            ev = gn_spectrum_blocks(fn, params) if isinstance(fn, Blocks) else gn_spectrum(fn, params)
            k, n = condition_from_spectrum(ev, cfg["kappa"].get("cutoff", 1e-8))
            rec[f"kappa_t{ts:g}"] = k
            rec[f"rank_t{ts:g}"] = n
            spectra[f"t{ts:g}"] = ev
        (run_dir / "kappa").mkdir(exist_ok=True)
        np.savez(run_dir / "kappa" / f"step{step}.npz", **spectra)  # eigenvalues of J J^T, descending
        log.write(rec)

    step, last_energies = start - 1, (None, None)
    for step in range(start, steps):
        if kappa_fns and step in kappa_steps:
            timed(do_kappa, step)
        params, opt_state, loss, logs = step_fn(params, opt_state, jnp.asarray(step, jnp.int32), data)
        if step % log_every == 0 or step == steps - 1:
            lv = float(loss)
            rec = {"step": step, "loss": lv, "lr": float(sched(step)) if callable(sched) else sched,
                   "elapsed": time.time() - t0, "diag_elapsed": diag["s"], **_floatify(logs)}  # fmt: skip
            log.write(rec)
            if not math.isfinite(lv):
                status, reason = "halted", "non-finite loss"
            loss0 = lv if loss0 is None else loss0
            if loss0 and lv > 1e3 * loss0:
                status, reason = "halted", f"loss {lv:.3e} > 1e3 x initial {loss0:.3e}"
            if est and time.time() - t0 > 2 * est:
                status, reason = "halted", "wall-clock > 2x estimate"
        if step % eval_every == 0 or step == steps - 1:
            t_ev = time.time() - t0
            log.write({"step": step, "event": "eval", "elapsed": t_ev, "diag_elapsed": diag["s"],
                       **_floatify(timed(eval_fn, params))})  # fmt: skip
            if energy_fn is not None:
                last_energies = (step, timed(energy_fn, params))
                log.write({"step": step, "event": "energies", **last_energies[1]})
        if (step % ckpt_every == 0 and step > start) or step == steps - 1 or status == "halted":
            eqx.tree_serialise_leaves(ck, (params, opt_state))
            meta_path.write_text(json.dumps({"step": step, "elapsed": time.time() - t0, "diag_elapsed": diag["s"]}))
        if status == "halted":
            break
    final = {"status": status, "reason": reason, "steps_done": step + 1, "steps_total": steps,
             "wall_clock": time.time() - t0, "diag_seconds": diag["s"], "setup_seconds": setup_seconds, "smoke": smoke,
             "config_hash": config_hash(cfg), **_floatify(eval_fn(params)),
             "labels": ad.meta.get("labels", {})}  # fmt: skip
    if energy_fn is not None:
        rec = last_energies[1] if last_energies[0] == step else energy_fn(params)
        final.update({k: v for k, v in rec.items() if k != "E_w"})
    if kappa_fns and kappa_final and status == "done":
        do_kappa(steps)
    (run_dir / "final.json").write_text(json.dumps(final, indent=2))
    log.close()
    return final
