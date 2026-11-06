"""Label-quality metrics and the run outputs they feed (kappa spectra, final.json labels)."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from pathenhancedfkpinn import fkdata, metrics
from pathenhancedfkpinn.problems import Heat
from pathenhancedfkpinn.signature.words import word_tables


def test_exact_beta_heat_closed_form():
    p = Heat(nu=0.05, k=1.0)
    tables = word_tables(1, 3, 1)
    x = jnp.array([[0.3], [-0.6]], jnp.float64)
    t = jnp.array([0.5, 0.9], jnp.float64)
    beta = np.asarray(metrics.exact_beta(p, x, t, tables))
    s, kp = np.sqrt(2 * p.nu), p.k * np.pi
    for j, (w, z) in enumerate(zip(tables.feats, tables.zeros), start=1):
        n = len(w)
        want = 0.0 * x[:, 0] if z else (s * kp) ** n * np.exp(-p.nu * kp**2 * t) * np.sin(kp * x[:, 0] + n * np.pi / 2)
        np.testing.assert_allclose(beta[:, j], want, atol=1e-12)
    np.testing.assert_allclose(beta[:, 0], np.exp(-p.nu * kp**2 * t) * np.sin(kp * x[:, 0]), atol=1e-12)


def test_mc_beta_matches_exact():
    """beta_hat from simulated heat paths agrees with exact_beta within 5 MC standard errors."""
    p = Heat(nu=0.05, k=1.0)
    tables = word_tables(1, 3, 1)
    x = jnp.array([[0.3], [-0.6], [0.8]], jnp.float64)
    t = jnp.array([0.5, 0.9, 0.3], jnp.float64)
    fk = fkdata.precompute(p, x, t, seed=3, M=4000, dt=1e-2, tables=tables, keep_paths=False)
    a = fk.Y[..., None] * fk.Phi / fk.norms[:, None, :]  # (P, M, W+1)
    beta_hat, se = a.mean(axis=1), a.std(axis=1, ddof=1) / np.sqrt(fk.M)
    z = np.abs(np.asarray(beta_hat - metrics.exact_beta(p, x, t, tables))) / np.asarray(se)
    assert z.max() < 5.0, z.max()


def test_label_quality_consistent():
    p = Heat(nu=0.05, k=1.0)
    tables = word_tables(1, 2, 1)
    x = jnp.array([[0.3], [-0.6]], jnp.float64)
    t = jnp.array([0.5, 0.9], jnp.float64)
    fk = fkdata.precompute(p, x, t, seed=0, M=256, dt=1e-2, tables=tables, keep_paths=False)
    q = metrics.label_quality(p, x, t, Y=fk.Y, Phi=fk.Phi, norms=fk.norms, tables=tables)
    assert q["beta_err_energy_level0"] == pytest.approx(q["ybar_rmse"] ** 2, rel=1e-10)
    assert {"beta_energy_level1", "beta_err_energy_level2", "ybar_se_rms", "ybar_mean_err"} <= set(q)


def test_word_energies_zero_residual_heat():
    """u_theta = u + e with e = 0.1 exp(-nu (3 pi)^2 t) sin(3 pi x) (also solves heat): E_w equals the
    closed form <p_w,p_w> (sigma^n d^n e)^2 for words without time letters, ~0 for time-letter words."""
    p = Heat(nu=0.05, k=1.0)
    tables = word_tables(1, 2, 1)
    x = jnp.array([[0.3], [-0.55]], jnp.float64)
    t = jnp.array([0.5, 0.8], jnp.float64)
    kk = 3 * jnp.pi

    def e(y, tau):
        return 0.1 * jnp.exp(-p.nu * kk**2 * tau) * jnp.sin(kk * y[0])

    def u(params, y, tau):
        return p.exact(y, tau) + e(y, tau)

    ev = metrics.eval_paths(p, x, t, tables, seed=1, M=20000, dt=1e-2)
    Ew = np.asarray(metrics.word_energies(u, {}, ev, p.sigma()))
    s = float(np.sqrt(2 * p.nu))
    for j, (w, z) in enumerate(zip(tables.feats, tables.zeros), start=1):
        n = len(w)
        de = (kk**n) * 0.1 * np.exp(-p.nu * kk**2 * t) * np.sin(kk * x[:, 0] + n * np.pi / 2)
        want = float(np.mean(np.asarray(ev["norms"][:, j]) * (s**n * de) ** 2))
        if z:
            assert abs(Ew[j]) < 0.05 * Ew[1:].max()
        else:
            assert Ew[j] == pytest.approx(want, rel=0.15), (w, Ew[j], want)
    assert Ew[0] == pytest.approx(float(np.mean(e(x.T, t) ** 2)), rel=0.15)


TINY = ["optim.steps=3", "data.n_coll=200", "data.M=16", "data.dt=0.05", "data.n_ic=16", "data.n_bc=16",
        "net.width=8", "net.depth=1", "eval.n_eval=16", "eval.every=100000", "log.every=100000",
        "kappa.n_c=8", "kappa.M=8", "kappa.steps=[0,final]", "arm_opts.N=2", "arm_opts.N0=1",
        "batch.B_x=2", "batch.M_prime=8", "data.path_block=16", "energies.n_points=2", "energies.M=8",
        "energies.path_chunk=null"]  # fmt: skip


@pytest.mark.parametrize("layer,arm,keys,ref", [
    ("configs/base.yaml", "B", ["ybar_rmse", "beta_err_energy_level2"], "exact_u_payoff"),
    ("configs/base.yaml", "C", ["ybar_rmse", "c_label2_rel_rmse"], "exact_u_payoff"),
    ("configs/manufactured/default.yaml", "D", [], "exact_u_payoff"),
])  # fmt: skip
def test_run_outputs(tmp_path, layer, arm, keys, ref):
    from pathenhancedfkpinn.config import resolve
    from pathenhancedfkpinn.train import train

    layers = ["configs/base.yaml"] + ([layer] if layer != "configs/base.yaml" else [])
    cfg = resolve(layers, TINY + [f"arm={arm}"])
    final = train(cfg, tmp_path / arm, smoke=True)
    assert set(keys) <= set(final["labels"])
    lines = [json.loads(ln) for ln in (tmp_path / arm / "metrics.jsonl").read_text().splitlines()]
    setup = lines[0]
    assert setup["event"] == "setup" and set(keys) <= set(setup["labels"])
    assert setup["energies"]["reference"] == ref and setup["energies"]["words"][0] == "empty"
    evals = [r["step"] for r in lines if r.get("event") == "eval"]
    energies = [r for r in lines if r.get("event") == "energies"]
    assert [r["step"] for r in energies] == evals and len(energies[0]["E_w"]) == len(setup["energies"]["words"])
    assert "E_level1" in final
    for step in (0, 3):  # kappa checkpoints: step 0 and final (= steps)
        with np.load(tmp_path / arm / "kappa" / f"step{step}.npz") as f:
            assert set(f.files) == {"t0.25", "t0.5", "t1"}
            ev = f["t1"]
            assert ev.dtype == np.float64 and np.all(np.diff(ev) <= 0)


def test_diag_time_recorded(tmp_path):
    from pathenhancedfkpinn.config import resolve
    from pathenhancedfkpinn.train import train

    final = train(resolve(["configs/base.yaml"], TINY + ["arm=Astar", "log.every=1", "eval.every=2"]), tmp_path, smoke=True)
    lines = [json.loads(ln) for ln in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    rows = [r for r in lines if "loss" in r]
    evals = [r for r in lines if r.get("event") == "eval"]
    assert all("diag_elapsed" in r for r in rows) and all("elapsed" in r for r in evals)
    assert 0 < final["diag_seconds"] <= final["wall_clock"]
    assert rows[-1]["diag_elapsed"] <= rows[-1]["elapsed"]
