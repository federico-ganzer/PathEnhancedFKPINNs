"""Snapshots: network output on a fixed grid at chosen steps. They must not change training."""

import json

import numpy as np

from pathenhancedfkpinn.config import resolve
from pathenhancedfkpinn.train import train

LAYERS = ["configs/base.yaml", "configs/manufactured/prelim.yaml"]
TINY = ["optim.steps=3", "data.n_coll=200", "data.n_ic=16", "data.n_bc=16", "net.width=8", "net.depth=1",
        "eval.n_eval=16", "eval.every=1", "log.every=1", "kappa.steps=[]", "energies.n_points=0"]  # fmt: skip
SNAP = ["snapshots.steps=[0,2,final]", "snapshots.n_x=17", "snapshots.t=[0.5,1.0]"]


def _losses(run_dir):
    rows = [json.loads(ln) for ln in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return [(r["step"], r["loss"]) for r in rows if "loss" in r]


def test_snapshots_written_and_training_unchanged(tmp_path):
    for arm in ("D", "P"):
        train(resolve(LAYERS, TINY + [f"arm={arm}"]), tmp_path / f"{arm}_off", smoke=True, resume=False)
        train(resolve(LAYERS, TINY + [f"arm={arm}"] + SNAP), tmp_path / arm, smoke=True, resume=False)
        assert _losses(tmp_path / arm) == _losses(tmp_path / f"{arm}_off")
        assert not (tmp_path / f"{arm}_off" / "snapshots").exists()

    snap = tmp_path / "D" / "snapshots"
    assert sorted(p.name for p in snap.glob("step*.npz")) == ["step0.npz", "step2.npz", "step3.npz"]  # final = 3
    grid, s0 = np.load(snap / "grid.npz"), np.load(snap / "step0.npz")
    assert sorted(s0.files) == ["d0", "d1", "d2"] and s0["d0"].shape == (2, 17)
    assert np.allclose(grid["t"], [0.5, 1.0]) and np.allclose(grid["x"][[0, -1]], [-1.0, 1.0])

    # exact derivatives of u* = sin(pi x) + eps sin(k pi x), k = 4, eps = 0.1 (static_source: no t dependence)
    x, k, eps = grid["x"], 4, 0.1
    want = [np.sin(np.pi * x) + eps * np.sin(k * np.pi * x),
            np.pi * np.cos(np.pi * x) + eps * k * np.pi * np.cos(k * np.pi * x),
            -np.pi**2 * np.sin(np.pi * x) - eps * (k * np.pi) ** 2 * np.sin(k * np.pi * x)]  # fmt: skip
    for n in range(3):
        np.testing.assert_allclose(grid[f"exact_d{n}"], np.broadcast_to(want[n], (2, 17)), atol=1e-9)

    # step 0 is the initial network, shared by the arms of a seed; later steps differ by arm
    p0 = np.load(tmp_path / "P" / "snapshots" / "step0.npz")
    np.testing.assert_array_equal(s0["d0"], p0["d0"])
    assert not np.array_equal(np.load(snap / "step3.npz")["d0"], s0["d0"])
