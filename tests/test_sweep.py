"""Run specs: grid parsing, grid x seeds expansion, smoke subset, resume guard, smoke step cap."""

import json

from pathenhancedfkpinn import sweep
from pathenhancedfkpinn.config import config_hash


def test_parse_grid():
    g = sweep.parse_grid(["problem.k=1,4,16", "data.dt=1e-3,5e-4"])
    assert g == {"problem.k": [1, 4, 16], "data.dt": [1e-3, 5e-4]}


def test_expand_grid_and_seeds(tmp_path):
    runs = sweep.expand(["configs/base.yaml"], ["arm=D"], {"problem.k": [1.0, 2.0]}, [0, 1, 2], results_root=tmp_path)
    assert len(runs) == 6  # 2 grid points x 3 seeds, grid outer
    assert [r.seed for r in runs[:3]] == [0, 1, 2]
    assert runs[3].grid_point == {"problem.k": 2.0} and runs[3].cfg["problem"]["k"] == 2.0
    assert all(r.cfg["arm"] == "D" for r in runs)
    assert runs[0].run_dir == tmp_path / "base__D" / "k1.0" / "seed0"
    assert runs[0].cfg is not runs[1].cfg


def test_default_seed_and_name(tmp_path):
    runs = sweep.expand(["configs/base.yaml", "configs/heat1d/legacy_parity.yaml"], results_root=tmp_path)
    assert len(runs) == 1 and runs[0].seed == 0 and runs[0].name == "heat1d-legacy_parity__P"


def test_smoke_subset(tmp_path):
    runs = sweep.expand(["configs/base.yaml"], grid={"problem.k": [1.0, 2.0]}, seeds=[3, 4], smoke=True,
                        results_root=tmp_path)  # fmt: skip
    assert len(runs) == 1 and runs[0].seed == 3
    assert runs[0].run_dir == tmp_path / "smoke" / "base__P" / "k1.0" / "seed3"


def test_resume_guard(tmp_path):
    (r,) = sweep.expand(["configs/base.yaml"], results_root=tmp_path)
    assert not sweep.holds_other_config(r)  # empty dir
    r.run_dir.mkdir(parents=True)
    (r.run_dir / "env.json").write_text(json.dumps({"config_hash": config_hash(r.cfg)}))
    assert not sweep.holds_other_config(r)  # same config: safe to resume
    (r2,) = sweep.expand(["configs/base.yaml"], ["optim.lr=5e-4"], results_root=tmp_path)
    assert r2.run_dir == r.run_dir and sweep.holds_other_config(r2)


def test_smoke_step_cap(tmp_path):
    from pathenhancedfkpinn.config import resolve
    from pathenhancedfkpinn.train import SMOKE_MAX_STEPS, train

    cfg = resolve(["configs/base.yaml"], ["optim.steps=100000", "data.n_coll=200", "data.n_ic=16", "data.n_bc=16",
                                          "net.width=8", "net.depth=1", "eval.n_eval=16", "eval.every=100000",
                                          "log.every=100000"])  # fmt: skip
    cfg["arm"] = "P"
    f = train(cfg, tmp_path / "run", smoke=True)
    assert f["steps_done"] == SMOKE_MAX_STEPS == 200
