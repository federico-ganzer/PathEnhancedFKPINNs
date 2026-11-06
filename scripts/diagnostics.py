"""Pre-training diagnostics (no arm is trained to completion).

    uv run python scripts/diagnostics.py d0 --layers configs/base.yaml configs/heat1d/default.yaml \
        --settings "dt=1e-3" "dt=1e-3,J=16" "dt=1e-3,mesh=graded" [--n-points 50] [--M 500]
    uv run python scripts/diagnostics.py d4 --layers configs/base.yaml configs/heat1d/legacy_parity.yaml \
        [--set arm=D] [--steps 50]    # est. hours incl. kappa checkpoints and evals (smoke)

Outputs JSON under results/diagnostics/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pathenhancedfkpinn.config import ROOT, parse_value, resolve


def _setting(s: str) -> dict:
    out = {}
    for kv in s.split(","):
        k, _, v = kv.partition("=")
        out[k.strip()] = parse_value(v)
    return out


def cmd_d0(a) -> int:
    from pathenhancedfkpinn.diagnostics.d0_cancellation import d0_map
    from pathenhancedfkpinn.train import build_points, build_problem

    cfg = resolve([ROOT / p for p in a.layers], a.set)
    problem = build_problem(cfg)
    pts = build_points(problem, cfg, int(cfg["seed"]))
    x, t = pts["x_fk"][: a.n_points], pts["t_fk"][: a.n_points]
    results = []
    for s in a.settings:
        st = _setting(s)
        t0 = time.time()
        r = d0_map(problem, x, t, seed=int(cfg["seed"]), M=st.get("M", a.M or cfg["data"]["M"]),
                   dt=st.get("dt", cfg["data"]["dt"]), mesh=st.get("mesh", "uniform"), J=st.get("J"))  # fmt: skip
        r["seconds"] = time.time() - t0
        results.append(r)
        print(f"{s:30s} ratio median {r['ratio_median']:.3f}  p90 {r['ratio_p90']:.3f}  "
              f"std(Y) {r['std_Y_median']:.3g}  std(R) {r['std_R_median']:.3g}  {'PASS' if r['pass'] else 'FAIL'}")  # fmt: skip
    out = ROOT / "results" / "diagnostics" / f"d0_{cfg['problem']['name']}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg, "results": results}, indent=1))
    print(f"wrote {out}")
    return 0


def cmd_d4(a) -> int:
    """Time a short smoke of one config and estimate the full run's wall-clock.

    Measures setup, training ms/step (diagnostics excluded), one kappa checkpoint (compile
    included, as at every checkpoint) and one eval with per-word energies (after compile), then
    scales each to the config's own steps, kappa.steps and eval.every."""
    import jax

    from pathenhancedfkpinn.sweep import default_name
    from pathenhancedfkpinn.train import train

    cfg = resolve([ROOT / p for p in a.layers], a.set)
    name = default_name(a.layers, cfg)
    full_steps, eval_every = int(cfg["optim"]["steps"]), int(cfg["eval"].get("every", 1000))
    kappa_steps = list(cfg.get("kappa", {}).get("steps") or [])
    cfg["optim"]["steps"] = a.steps
    cfg["log"]["every"] = 1
    cfg["eval"]["every"] = 10**9  # eval (+ energies) at step 0 and at the last step
    cfg["kappa"]["steps"] = [0] if kappa_steps else []
    out_dir = ROOT / "results" / "smoke" / "d4" / name
    final = train(cfg, out_dir, smoke=True, resume=False)
    recs = [json.loads(ln) for ln in (out_dir / "metrics.jsonl").read_text().splitlines() if '"loss"' in ln]
    el = [r["elapsed"] - r["diag_elapsed"] for r in recs]  # training-only clock
    per_step = (el[-1] - el[1]) / max(len(el) - 2, 1) if len(el) > 2 else float("nan")
    kappa_s = recs[0]["diag_elapsed"] if kappa_steps else 0.0  # kappa at step 0 runs before the first row
    eval_s = final["diag_seconds"] - recs[-1]["diag_elapsed"]  # the last eval: no compile
    n_evals = (full_steps - 1) // eval_every + 1 + (1 if (full_steps - 1) % eval_every else 0)
    mem = None
    try:
        stats = jax.devices()[0].memory_stats()
        mem = stats.get("peak_bytes_in_use") if stats else None
    except Exception:  # noqa: BLE001
        pass
    est = final["setup_seconds"] + full_steps * per_step + len(kappa_steps) * kappa_s + n_evals * eval_s
    res = {"name": name, "device": str(jax.devices()[0]), "setup_seconds": final["setup_seconds"],
           "ms_per_step": 1e3 * per_step, "kappa_seconds_each": kappa_s, "eval_seconds_each": eval_s,
           "train_steps": full_steps, "n_kappa": len(kappa_steps), "n_evals": n_evals,
           "est_hours": est / 3600, "peak_bytes": mem, "steps_timed": len(el)}  # fmt: skip
    print(json.dumps(res, indent=1))
    out = ROOT / "results" / "diagnostics" / f"d4_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("d0")
    s.add_argument("--layers", nargs="+", default=["configs/base.yaml"])
    s.add_argument("--set", nargs="*", default=[])
    s.add_argument("--settings", nargs="+", default=["dt=1e-3"])
    s.add_argument("--n-points", type=int, default=50)
    s.add_argument("--M", type=int, default=None)
    s = sub.add_parser("d4")
    s.add_argument("--layers", nargs="+", default=["configs/base.yaml"])
    s.add_argument("--set", nargs="*", default=[])
    s.add_argument("--steps", type=int, default=50)
    a = ap.parse_args(argv)
    return {"d0": cmd_d0, "d4": cmd_d4}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
