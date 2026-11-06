"""Train the runs of one run spec: config layers + overrides + optional grid + seeds.

    uv run python scripts/run.py configs/base.yaml configs/heat1d/legacy_parity.yaml --seeds 0 1 2 3 4
    uv run python scripts/run.py configs/base.yaml configs/manufactured/default.yaml \
        --set arm=D --grid problem.k=1,4,16 problem.T=0.25,1 --seeds 0 1 2 3 4
    uv run python scripts/run.py LAYERS... --smoke     # <= 200 steps, first seed and grid point
    uv run python scripts/run.py LAYERS... --list      # print the runs and exit

Config layers come first. Without --task-index every run executes sequentially (laptop). With it,
only that run (Slurm job arrays, see scripts/submit.py). A run directory is resumed only if it
holds the same resolved config; otherwise pass a different --name, or --no-resume to overwrite.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from pathenhancedfkpinn import sweep
from pathenhancedfkpinn.train import train


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(parents=[sweep.spec_parser()])
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="print the runs and exit")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args(argv)

    runs = sweep.from_args(a)
    if a.list:
        for i, r in enumerate(runs):
            print(i, r.seed, r.grid_point, r.run_dir)
        return 0
    if a.task_index is not None:
        runs = [runs[a.task_index]]
    rc = 0
    for r in runs:
        if not a.no_resume and sweep.holds_other_config(r):
            print(f"[refused] {r.run_dir} holds a different config: use another --name, or --no-resume", flush=True)
            rc = 1
            continue
        print(f"[run] {r.name} seed={r.seed} grid={r.grid_point} -> {r.run_dir}", flush=True)
        try:
            final = train(r.cfg, r.run_dir, smoke=a.smoke, resume=not a.no_resume)
            print(f"[done] status={final['status']} rel_l2_T={final.get('rel_l2_T')}", flush=True)
            if final["status"] != "done":
                rc = 2
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            r.run_dir.mkdir(parents=True, exist_ok=True)
            (r.run_dir / "error.txt").write_text(traceback.format_exc())
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
