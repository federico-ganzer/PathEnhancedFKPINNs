"""Submit a run spec to Slurm as a job array (one task per grid point x seed).

    uv run python scripts/submit.py configs/base.yaml configs/heat1d/legacy_parity.yaml --seeds 0 1 2 3 4 \
        [--walltime 02:00:00] [--dry-run]

Takes the run-spec arguments of scripts/run.py, plus --walltime, --cluster and --dry-run.
Renders slurm/array.sbatch.tmpl with slurm/cluster.yaml, writes slurm/jobs/<name>.sbatch and
calls sbatch (unless --dry-run). Walltime: --walltime, else cluster default_walltime.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from pathenhancedfkpinn import sweep
from pathenhancedfkpinn.config import ROOT


def render(name: str, n_runs: int, run_args: list[str], walltime: str, cluster: dict) -> str:
    tmpl = (ROOT / "slurm" / "array.sbatch.tmpl").read_text()
    return tmpl.format(
        job_name=name[:60], account=cluster["account"], partition=cluster["partition"],
        gpus=cluster.get("gpus", 1), walltime=walltime, last_index=n_runs - 1,
        log_dir=cluster.get("log_dir", "slurm/logs"), extra_sbatch="\n".join(cluster.get("extra_sbatch") or []),
        repo=cluster["repo"], module_lines="\n".join(f"module load {m}" for m in cluster.get("modules") or []),
        cuda_extra=cluster.get("cuda_extra", "cuda13"), run_args=shlex.join(run_args),
    )  # fmt: skip


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(usage="submit.py LAYERS... [run.py options] [--walltime HH:MM:SS] [--dry-run]")
    ap.add_argument("--cluster", default=str(ROOT / "slurm" / "cluster.yaml"))
    ap.add_argument("--walltime", default=None, help="HH:MM:SS; default: cluster default_walltime")
    ap.add_argument("--dry-run", action="store_true")
    a, run_args = ap.parse_known_args(argv)
    spec = argparse.ArgumentParser(prog="submit.py (run spec)", parents=[sweep.spec_parser()]).parse_args(run_args)
    runs = sweep.from_args(spec)
    name = runs[0].name + ("__smoke" if spec.smoke else "")

    cluster = yaml.safe_load(Path(a.cluster).read_text())
    script = render(name, len(runs), run_args, a.walltime or cluster.get("default_walltime", "02:00:00"), cluster)
    out = ROOT / "slurm" / "jobs" / f"{name}.sbatch"
    out.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / cluster.get("log_dir", "slurm/logs")).mkdir(parents=True, exist_ok=True)
    out.write_text(script, newline="\n")
    print(f"wrote {out} ({len(runs)} array tasks)")
    if a.dry_run:
        print(script)
        return 0
    res = subprocess.run(["sbatch", str(out)], capture_output=True, text=True)
    print(res.stdout or res.stderr)
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
