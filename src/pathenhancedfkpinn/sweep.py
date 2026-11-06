"""Run specs: config layers, then --set overrides, then one grid point, then one seed.

Run directories: results/<name>/<grid slug>/seed<k>; smoke runs under results/smoke/<name>/.
Order is grid point (outer) x seed (inner), so Slurm array task i runs runs[i].
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from pathenhancedfkpinn.config import ROOT, config_hash, parse_value, resolve, set_dotted


@dataclass
class Run:
    name: str
    seed: int
    grid_point: dict
    cfg: dict
    run_dir: Path


def parse_grid(items: list[str] | None) -> dict[str, list]:
    """['problem.k=1,4,16'] -> {'problem.k': [1, 4, 16]}."""
    grid = {}
    for s in items or []:
        k, _, v = s.partition("=")
        grid[k.strip()] = [parse_value(x) for x in v.split(",")]
    return grid


def grid_points(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def slug(gp: dict) -> str:
    if not gp:
        return "default"
    return "_".join(f"{k.split('.')[-1]}{v}" for k, v in gp.items()).replace(" ", "")


def default_name(layers: list[str], cfg: dict) -> str:
    """configs/heat1d/legacy_parity.yaml + arm P -> 'heat1d-legacy_parity__P'; configs/base.yaml -> 'base__P'."""
    last = Path(layers[-1])
    stem = last.stem if last.parent.name == "configs" else f"{last.parent.name}-{last.stem}"
    return f"{stem}__{cfg['arm']}"


def expand(layers: list[str], overrides: list[str] | None = None, grid: dict | None = None,
           seeds: list[int] | None = None, name: str | None = None, smoke: bool = False,
           results_root: Path | None = None) -> list[Run]:  # fmt: skip
    """All runs of a spec. Layer paths are relative to the repo root. Default seeds: cfg['seed']."""
    results_root = Path(results_root or ROOT / "results")
    base_cfg = resolve([ROOT / p for p in layers], overrides or [])
    name = name or default_name(layers, base_cfg)
    seeds = [int(s) for s in seeds] if seeds else [int(base_cfg["seed"])]
    gps = grid_points(grid or {})
    if smoke:
        gps, seeds = gps[:1], seeds[:1]
    base = results_root / "smoke" / name if smoke else results_root / name
    runs = []
    for gp in gps:
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            for k, v in gp.items():
                set_dotted(cfg, k, v)
            cfg["seed"] = seed
            runs.append(Run(name, seed, gp, cfg, base / slug(gp) / f"seed{seed}"))
    return runs


def holds_other_config(run: Run) -> bool:
    """True if run_dir already holds a run with a different resolved config (unsafe to resume)."""
    env = run.run_dir / "env.json"
    return env.exists() and json.loads(env.read_text()).get("config_hash") != config_hash(run.cfg)


def spec_parser() -> argparse.ArgumentParser:
    """Run-spec arguments shared by scripts/run.py and scripts/submit.py. Layers come first."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("layers", nargs="+", help="config layers, merged in order (paths from repo root)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, e.g. arm=D data.M=64")
    ap.add_argument("--grid", nargs="*", default=[], metavar="KEY=V1,V2", help="sweep: product over keys")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="default: seed in the config")
    ap.add_argument("--name", default=None, help="results/<name>/; default from last layer and arm")
    ap.add_argument("--smoke", action="store_true", help="<= 200 steps, first seed and grid point")
    return ap


def from_args(a: argparse.Namespace) -> list[Run]:
    return expand(a.layers, a.set, parse_grid(a.grid), a.seeds, a.name, a.smoke)
