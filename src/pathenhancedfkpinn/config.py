"""Config handling: every run = configs/base.yaml + benchmark layers + command-line overrides.

The resolved config is hashed (sha256 of canonical JSON) and the hash is stored in each run's
env.json, so every result traces back to one exact config.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> dict:
    """set_dotted(cfg, "data.M", 64)."""
    cur = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
    return cfg


def parse_value(v: str) -> Any:
    """YAML scalar, with a float fallback: YAML 1.1 reads '1e-3' as a string."""
    out = yaml.safe_load(v)
    if isinstance(out, str):
        try:
            return float(out)
        except ValueError:
            return out
    return out


def parse_override(s: str) -> tuple[str, Any]:
    """'data.M=64' -> ('data.M', 64); 'data.dt=1e-3' -> ('data.dt', 0.001)."""
    k, _, v = s.partition("=")
    return k.strip(), parse_value(v)


def resolve(layers: list[str | Path | dict], overrides: list[str] | None = None) -> dict:
    """Merge YAML files / dicts in order, then apply dotted overrides."""
    cfg: dict = {}
    for layer in layers:
        cfg = deep_merge(cfg, layer if isinstance(layer, dict) else load_yaml(layer))
    for o in overrides or []:
        set_dotted(cfg, *parse_override(o))
    return cfg


def config_hash(cfg: dict) -> str:
    """Stable hash of a resolved config (key order and whitespace do not matter)."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
