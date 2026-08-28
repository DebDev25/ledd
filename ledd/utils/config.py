"""Minimal YAML config loader with `_base_` inheritance and dotted overrides."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config, resolving a single `_base_` chain relative to the file."""
    path = os.path.abspath(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base_rel = cfg.pop("_base_", None)
    if base_rel:
        base_path = os.path.normpath(os.path.join(os.path.dirname(path), base_rel))
        cfg = _deep_merge(load_config(base_path), cfg)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply CLI overrides of the form `train.lr_new=1e-4`."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Bad override (expected key=value): {item}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


class Cfg(dict):
    """dict with attribute access, so cfg.train.lr_new works."""

    def __getattr__(self, item):
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, key, value):
        self[key] = value
