"""
config.py — YAML config loader with single-level inheritance.

Usage:
    cfg = load_config("configs/mvst_bts_plus.yaml")
    print(cfg.model.ast.hidden_size)  # → 768
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """
    Load a YAML config, resolving 'base:' inheritance recursively.

    A child config can declare `base: path/to/parent.yaml`.
    The parent is loaded first, then the child is merged on top
    (child values take precedence).

    Args:
        config_path: Path to the config file.
        overrides:   Optional list of dotlist overrides, e.g.
                     ["training.batch_size=64", "optimizer.base_lr=1e-4"]
    """
    config_path = Path(config_path)
    raw = _load_yaml(config_path)

    if "base" in raw:
        # Resolve the base path relative to the project root
        base_path = Path(raw.pop("base"))
        base_cfg = load_config(base_path)
        # Merge child on top of base
        cfg = OmegaConf.merge(base_cfg, OmegaConf.create(raw))
    else:
        cfg = OmegaConf.create(raw)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg


def save_config(cfg: DictConfig, output_path: str | Path) -> None:
    """Snapshot a config to disk (for experiment reproducibility)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path)