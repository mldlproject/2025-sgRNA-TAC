"""Configuration utilities for sgRNA-TAC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Config:
    """Typed wrapper around the YAML configuration file."""

    raw: Dict[str, Any]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def optimizer(self) -> Dict[str, Any]:
        return self.raw.get("optimizer", {})

    @property
    def paths(self) -> Dict[str, Any]:
        return self.raw.get("paths", {})

    @property
    def device(self) -> str:
        return str(self.raw.get("device", "auto")).lower()


def load_config(path: str | Path) -> Config:
    """Load a YAML config file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)

    return Config(raw=data)

