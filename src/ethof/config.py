"""Configuration loading and canonical data-directory layout."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    data_dir = Path(os.environ.get("ETHOF_DATA_DIR", cfg["data_dir"]))
    cfg["data_dir"] = data_dir if data_dir.is_absolute() else REPO_ROOT / data_dir
    return cfg


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path: return self.root / "raw"
    @property
    def processed(self) -> Path: return self.root / "processed"
    @property
    def reports(self) -> Path: return self.root / "reports"
    @property
    def logs(self) -> Path: return self.root / "logs"
    @property
    def manifest(self) -> Path: return self.root / "manifest.csv"

    def make(self) -> "Paths":
        for p in (self.raw, self.processed, self.reports, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
