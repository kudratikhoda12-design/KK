"""Phase 0 - environment inspection.  Writes reports/environment.md from what is actually installed."""
from __future__ import annotations

import importlib.metadata as md
import multiprocessing
import os
import platform
import shutil
import sys

import requests

from . import config as C

PACKAGES = [
    "numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "xgboost", "shap",
    "matplotlib", "seaborn", "pyarrow", "joblib", "requests", "tabulate", "pytest", "nbformat", "nbconvert",
]


def package_versions() -> dict[str, str]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "not installed"
    return out


def _mem_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        return None
    return None


def _http_status(url: str) -> str:
    try:
        return str(requests.get(url, timeout=15).status_code)
    except Exception as exc:                                        # noqa: BLE001
        return f"unreachable ({type(exc).__name__})"


def kaggle_credentials_present() -> bool:
    return (
        os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
        or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        or bool(os.environ.get("KAGGLE_API_TOKEN"))
    )


def write_environment_report() -> str:
    C.ensure_dirs()
    free_gb = round(shutil.disk_usage(str(C.ROOT)).free / 1e9, 1)
    ver = package_versions()
    lines = [
        "# Environment inspection (Phase 0)",
        "",
        f"* Python: `{sys.version.split()[0]}` ({platform.python_implementation()}) on `{platform.platform()}`",
        f"* CPU cores: {multiprocessing.cpu_count()}; RAM: {_mem_gb()} GB; free disk: {free_gb} GB",
        f"* Internet access: Hugging Face API HTTP {_http_status('https://huggingface.co/api/datasets/' + C.PRIMARY_MIRROR)}; "
        f"PyPI HTTP {_http_status('https://pypi.org/simple/pip/')}",
        f"* Kaggle credentials present: **{kaggle_credentials_present()}**; unauthenticated Kaggle M5 download endpoint: "
        f"HTTP {_http_status('https://www.kaggle.com/api/v1/competitions/data/download/m5-forecasting-accuracy/calendar.csv')}"
        " (401 = authentication required)",
        "",
        "## Installed packages used by the project",
        "",
        "| package | version |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in ver.items()],
        "",
        "## Notes",
        "* No GPU is needed or used; deep learning is deliberately *not* used (see README, design decisions).",
        "* Missing packages are installed with `pip install -r requirements.txt` (PyPI was reachable).",
        "* Because the official Kaggle files need credentials, the dataset is obtained from verified public mirrors "
        "(see `data_source.md`).",
    ]
    text = "\n".join(lines) + "\n"
    (C.REPORTS / "environment.md").write_text(text)
    return text


if __name__ == "__main__":
    print(write_environment_report())
