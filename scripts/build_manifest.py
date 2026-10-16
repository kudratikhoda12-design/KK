"""Write reports/run_manifest.json: environment, hardware, git, data provenance, seeds, configs, commands, stages."""
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()


lock = dict(l.split("==") for l in (ROOT / "requirements-lock.txt").read_text().split() if "==" in l)
cmds = [json.loads(l) for l in (ROOT / "reports/logs/commands.jsonl").read_text().splitlines() if l.strip()]
stages = json.loads((ROOT / "reports/stage_status.json").read_text())
manifest = {
    "project": "KAN-FAE research prototype",
    "generated_by": "scripts/build_manifest.py",
    "git": {"branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"), "head": sh("git", "rev-parse", "HEAD"),
            "pushed_to_remote": False, "note": "no push to any remote (owner instruction, D21)"},
    "environment": {"python": platform.python_version(), "platform": platform.platform(),
                    "venv": ".venv (project-local)", "lock_file": "requirements-lock.txt",
                    "lock_sha256": sha(ROOT / "requirements-lock.txt"),
                    "key_packages": {k: lock.get(k) for k in ("torch", "numpy", "scipy", "pandas", "scikit-learn",
                                                             "lifelines", "scikit-survival", "matplotlib", "rdata")},
                    "torch_build": "CPU only (torch.version.cuda is None)"},
    "hardware": {"cpus": os.cpu_count(), "memory_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1),
                 "gpu": None},
    "data": {
        "synthetic": {"generator": "src/kanfae/data/synthetic.py", "replicate_seeds": [1000, 1001, 1002, 1003, 1004],
                      "gate_seed": 21, "kind": "SYNTHETIC"},
        "cmapss": {"source_page": "https://data.phmsociety.org/nasa/",
                   "url": "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
                   "outer_zip_sha256": sha(ROOT / "data/raw/cmapss_download/cmapss.zip"),
                   "inner_zip_sha256": sha(ROOT / "data/raw/cmapss_download/CMAPSSData.zip"),
                   "files_sha256": {p.name: sha(p) for p in sorted((ROOT / "data/raw/cmapss").glob("*FD001.txt"))},
                   "citation": "A. Saxena and K. Goebel (2008). Turbofan Engine Degradation Simulation Data Set, "
                               "NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA",
                   "kind": "REAL (simulated turbofan run-to-failure; RUL benchmark; not naturally censored)"},
        "pbc": {"url": "https://cran.r-project.org/src/contrib/survival_3.8-12.tar.gz",
                "tarball_sha256": sha(ROOT / "data/raw/pbc_download/survival_3.8-12.tar.gz"),
                "pbc_rda_sha256": sha(ROOT / "data/raw/pbc/data/pbc.rda"), "object": "pbcseq",
                "license": "LGPL (>= 2) (package DESCRIPTION)", "kind": "REAL (right-censored, repeated measures)"},
    },
    "configs": sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs").rglob("*.json")),
    "seeds": {"tuning": 0, "final": [0, 1, 2], "note": "stochastic models (RSF) use seeds 0-2; deterministic models run once"},
    "stages": stages,
    "commands": cmds,
    "deliverables": {k: str(v) for k, v in {
        "pdf": "reports/final_research_prototype_report.pdf", "markdown": "reports/final_research_prototype_report.md",
        "test_results": "reports/test_results.csv", "test_results_latest": "reports/test_results_latest.csv",
        "experiment_results": "reports/experiment_results.csv", "experiment_records": "reports/experiment_results.jsonl",
        "manifest": "reports/run_manifest.json", "logs": "reports/logs/", "figures": "reports/figures/",
        "tables": "reports/tables/", "final_status": "reports/FINAL_STATUS.md"}.items()},
}
(ROOT / "reports/run_manifest.json").write_text(json.dumps(manifest, indent=1))
print("manifest written:", len(cmds), "commands")
