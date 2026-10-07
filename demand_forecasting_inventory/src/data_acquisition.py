"""Phase 1 - dataset acquisition with integrity verification.

Why not the official source?  The official M5 data are distributed through Kaggle and need an
authenticated API call (HTTP 401 without credentials; no ~/.kaggle/kaggle.json was available).
We therefore use a *legitimate public mirror* on Hugging Face that hosts the unmodified Kaggle
files, and we verify the files two ways:

1. SHA-256 (git-LFS object id) of every file equals the value published by the mirror's API;
2. a second, independent mirror (different uploader) publishes byte-identical hashes.

Raw files are never modified; provenance is written to reports/ and outputs/tables/.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from . import config as C


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """SHA-1 of the file as a git blob (what Hugging Face reports for non-LFS files)."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch_mirror_manifest(repo: str, timeout: int = 60) -> dict[str, dict]:
    """File name -> {size, lfs_sha256 | None, blob_id} as published by the Hugging Face API."""
    r = requests.get(C.HF_API.format(repo=repo), timeout=timeout)
    r.raise_for_status()
    out = {}
    for s in r.json().get("siblings", []):
        lfs = s.get("lfs") or {}
        out[s["rfilename"]] = {"size": s.get("size"), "lfs_sha256": lfs.get("sha256"), "blob_id": s.get("blobId")}
    return out


def download_file(repo: str, filename: str, dest: Path, retries: int = 3, timeout: int = 900) -> None:
    url = C.HF_RESOLVE.format(repo=repo, filename=filename)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for block in r.iter_content(chunk_size=1 << 20):
                        f.write(block)
            os.replace(tmp, dest)
            return
        except Exception:                                   # noqa: BLE001 - retry on any transport error
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def _verify_one(path: Path, name: str, primary: dict, check: dict) -> dict:
    """Compare local hashes with both mirrors' published hashes."""
    meta, meta2 = primary[name], check.get(name, {})
    local_size = path.stat().st_size
    row = {
        "file": name,
        "local_bytes": local_size,
        "size_matches_primary": local_size == meta["size"],
        "size_matches_crosscheck": local_size == meta2.get("size"),
    }
    if meta.get("lfs_sha256"):
        local = sha256_file(path)
        row.update(
            hash_type="sha256(lfs)",
            local_hash=local,
            hash_matches_primary=local == meta["lfs_sha256"],
            hash_matches_crosscheck=local == meta2.get("lfs_sha256"),
        )
    else:
        local = git_blob_sha1(path)
        row.update(
            hash_type="git-blob-sha1",
            local_hash=local,
            hash_matches_primary=local == meta["blob_id"],
            hash_matches_crosscheck=local == meta2.get("blob_id"),
        )
        row["sha256_local"] = sha256_file(path)
    return row


def acquire(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Download (if missing) and verify the M5 files.  Returns the verification table."""
    C.ensure_dirs()
    primary = fetch_mirror_manifest(C.PRIMARY_MIRROR)
    check = fetch_mirror_manifest(C.CROSSCHECK_MIRROR)
    rows = []
    for key, name in C.RAW_FILES.items():
        dest = C.DATA_RAW / name
        if force or not dest.exists():
            if verbose:
                print(f"[acquire] downloading {name} from {C.PRIMARY_MIRROR} ...")
            download_file(C.PRIMARY_MIRROR, name, dest)
        rows.append(_verify_one(dest, name, primary, check))
    ver = pd.DataFrame(rows)
    ver.to_csv(C.OUT_TAB / "raw_data_verification.csv", index=False)
    ok = ver[["size_matches_primary", "size_matches_crosscheck", "hash_matches_primary", "hash_matches_crosscheck"]].all().all()
    if not ok:
        raise RuntimeError("Raw-data verification FAILED - do not use these files:\n" + ver.to_string())
    if verbose:
        print("[acquire] all raw files verified against two independent mirrors")
    return ver


def structural_check() -> dict:
    """Check the documented structure of the M5 release (row/column counts, id uniqueness)."""
    cal = pd.read_csv(C.DATA_RAW / C.RAW_FILES["calendar"], usecols=["date", "d"])
    head = pd.read_csv(C.DATA_RAW / C.RAW_FILES["sales_evaluation"], nrows=2)
    n_days = sum(c.startswith("d_") for c in head.columns)
    return {
        "calendar_rows": len(cal),
        "calendar_first_date": cal["date"].iloc[0],
        "calendar_last_date": cal["date"].iloc[-1],
        "sales_eval_day_columns": n_days,
    }


def write_source_report(ver: pd.DataFrame) -> Path:
    s = structural_check()
    lines = [
        "# Data source and provenance",
        "",
        "## Dataset",
        "**M5 Forecasting - Accuracy** (Makridakis Open Forecasting Center, University of Nicosia; Walmart unit "
        "sales by item, store and day, 2011-01-29 to 2016-06-19, with calendar/event/SNAP information and weekly "
        "selling prices).",
        "",
        "## Why a mirror was used (and why it is legitimate)",
        "* The official distribution is the Kaggle competition `m5-forecasting-accuracy`. Downloading it requires "
        "Kaggle credentials; an unauthenticated request returns **HTTP 401**, and no `~/.kaggle/kaggle.json` or "
        "`KAGGLE_*` variables exist in this environment.",
        "* We therefore used the public Hugging Face dataset repositories "
        f"`{C.PRIMARY_MIRROR}` (primary) and `{C.CROSSCHECK_MIRROR}` (cross-check). Both host the four competition "
        "files under their original names.",
        "* **Integrity verification** (see `outputs/tables/raw_data_verification.csv`): every downloaded file has "
        "(i) the SHA-256 published by the primary mirror's API, and (ii) a hash identical to the value published by "
        "the independent second mirror. Calendar (a plain git blob) is verified through its git blob SHA-1.",
        "* **Limitation of this evidence:** Kaggle's own checksums are not accessible without credentials, so we cannot "
        "prove byte-equality with the official download. Our evidence is agreement between two independent uploads "
        "plus the structural checks below and in `data_inspection.md` (e.g. 30,490 series, 1,941 days, 6,841,121 price rows, "
        "validation file = prefix of the evaluation file).",
        "* Structural check against the documented M5 release: "
        f"calendar has {s['calendar_rows']} days ({s['calendar_first_date']} to {s['calendar_last_date']}); the "
        f"evaluation sales file has {s['sales_eval_day_columns']} observed day columns (d_1 ... d_{s['sales_eval_day_columns']}).",
        "",
        "## Verification table",
        "",
        ver.drop(columns=[c for c in ("sha256_local",) if c in ver.columns]).to_markdown(index=False),
        "",
        "## Files used",
        "* `calendar.csv`, `sell_prices.csv`, `sales_train_evaluation.csv` (all 1,941 observed days). "
        "`sales_train_validation.csv` is a strict prefix of the evaluation file (d_1 ... d_1913) and is only used "
        "for a consistency check.",
        "* The 28 days after d_1941 are *not* in the public data (held out by the competition) - calendar and prices "
        "for them exist but sales do not.",
        "",
        "## Differences from the competition set-up",
        "* The competition's hidden evaluation period is not available; we run our own chronological "
        "train/validation/test split inside the 1,941 observed days (see `validation_strategy.md`).",
        "* Our metrics are MAE/RMSE/sMAPE/WAPE on the sums of daily demand, not the competition's WRMSSE.",
        "",
        "## Licence / attribution",
        "M5 data are released by the organisers for research; cite: Makridakis, Spiliotis & Assimakopoulos (2022), "
        "*The M5 competition: Background, organization, and implementation*, International Journal of Forecasting 38(4).",
    ]
    p = C.REPORTS / "data_source.md"
    p.write_text("\n".join(lines) + "\n")
    return p


if __name__ == "__main__":
    v = acquire()
    write_source_report(v)
    print(json.dumps(structural_check(), indent=2))
