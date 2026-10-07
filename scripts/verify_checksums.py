"""Check that locally downloaded Binance archives are the exact files used in the study.

Every archive is already verified against Binance's own .CHECKSUM at download
time (src/data.py). This script adds a second check: it compares the SHA-256 of
each local file with the reference manifests in reproduction/checksums/, which
record the files the published results were computed from. If Binance ever
revises an archive, this is where it shows up.

    python scripts/verify_checksums.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reproduction" / "checksums"
SETS = {
    "BTCUSDT_1m_spot_manifest.csv": ROOT / "data/raw/spot/BTCUSDT/1m",
    "ETHUSDT_1m_spot_manifest.csv": ROOT / "data/raw/spot/ETHUSDT/1m",
    "BTCUSDT_fundingRate_manifest.csv": ROOT / "data/raw/um/BTCUSDT/fundingRate",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    problems = 0
    for manifest, folder in SETS.items():
        ref = pd.read_csv(REF / manifest)
        present = mismatched = 0
        missing = []
        for row in ref.itertuples():
            f = folder / row.file
            if not f.exists():
                missing.append(row.file)
                continue
            present += 1
            if sha256(f) != row.sha256:
                mismatched += 1
                print(f"  MISMATCH {f.relative_to(ROOT)}")
        problems += mismatched + len(missing)
        status = "OK" if not mismatched and not missing else "CHECK"
        print(f"[{status}] {manifest}: {present}/{len(ref)} files present, {mismatched} hash mismatches"
              + (f", {len(missing)} missing (download them first)" if missing else ""))
    print("All files match the study's reference hashes." if problems == 0 else
          "Some files are missing or differ from the files used in the study.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
