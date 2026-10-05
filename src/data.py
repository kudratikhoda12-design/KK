"""Acquire and load Binance Vision kline data.

Design rules
------------
* Raw zip files are stored exactly as downloaded, each verified against the
  SHA-256 in Binance's companion ``.CHECKSUM`` file.
* Every download is recorded in ``data/raw/<market>/<symbol>/<interval>/manifest.csv``
  (URL, size, hash, check result, UTC download time) so the dataset can be
  re-created and audited later. Binance occasionally revises archived files.
* Loading does NOT clean anything. It only parses and types. Cleaning
  decisions live in ``quality.py`` and are documented in the audit report.
* Timestamp units are detected per file: Binance spot files use milliseconds
  before 2025-01-01 and microseconds from 2025-01-01 onwards.

Command line
------------
    python -m src.data download --symbol BTCUSDT --start 2017-08 --end 2026-09
    python -m src.data build    --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

BASE_URL = "https://data.binance.vision/data"
USER_AGENT = "isi-research-project/1.0 (academic, non-commercial)"
MANIFEST_FIELDS = ["file", "url", "granularity", "bytes", "sha256", "expected_sha256",
                   "checksum_ok", "downloaded_utc"]


# --------------------------------------------------------------------------
# URLs and file names
# --------------------------------------------------------------------------
def month_list(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings."""
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def kline_url(symbol: str, interval: str, period: str, market: str = "spot") -> str:
    """URL of a monthly ('YYYY-MM') or daily ('YYYY-MM-DD') kline archive."""
    granularity = "monthly" if len(period) == 7 else "daily"
    prefix = {"spot": "spot", "um": "futures/um"}[market]
    return (f"{BASE_URL}/{prefix}/{granularity}/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{period}.zip")


def funding_url(symbol: str, month: str) -> str:
    return f"{BASE_URL}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"


def raw_dir(symbol: str, interval: str, market: str = "spot") -> Path:
    return config.RAW_DIR / market / symbol / interval


# --------------------------------------------------------------------------
# Download with checksum verification
# --------------------------------------------------------------------------
def _http_get(url: str, retries: int = 4, timeout: int = 60) -> bytes | None:
    """GET with exponential back-off. Returns None on HTTP 404 (file does not exist)."""
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(delay)
        delay *= 2
    return None


def _append_manifest(manifest: Path, row: dict) -> None:
    new = not manifest.exists()
    with manifest.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def download_verified(url: str, dest_dir: Path, granularity: str, overwrite: bool = False) -> str:
    """Download one archive plus its CHECKSUM and verify SHA-256.

    Returns 'ok', 'exists', 'missing' (404) or 'checksum_mismatch'.
    A file that fails verification is kept with a '.bad' suffix for inspection,
    never silently used.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[1]
    dest = dest_dir / name
    if dest.exists() and not overwrite:
        return "exists"

    checksum = _http_get(url + ".CHECKSUM")
    body = _http_get(url)
    if body is None:
        return "missing"

    actual = hashlib.sha256(body).hexdigest()
    expected = checksum.decode().split()[0] if checksum else ""
    ok = bool(expected) and actual == expected
    target = dest if ok else dest.with_suffix(".zip.bad")
    target.write_bytes(body)
    _append_manifest(dest_dir / "manifest.csv", dict(
        file=target.name, url=url, granularity=granularity, bytes=len(body), sha256=actual,
        expected_sha256=expected, checksum_ok=ok,
        downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    ))
    return "ok" if ok else "checksum_mismatch"


def download_klines(symbol: str, start: str, end: str, interval: str = "1m",
                    market: str = "spot", pause: float = 0.2) -> pd.DataFrame:
    """Download monthly archives; if a month is not published, fall back to daily files.

    Returns a status table (one row per month) so missing months are visible.
    """
    out = raw_dir(symbol, interval, market)
    status = []
    for month in month_list(start, end):
        s = download_verified(kline_url(symbol, interval, month, market), out, "monthly")
        if s == "missing":
            # Monthly files appear a few days after month end; daily files the next day.
            days = pd.date_range(f"{month}-01", periods=pd.Period(month).days_in_month, freq="D")
            daily = [download_verified(kline_url(symbol, interval, d.strftime("%Y-%m-%d"), market),
                                       out, "daily") for d in days]
            s = f"daily:{sum(x in ('ok', 'exists') for x in daily)}/{len(days)}"
        status.append({"month": month, "status": s})
        print(f"{symbol} {month}: {s}", flush=True)
        time.sleep(pause)   # be polite to the archive (Vision Dataset Terms s.7)
    return pd.DataFrame(status)


def download_funding(symbol: str, start: str, end: str, pause: float = 0.2) -> pd.DataFrame:
    out = config.RAW_DIR / "um" / symbol / "fundingRate"
    status = []
    for month in month_list(start, end):
        s = download_verified(funding_url(symbol, month), out, "monthly")
        status.append({"month": month, "status": s})
        time.sleep(pause)
    return pd.DataFrame(status)


# --------------------------------------------------------------------------
# Loading (parse + type only, no cleaning)
# --------------------------------------------------------------------------
def detect_time_unit(values: np.ndarray) -> str:
    """'ms' for 13-digit epoch values, 'us' for 16-digit values.

    Epoch milliseconds for 2017-2030 are ~1.5e12-1.9e12; microseconds are
    ~1.5e15-1.9e15. Anything else is an error, not something to guess about.
    """
    v = np.nanmedian(values.astype("float64"))
    if 1e12 <= v < 1e13:
        return "ms"
    if 1e15 <= v < 1e16:
        return "us"
    raise ValueError(f"Unrecognised epoch magnitude {v:.3e}")


def read_kline_zip(path: Path) -> pd.DataFrame:
    """Parse one Binance kline zip into a typed DataFrame (no rows removed)."""
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{path.name}: expected exactly one CSV, found {members}")
        raw = zf.read(members[0])

    first_field = raw.split(b",", 1)[0].strip()
    has_header = not first_field.isdigit()   # some Binance files carry a header row
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if df.shape[1] != len(config.KLINE_COLUMNS):
        raise ValueError(f"{path.name}: {df.shape[1]} columns, expected {len(config.KLINE_COLUMNS)}")
    df.columns = config.KLINE_COLUMNS

    unit = detect_time_unit(df["open_time"].to_numpy())
    out = pd.DataFrame({
        "open_time": pd.to_datetime(df["open_time"].astype("int64"), unit=unit, utc=True),
        "close_time": pd.to_datetime(df["close_time"].astype("int64"), unit=unit, utc=True),
        "raw_open_time": df["open_time"].astype("int64"),   # kept for the timestamp audit
    })
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"]:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    out["n_trades"] = pd.to_numeric(df["n_trades"], errors="coerce").astype("Int64")
    out["ts_unit"] = unit
    out["row_in_file"] = np.arange(len(out), dtype="int64")   # original order, for the audit
    out["source_file"] = path.name
    out["has_header"] = has_header
    return out


def build_interim(symbol: str, interval: str = "1m", market: str = "spot") -> Path:
    """Combine all verified zips into one parquet file, sorted, NOT de-duplicated."""
    files = sorted(raw_dir(symbol, interval, market).glob(f"{symbol}-{interval}-*.zip"))
    if not files:
        raise FileNotFoundError(
            f"No zip files in {raw_dir(symbol, interval, market)}. Run the download step first "
            "(see data/README.md).")
    df = pd.concat([read_kline_zip(f) for f in files], ignore_index=True)
    df["ts_unit"] = df["ts_unit"].astype("category")
    df["source_file"] = df["source_file"].astype("category")
    df = df.sort_values(["open_time", "source_file"], kind="stable").reset_index(drop=True)
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = config.INTERIM_DIR / f"{symbol}_{interval}_raw.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} with {len(df):,} rows from {len(files)} files")
    return out


def load_interim(symbol: str = config.SYMBOL, interval: str = "1m") -> pd.DataFrame:
    path = config.INTERIM_DIR / f"{symbol}_{interval}_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for how to obtain the data.")
    return pd.read_parquet(path)


def load_processed(symbol: str = config.SYMBOL, interval: str = "1m") -> pd.DataFrame:
    """Audited 1-minute grid (output of quality.build_processed)."""
    path = config.PROCESSED_DIR / f"{symbol}_{interval}_grid.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the audit stage first.")
    return pd.read_parquet(path)


def load_funding(symbol: str = config.SYMBOL) -> pd.DataFrame | None:
    """Funding-rate history (USD-M perpetual) or None if not downloaded."""
    files = sorted((config.RAW_DIR / "um" / symbol / "fundingRate").glob("*.zip"))
    if not files:
        return None
    frames = []
    for f in files:
        with zipfile.ZipFile(f) as zf:
            frames.append(pd.read_csv(zf.open(zf.namelist()[0])))
    df = pd.concat(frames, ignore_index=True)
    time_col = [c for c in df.columns if "time" in c.lower()][0]
    rate_col = [c for c in df.columns if "rate" in c.lower()][0]
    unit = detect_time_unit(df[time_col].to_numpy())
    return pd.DataFrame({
        "funding_time": pd.to_datetime(df[time_col].astype("int64"), unit=unit, utc=True),
        "funding_rate": df[rate_col].astype("float64"),
    }).sort_values("funding_time").reset_index(drop=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--symbol", default=config.SYMBOL)
    d.add_argument("--start", default=config.DATA_START_MONTH)
    d.add_argument("--end", default=config.DATA_END_MONTH)
    d.add_argument("--interval", default=config.INTERVAL)
    d.add_argument("--funding", action="store_true", help="also download USD-M funding rates")
    b = sub.add_parser("build")
    b.add_argument("--symbol", default=config.SYMBOL)
    b.add_argument("--interval", default=config.INTERVAL)
    args = p.parse_args()

    if args.cmd == "download":
        st = download_klines(args.symbol, args.start, args.end, args.interval)
        bad = st[~st["status"].isin(["ok", "exists"])]
        print(f"\n{len(st)} months requested; {len(bad)} not complete as monthly files:")
        print(bad.to_string(index=False) if len(bad) else "  none")
        if args.funding:
            download_funding(args.symbol, max(args.start, "2019-09"), args.end)
    else:
        build_interim(args.symbol, args.interval)


if __name__ == "__main__":
    main()
