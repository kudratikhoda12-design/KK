"""Data acquisition from the Binance public archive (https://data.binance.vision).

URL convention (USD-M futures), verified 2026-10-07:

    {base}/{granularity}/{dataset}/{SYMBOL}/{SYMBOL}-{dataset}-{period}.zip
    {base}/{granularity}/{dataset}/{SYMBOL}/{SYMBOL}-{dataset}-{period}.zip.CHECKSUM

    base        = https://data.binance.vision/data/futures/um
    granularity = monthly (period = YYYY-MM) | daily (period = YYYY-MM-DD)
    aggTrades, fundingRate -> monthly;  bookDepth, metrics -> daily only

Each .CHECKSUM file holds "<sha256>  <filename>". Every download is verified against it and
recorded in a manifest (CSV) so that the exact raw inputs of every run are auditable.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteFile:
    dataset: str
    granularity: str
    symbol: str
    period: str          # YYYY-MM or YYYY-MM-DD
    base_url: str

    @property
    def filename(self) -> str:
        return f"{self.symbol}-{self.dataset}-{self.period}.zip"

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.granularity}/{self.dataset}/{self.symbol}/{self.filename}"

    @property
    def checksum_url(self) -> str:
        return self.url + ".CHECKSUM"

    def local_path(self, raw_dir: Path) -> Path:
        return raw_dir / self.dataset / self.symbol / self.filename


def periods(start: str, end: str, granularity: str) -> list[str]:
    """All monthly ('YYYY-MM') or daily ('YYYY-MM-DD') periods in [start, end]."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out: list[str] = []
    if granularity == "monthly":
        y, m = s.year, s.month
        while (y, m) <= (e.year, e.month):
            out.append(f"{y:04d}-{m:02d}")
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    elif granularity == "daily":
        d = s
        while d <= e:
            out.append(d.isoformat())
            d = date.fromordinal(d.toordinal() + 1)
    else:
        raise ValueError(granularity)
    return out


def remote_files(cfg: dict, dataset: str, symbol: str | None = None) -> list[RemoteFile]:
    spec = cfg["datasets"][dataset]
    sym = symbol or cfg["source"]["symbol"]
    return [RemoteFile(dataset, spec["granularity"], sym, p, cfg["source"]["base_url"])
            for p in periods(cfg["period"]["start"], cfg["period"]["end"], spec["granularity"])]


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


@dataclass
class DownloadRecord:
    dataset: str
    period: str
    filename: str
    url: str
    status: str                 # ok | cached | missing_remote | checksum_mismatch | error
    bytes: int = 0
    sha256_expected: str = ""
    sha256_actual: str = ""
    downloaded_at_utc: str = ""
    message: str = ""


_session = requests.Session()


def _get(url: str, timeout: float, retries: int, stream: bool = False) -> requests.Response | None:
    """GET with exponential back-off on network errors / 5xx. Returns None on 404."""
    for attempt in range(retries + 1):
        try:
            r = _session.get(url, timeout=timeout, stream=stream)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500:
                raise
            if attempt == retries:
                raise
            wait = 2 ** (attempt + 1)
            log.warning("GET %s failed (%s); retry in %ss", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def download(rf: RemoteFile, raw_dir: Path, timeout: float = 300, retries: int = 4) -> DownloadRecord:
    """Download one archive file and verify its SHA-256 against the published .CHECKSUM.

    If a local copy already exists and matches the checksum it is reused (status 'cached').
    A mismatching file is deleted and reported, never silently used.
    """
    rec = DownloadRecord(rf.dataset, rf.period, rf.filename, rf.url, status="error")
    dest = rf.local_path(raw_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        cr = _get(rf.checksum_url, timeout, retries)
        if cr is None:
            rec.status, rec.message = "missing_remote", "no .CHECKSUM (file not published)"
            return rec
        rec.sha256_expected = cr.text.split()[0].strip().lower()

        if dest.exists() and sha256_file(dest) == rec.sha256_expected:
            rec.status, rec.bytes, rec.sha256_actual = "cached", dest.stat().st_size, rec.sha256_expected
            return rec

        tmp = dest.with_suffix(".part")
        for attempt in range(retries + 1):   # retry the whole transfer: connections can drop mid-body
            r = _get(rf.url, timeout, retries, stream=True)
            if r is None:
                rec.status, rec.message = "missing_remote", "checksum present but zip 404"
                return rec
            h, n = hashlib.sha256(), 0
            try:
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
                        h.update(chunk)
                        n += len(chunk)
                break
            except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                if attempt == retries:
                    raise
                log.warning("transfer of %s broken (%s); retry %d", rf.filename, exc, attempt + 1)
                time.sleep(2 ** (attempt + 1))
        rec.bytes, rec.sha256_actual = n, h.hexdigest()
        rec.downloaded_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if rec.sha256_actual != rec.sha256_expected:
            tmp.unlink(missing_ok=True)
            rec.status, rec.message = "checksum_mismatch", "deleted; not used"
            log.error("Checksum mismatch for %s", rf.filename)
            return rec
        tmp.replace(dest)
        rec.status = "ok"
        return rec
    except Exception as exc:  # recorded in the manifest, never swallowed silently
        rec.message = repr(exc)
        log.exception("Download failed: %s", rf.url)
        return rec


def download_many(files: list[RemoteFile], raw_dir: Path, workers: int = 8, **kw) -> list[DownloadRecord]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda rf: download(rf, raw_dir, **kw), files))


MANIFEST_FIELDS = list(DownloadRecord.__dataclass_fields__)


def append_manifest(path: Path, records: list[DownloadRecord]) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        for r in records:
            w.writerow(asdict(r))
