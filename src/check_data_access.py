"""Check which candidate data sources are reachable from YOUR network.

Phase 1 helper. It only sends small read-only requests and downloads one tiny
Binance file to show the checksum workflow. It does NOT download any
Polymarket or Kalshi data.

Do not use a VPN to get around a government block (e.g. India's May 2026
order against Polymarket). If a source is blocked, record it as blocked.

Usage:
    python src/check_data_access.py
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
import zipfile

TIMEOUT = 15

# (label, url). Any HTTP response, even 4xx, means the host is reachable.
# A connection error means it is not reachable from this network.
PROBES = [
    ("Binance Vision archive (recommended)",
     "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip.CHECKSUM"),
    ("Binance public REST mirror",
     "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1"),
    ("Polymarket Gamma API", "https://gamma-api.polymarket.com/markets?limit=1"),
    ("Polymarket Data API", "https://data-api.polymarket.com/trades?limit=1"),
    ("Polymarket CLOB API", "https://clob.polymarket.com/markets?limit=1"),
    ("Kalshi API", "https://api.elections.kalshi.com/trade-api/v2/markets?limit=1"),
    ("Hugging Face (Polymarket on-chain dump)",
     "https://huggingface.co/api/datasets/SII-WANGZJ/Polymarket_data"),
]

BINANCE_SAMPLE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"


def fetch(url: str) -> tuple[int | None, bytes, str]:
    """Return (http_status, body, error_message)."""
    req = urllib.request.Request(url, headers={"User-Agent": "research-access-check/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read(), ""
    except urllib.error.HTTPError as e:  # server answered, so the host is reachable
        return e.code, b"", ""
    except Exception as e:  # DNS failure, connection refused, TLS error, timeout...
        return None, b"", f"{type(e).__name__}: {e}"


def check_binance_checksum() -> None:
    """Download one small monthly file and verify its SHA-256 against Binance's CHECKSUM file."""
    status, checksum_body, err = fetch(BINANCE_SAMPLE + ".CHECKSUM")
    if status != 200:
        print(f"  checksum file not available ({status or err})")
        return
    expected = checksum_body.decode().split()[0]

    status, zip_body, err = fetch(BINANCE_SAMPLE)
    if status != 200:
        print(f"  sample zip not available ({status or err})")
        return
    actual = hashlib.sha256(zip_body).hexdigest()
    print(f"  SHA-256 match: {actual == expected}")

    with zipfile.ZipFile(io.BytesIO(zip_body)) as zf:
        rows = zf.read(zf.namelist()[0]).decode().strip().splitlines()
    # January 2024 has 31 days * 24 hours = 744 hourly candles if nothing is missing.
    print(f"  rows in BTCUSDT 1h 2024-01: {len(rows)} (744 expected if no gaps)")
    print(f"  first row: {rows[0]}")


def main() -> None:
    print(f"{'Source':45s} {'Reachable':10s} Detail")
    for label, url in PROBES:
        status, _, err = fetch(url)
        reachable = "yes" if status is not None else "NO"
        print(f"{label:45s} {reachable:10s} {('HTTP ' + str(status)) if status else err[:80]}")

    print("\nBinance checksum workflow:")
    check_binance_checksum()


if __name__ == "__main__":
    main()
