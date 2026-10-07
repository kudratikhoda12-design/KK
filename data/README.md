# Data

Raw data is **not committed** to this repository, for two reasons:
1. Size.
2. The Binance Vision Dataset Terms license it under CC BY-NC-SA 4.0. Personal academic research and backtesting are allowed; commercial use is not.

Everything here is re-created by script, and every file is checked against Binance's published SHA-256 hash.

## What is needed

| Item | Value |
|---|---|
| Source | Binance Vision public archive, <https://data.binance.vision/> |
| Market / symbol / interval | spot / **BTCUSDT** / **1m** |
| Months | 2017-08 to 2026-09 (110 monthly zip files) |
| File pattern | `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip` (+ `.CHECKSUM`) |
| Columns (no header) | open_time, open, high, low, close, volume, close_time, quote_volume, n_trades, taker_buy_base, taker_buy_quote, ignore |
| Timestamp unit | **milliseconds before 2025-01-01, microseconds from 2025-01-01**. Detected automatically per file. |
| Expected rows | 4,797,840 one-minute candles from 2017-08-17 04:00 to 2026-09-30 23:59 if nothing were missing. Measured: 4,789,279 rows, of which 21,602 are off the minute grid (see audit). |
| Disk (measured) | BTCUSDT zips 235 MB; ETHUSDT zips 219 MB; funding 0.1 MB; BTCUSDT interim parquet 356 MB |
| Memory (estimate) | ≈ 0.5 GB for the full typed 1-minute table in pandas (13 columns × 8 bytes × 4.8M rows). |
| Also used | ETHUSDT 1m (110 files, second-asset replication); BTCUSDT USD-M `fundingRate` (81 files, 2020-01 → 2026-09, cost model) |

## Option A: let the cloud session download it (preferred)

Allow `data.binance.vision` in the cloud environment's network settings: environment menu → Edit → Network access → Custom, add `data.binance.vision`, and keep the default package-manager list. Then tell Claude to continue; the pipeline downloads and verifies everything itself.

## Option B: download on your own machine

```bash
git clone https://github.com/kudratikhoda12-design/KK.git && cd KK
git checkout claude/gifted-franklin-fgah4a
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Check that your network can reach Binance Vision (also verifies one checksum)
python src/check_data_access.py

# 2. Download + verify all 110 months (resumable: re-running skips finished files)
python -m src.data download --symbol BTCUSDT --start 2017-08 --end 2026-09

# 3. Combine into one typed parquet file (no cleaning happens here)
python -m src.data build --symbol BTCUSDT
#    -> data/interim/BTCUSDT_1m_raw.parquet

# 4. Funding rates (cost model) and ETHUSDT (replication)
python -m src.data download --symbol BTCUSDT --funding     # klines already present are skipped
python -m src.data download --symbol ETHUSDT
python -m src.data build --symbol ETHUSDT
```

On Linux or macOS, `bash scripts/download_data.sh BTCUSDT 2017-08 2026-09` is a shell-only alternative for step 2.

### Getting the file to the cloud session

Upload **`data/interim/BTCUSDT_1m_raw.parquet`** together with the **`manifest.csv`** from `data/raw/spot/BTCUSDT/1m/`, the record of every file's hash and download time. Place them at:

```
data/interim/BTCUSDT_1m_raw.parquet
data/raw/spot/BTCUSDT/1m/manifest.csv
```

You can also run the whole pipeline locally: `python scripts/run_pipeline.py --stage all`.

## Layout

```
data/
├── raw/spot/BTCUSDT/1m/      zips exactly as downloaded + manifest.csv (hash, size, time)
├── raw/um/BTCUSDT/fundingRate/   optional
├── interim/                  combined + typed, NOT cleaned
└── processed/                audited regular 1-minute grid used for research
```

## Attribution

Market data © Binance, provided through Binance Vision under CC BY-NC-SA 4.0
(Binance Vision Dataset Terms v1.0, updated 26 Aug 2026). Used here for non-commercial academic research only.
