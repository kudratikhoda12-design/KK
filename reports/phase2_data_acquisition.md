# Phase 2: Data Acquisition

**Status: BLOCKED in the cloud environment. Code complete and tested. No data has been downloaded and none has been substituted.**

## A. What we did

1. Re-tested access from this cloud environment on 5 Oct 2026. `data.binance.vision`, `data-api.binance.vision` and `api.binance.com` were all refused by the environment's egress proxy (HTTP 403 on CONNECT, recorded as an organisation policy denial). This is a network-policy restriction of the research environment, not a Binance outage.
2. Built the acquisition code so the data can be fetched reproducibly:
   * `src/data.py download`: 110 monthly archives (2017-08 to 2026-09). Each is verified against Binance's SHA-256 `.CHECKSUM`, logged in a manifest (URL, bytes, hash, check result, UTC time), and falls back to daily files if a monthly file is not yet published. Re-running resumes.
   * `src/data.py build`: parses and types without cleaning. Timestamp units are detected per file (milliseconds before 2025, microseconds from 2025). Optional header rows are handled. Original file and row order are recorded for the audit.
   * `scripts/download_data.sh`: a shell-only alternative using `curl` and `sha256sum -c`.
3. Tested the loader on fixture files covering millisecond, microsecond and header variants. It correctly maps `1735689600000000` µs to 2025-01-01 00:00 UTC and refuses epoch values it cannot classify.
4. Ran the **entire pipeline** end to end on a synthetic random walk of the **same size** as the real data (4.8M minutes), writing only to a temporary folder. This proves the code runs and gives a run-time estimate. None of it is reported as a result about BTC.

## B. Why

The rule for this project is no fabricated data. When the data is unreachable, the right move is to make the real run a one-command step and to finish everything that does not depend on the data's values.

## C. Exact specification of the required data

| Item | Value |
|---|---|
| URL pattern | `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip` (+ `.CHECKSUM`) |
| Format | Zip containing one headerless CSV with 12 columns |
| Columns | open_time, open, high, low, close, volume, close_time, quote_volume, n_trades, taker_buy_base, taker_buy_quote, ignore |
| Range | 2017-08 to 2026-09: 110 files, 3,331 days |
| Expected observations | 4,797,840 one-minute candles (2017-08-17 04:00 to 2026-09-30 23:59) if nothing were missing |
| Storage | `data/raw/spot/BTCUSDT/1m/` (zips + manifest) → `data/interim/BTCUSDT_1m_raw.parquet` → `data/processed/BTCUSDT_1m_grid.parquet` |
| Memory | About 0.5 GB for the typed 1-minute table; the full run fits in 16 GB RAM |
| Why the full history | It covers several regimes: the 2017 bubble, the 2018 bear market, the March 2020 crash, the 2021 bull market, the 2022 LUNA/FTX collapses, and 2024–26. Using a shorter window to save compute would weaken the robustness tests. |

## D. What you need to do (one of these)

**Option A (preferred):** allow `data.binance.vision` in this environment's network settings: environment menu → Edit → Network access → Custom, add the host, and keep the default package list. Then ask Claude to continue. The download, audit and every later stage then run here.

**Option B:** run on your machine:

```bash
python src/check_data_access.py
python -m src.data download --symbol BTCUSDT --start 2017-08 --end 2026-09
python -m src.data build --symbol BTCUSDT
python scripts/run_pipeline.py --stage all        # optional: run everything locally
```

Then either upload `data/interim/BTCUSDT_1m_raw.parquet` together with `data/raw/spot/BTCUSDT/1m/manifest.csv` to the session, or share the `reports/` and `figures/` folders produced locally.

## E. Decision

Phases 3–19 are fully implemented, tested and pre-registered, and wait only on the real file. **No result section of this project will be written until it has been computed from the real Binance data.**
