# Ethereum Order-Flow Alpha
### Short-horizon return prediction and a cost-aware trading strategy (Binance ETHUSDT perpetual)

**Research question.** Does order-flow and market-microstructure information carry statistically
significant information about short-horizon ETH returns? If so, does it survive as trading alpha after
transaction costs, slippage and funding?

> Status: **Stage 2 of 10 complete** (data acquisition and data quality). No features, models or
> backtests have been run yet, so this README makes no performance claims.

| Stage | Content | Status |
|---|---|---|
| 1 | Dataset selection ([`docs/stage1_data_selection.md`](docs/stage1_data_selection.md)) | done |
| 2 | Acquisition + data-quality pipeline ([`docs/stage2_data_quality_report.md`](docs/stage2_data_quality_report.md)) | done |
| 3–10 | Features, EDA/alpha tests, baselines, ML, walk-forward, backtest, robustness, report | pending |

## Data

- **Source:** Binance public archive, https://data.binance.vision, USD-M futures **ETHUSDT perpetual**.
- **Period:** 2023-03-01 → 2026-09-30 UTC. The development set runs to 2025-09-30. The final holdout is 2025-10-01 → 2026-09-30 and is not used until the methodology is frozen.
- **Trades:** `aggTrades`, with the taker side given explicitly (`is_buyer_maker`).
- **Order book:** `bookDepth`, which gives *cumulative resting quantity/notional within ±1…5 % of the
  price*, about every 30 s. **This is not a level-by-level (top-N) L2 book, and it provides no best bid/ask or spread.**
- **Funding rates** and **open interest** (5-min `metrics`).

The schema and cleaning policy are in [`docs/data_schema.md`](docs/data_schema.md).

The full data are **not** in this repository. They are rebuilt from the public archive, and every file is
SHA-256-verified against Binance's published checksums.

## Reproduce

```bash
pip install -r requirements.txt
pytest -q                                    # unit tests, using the 1-hour sample in data_sample/
python scripts/run_stage2.py                 # full download + QC + 1-min bars (~25 GB download,
                                             # streamed day by day; ~1 GB kept on disk)
python scripts/run_stage2.py --start 2024-06-11 --end 2024-06-13 --data-dir /tmp/ethof_demo   # small demo
```

Outputs go to `data/` (or `$ETHOF_DATA_DIR`): `raw/`, `processed/bars_1m/*.parquet`,
`processed/book_snapshots/`, `processed/funding/`, `processed/metrics/`, `reports/`, `logs/`, and
`manifest.csv`. The manifest records every file's URL, size, expected and actual SHA-256, and status.
Re-runs are idempotent: verified files and processed days are reused.

Settings are in [`config/config.yaml`](config/config.yaml).

## Repository layout

```
config/config.yaml        all parameters (period, split, QC thresholds, size buckets)
src/ethof/data.py         archive URLs, download, SHA-256 verification, manifest
src/ethof/preprocessing.py parsing, ms/µs detection, cleaning, QC, 1-minute aggregation, as-of book join
src/ethof/pipeline.py     Stage 2 orchestration (parallel per-day processing)
src/ethof/quality.py      data-quality / coverage report
scripts/run_stage2.py     CLI entry point
tests/                    unit tests
data_sample/              1-hour subset of official files for tests
docs/                     stage reports, schema
```
