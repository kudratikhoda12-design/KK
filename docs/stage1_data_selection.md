# Stage 1 — Dataset Selection

Project: **Ethereum Order-Flow Alpha: Short-Horizon Return Prediction and Cost-Aware Trading Strategy**

Everything below was checked against the live archive on 2026-10-07 by downloading
sample files and running HTTP HEAD requests for each day. Nothing here is a model result.

## 1. Candidates considered

| Source | Trades w/ aggressor side | Historical order book | Accessible here | Verdict |
|---|---|---|---|---|
| Coinbase Exchange ETH-USD (REST `/products/ETH-USD/trades`) | Yes (`side`) | No historical L2 (live websocket only) | **No** (blocked by the sandbox's egress policy, HTTP 403) | Rejected: pulling years of data 1,000 trades per request is not practical, and it has no historical book |
| Tardis.dev (Coinbase/Binance L2 + trades) | Yes | Yes (true L2 snapshots) | **No** (403). It's also a paid service; only the first day of each month is free | Rejected: not reproducible for free |
| Binance Spot ETHUSDT (data.binance.vision) | Yes (`is_buyer_maker`) | **No** order-book files for spot | Yes | Secondary: robustness only |
| **Binance USDⓈ-M Futures ETHUSDT perpetual (data.binance.vision)** | **Yes** (`is_buyer_maker`) | **Yes, but coarse:** `bookDepth` (see §3) | **Yes** | **Recommended** |

## 2. Recommended dataset

- **Exchange / venue:** Binance USDⓈ-M Futures
- **Instrument:** ETHUSDT perpetual swap (linear, USDT-margined)
- **Source:** Binance Public Data archive, https://data.binance.vision
  (documentation: https://github.com/binance/binance-public-data)
- **Proposed period:** **2023-03-01 → 2026-09-30 (43 months)**, all timestamps in UTC
- **Primary research frequency:** 1-minute bars built from trades (about 1.88M bars); 5-minute bars as a robustness check
- **Access:** free, no API key. Plain HTTPS downloads of daily and monthly ZIP files. Each file has a `.CHECKSUM` (SHA-256) sidecar.

### Files used

| Dataset | URL pattern | Fields (verified) | Cadence | Size |
|---|---|---|---|---|
| `aggTrades` | `/data/futures/um/monthly/aggTrades/ETHUSDT/ETHUSDT-aggTrades-YYYY-MM.zip` | `agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time (ms), is_buyer_maker` | tick (~1M agg trades/day) | 350–730 MB zipped per month, about 20 GB in total |
| `bookDepth` | `/data/futures/um/daily/bookDepth/ETHUSDT/ETHUSDT-bookDepth-YYYY-MM-DD.zip` | `timestamp, percentage (±1..±5), depth (ETH), notional (USDT)` | snapshot about every 30 s (2,881 per day) | ~0.5 MB per day, ~0.6 GB in total |
| `fundingRate` | `/data/futures/um/monthly/fundingRate/ETHUSDT/...` | funding time, rate | 8 h | negligible |
| `metrics` (optional) | `/data/futures/um/daily/metrics/ETHUSDT/...` | open interest, long/short ratios, taker buy/sell volume ratio | 5 min | negligible |
| `bookTicker` (validation only) | `/data/futures/um/daily/bookTicker/ETHUSDT/...` | best bid/ask price and qty, tick-by-tick | tick | **3.4 GB CSV per day**; the archive **ends 2024-03-30** |

Replication asset (same schema): **SOLUSDT** perpetual. I confirmed that `bookDepth` exists for it.

## 3. What the data do and do not contain (honest limitations)

1. **Trade direction is explicit, not inferred.** `is_buyer_maker = true` means the
   aggressor was a seller. Buy/sell volume therefore needs no Lee–Ready or tick-rule
   classification. One caveat: `aggTrades` merges fills from the same taker order at the same price,
   so "number of trades" counts aggregated trades. Large-trade statistics should be read
   with that in mind.
2. **The order book is not a level-by-level L2 book.** `bookDepth` gives *cumulative* resting
   quantity within ±1%, ±2%, …, ±5% of the price, snapshotted about every 30 seconds. So:
   - OBI can be computed at **depth bands of 1%, 2%, 3%, 4% and 5%**, but **not** at "top-1 / top-5 / top-10 levels".
     ±1% of ETH is about 300 ticks, far deeper than the touch.
   - **Best bid, best ask, mid and spread are *not* in `bookDepth`.** The only historical
     top-of-book file (`bookTicker`) stopped on 2024-03-30, and each day is about 3.4 GB.
   - Plan: (a) use `bookDepth` band OBI as the order-book feature set; (b) measure the
     actual quoted spread from `bookTicker` on a sample of days (Q1 2024) so that the cost model
     uses an *empirical* spread rather than an assumption; (c) report this mismatch in the
     README rather than labelling band depth as "top-N levels".
   - Snapshots are every 30 s, and a 1-minute bar's book feature will use **the last snapshot
     at or before the bar close**. That is an as-of backward join, so no forward fill crosses the bar boundary.
3. **Depth coverage gaps** (from HEAD checks of all 1,374 days from 2023-01-01 to 2026-10-05):
   - Jan–Mar 2023 has missing or truncated days. For example, 2023-01-15 has only 7 snapshots. The first unbroken
     30-day run starts on 2023-02-10, **which is why the study starts on 2023-03-01**.
   - After that, only one short day was found (in 2024-06). It will be flagged, not imputed.
   - 2025-04/05 files are smaller on disk, but they hold complete days (2,881 snapshots). They just compress better.
4. **Timestamp unit changes.** Spot archives switch from milliseconds to **microseconds on
   2025-01-01**. The futures files I checked are in ms. The parser will detect the unit per file and
   check it.
5. **This is a perpetual, not spot.** Returns include basis moves, and holding positions incurs
   **funding** every 8 h. Funding is downloaded and charged in the backtest. Shorting is
   native, which suits a long/short signal. Binance futures is the highest-volume ETH venue, so it is
   a defensible venue for price discovery. It is still *one* venue, though, and there is no cross-venue consolidated flow.
6. **Licensing and access.** The data are published by Binance for free public download and come under
   Binance's terms of use. Raw files **will not be committed** to the repo. The repo will ship the
   download script, a manifest of files and SHA-256 checksums, so anyone can rebuild the data.
7. **Survivorship and selection.** ETH was chosen in advance, not because of results.
   The period was chosen only because of data coverage, before looking at any returns.

## 4. Practical size plan

- About 20 GB of zipped futures trades against about 26 GB of free disk in this container. The pipeline will
  stream **one month at a time**: download → verify SHA-256 → parse → aggregate to 1-minute bars →
  write Parquet → (optionally) delete the raw zip. No rows are dropped. Only the storage of the
  raw zip is temporary, and the manifest records every file processed.
- Processed 1-minute bars are about 1.9M rows. That fits in memory easily.

## 5. Pre-registered split (fixed now, before any returns are examined)

| Segment | Period | Use |
|---|---|---|
| Development (walk-forward) | 2023-03-01 → 2025-09-30 (31 months) | EDA, features, expanding walk-forward CV with purge/embargo, model and threshold selection |
| **Final holdout** | **2025-10-01 → 2026-09-30 (12 months)** | Touched once, after the methodology is frozen |

Note: EDA and the statistical alpha tests (Stage 4) will also be run on the development period only.
