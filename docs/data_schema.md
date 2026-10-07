# Data schema and conventions

## Raw inputs (Binance public archive, USD-M futures, ETHUSDT perpetual)

`https://data.binance.vision/data/futures/um/{daily|monthly}/{dataset}/ETHUSDT/ETHUSDT-{dataset}-{period}.zip`
plus a `.zip.CHECKSUM` sidecar (`<sha256>  <filename>`). Each download is verified before use and logged in `data/manifest.csv`.

| Dataset | Granularity used | Columns | Notes |
|---|---|---|---|
| `aggTrades` | daily (`YYYY-MM-DD`) | `agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker` | `transact_time` in epoch **ms** for futures (the unit is detected per file; spot switched to µs in 2025). `is_buyer_maker=true` ⇒ the **seller** was the taker. One aggTrade = the fills of one taker order at one price. |
| `bookDepth` | daily | `timestamp, percentage, depth, notional` | Snapshot about every 30 s. `percentage ∈ {−5…−1, 1…5}`: **cumulative** quantity (`depth`, ETH) and value (`notional`, USDT) resting within k % below (bids, negative) / above (asks, positive) the reference price. **Not level-by-level L2. No best bid/ask or spread.** Timestamps are `YYYY-MM-DD HH:MM:SS`, which we read as UTC and check in the Stage 2 report. |
| `fundingRate` | monthly | `calc_time, funding_interval_hours, last_funding_rate` | `calc_time` ms. |
| `metrics` | daily | `create_time, symbol, sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio` | 5-minute. Downloaded; whether it is used is decided in Stage 3. |

## Processed outputs (`data/processed/`, git-ignored)

### `bars_1m/ETHUSDT-bars_1m-YYYY-MM.parquet`

One row per UTC minute. The grid is complete: minutes without trades are kept, never dropped. A bar labelled `ts` covers `[ts, ts+1min)`.

| Column(s) | Definition |
|---|---|
| `ts` | bar start, `Datetime[us, UTC]` |
| `open, high, low, close` | first / max / min / last trade price in the bar; **null if no trade** (not forward-filled) |
| `volume`, `notional` | Σ qty (ETH), Σ price·qty (USDT) |
| `vwap` | notional / volume (null if no trade) |
| `n_trades` | number of aggregated trades; `n_fills` = number of underlying fills (Σ last_id − first_id + 1) |
| `buy_volume, sell_volume, buy_notional, sell_notional, buy_trades, sell_trades` | split by **taker side** (buy = `is_buyer_maker == false`) |
| `trade_size_mean, _median, _max, _p90, _p95` | distribution of aggTrade sizes **within** the bar |
| `buy_volume_ge{k}, sell_volume_ge{k}` | taker volume from aggTrades with qty ≥ k ETH, k ∈ {1, 5, 10, 25, 50, 100, 250}. The thresholds are fixed in advance, so Stage 3 can build trailing large-trade features without re-reading the ticks |
| `last_trade_ts` | timestamp of the bar's last trade |
| `n_bad_tick_flags` | trades in the bar more than 2 % from the bar's median price (QC flag only) |
| `has_trades` | `n_trades > 0` |
| `book_ts` | timestamp of the book snapshot attached as-of: last snapshot stamped **strictly before** `ts + 1min` |
| `bid_depth_k, ask_depth_k, bid_notional_k, ask_notional_k` (k = 1…5) | that snapshot's band values |
| `book_age_s` | (`ts` + 60 s) − `last_change_ts` (the last snapshot whose values changed), so a frozen feed ages and becomes stale |
| `book_stale` | no snapshot, or `book_age_s > 120` |
| `book_short_day` | the bar's UTC day has < 90 % of the 2,880 expected snapshots (e.g. 2024-06-12, which has only 2). **Flag only; nothing is interpolated** |
| `book_day_snapshots` | number of snapshots on that day |

### Other outputs

- `book_snapshots/ETHUSDT-book_snapshots.parquet`: cleaned snapshots in wide format (`ts` + the 20 band columns above, `book_frozen` = all 20 values equal the previous snapshot, `last_change_ts`).
- `research/ETHUSDT-research.parquet` (Stage 3): the bars above plus 47 features (`ethof.features.GROUPS`), `px` / `exec_px`, and the targets `fwd_ret_h`, `up_h`, `cls3_h`, `label_end_h` for h ∈ {5, 15, 30, 60}.
- `funding/ETHUSDT-funding.parquet`: `calc_time, funding_interval_hours, last_funding_rate, ts`.
- `metrics/ETHUSDT-metrics.parquet`: the metrics columns + `ts`.

## Cleaning policy

| Category | Action |
|---|---|
| null / non-finite values, price ≤ 0, qty ≤ 0, `first_trade_id > last_trade_id`, timestamp outside the file's UTC day, unparseable side flag | **removed** (counted) |
| exact duplicate rows | one copy kept (counted) |
| same `agg_trade_id` with different content | first kept (counted) |
| open interest ≤ 0, ±inf feature ratios | set to missing (counted) |
| frozen (repeated) book snapshots | kept, but treated as stale: book features are missing |
| trades far from the minute median, extreme 1-min moves, trade-id sequence gaps, empty minutes, missing/short book days, stale snapshots | **flagged, kept** |

Large price moves are never removed just because they are large. The Stage 2 report lists every 1-minute bar with |log return| or log(high/low) > 3 % for review. A move is classed as an error candidate only if it fully reverts in the next bar on thin activity.
