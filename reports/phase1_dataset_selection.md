# Phase 1: Dataset Selection

**Project:** Short-horizon signal research in 24/7 crypto markets, with a bridge to prediction markets
**Author:** Kudrati Khoda (M.Tech QR&OR, ISI Kolkata)
**Date of research:** 5 October 2026
**Status:** Recommendation made. Waiting for confirmation before Phase 2. No data downloaded and no models built.

---

## 0. Recommendation in one paragraph

Use the **Binance Vision public data archive** (`data.binance.vision`) as the primary dataset, starting with **BTCUSDT spot 1-minute klines**. Use ETH, SOL and XRP for cross-asset robustness, and USD-M perpetual klines plus funding rates for execution-cost modelling. The main prediction target will be the event **"the Binance BTC/USDT 1-hour candle closes ≥ its open."** Polymarket's hourly *Bitcoin Up or Down* markets settle on exactly this event, so the project is directly about the underlying of a real prediction-market contract.

Polymarket's own data is the most relevant to Gravia, but I do **not** recommend it as your primary source today, for four reasons:
1. India's MeitY ordered Polymarket blocked at the ISP level on 21 May 2026.
2. There is no official historical order book.
3. Fine-grained price history is kept only for about 7–90 days.
4. The full on-chain route currently depends on an unverified third-party dump of about 107 GB.

Polymarket stays in the plan as a **gated extension** (section 6).

---

## 1. What the dataset must support

These criteria come from the research question, not from convenience:

| # | Requirement | Why it matters |
|---|---|---|
| R1 | Exact timestamps for every observation | Without them we cannot prove a feature used only information available at time *t* (look-ahead control). |
| R2 | A price that approximates what we could actually trade at | A backtest on prices nobody could trade at is fiction. |
| R3 | Information to estimate costs (fee schedule, spread or trade data) | The question is explicitly about value **after** costs. |
| R4 | Enough independent observations | We need statistical power to tell a 1–2 percentage-point edge from noise. |
| R5 | Several market regimes | Robustness tests need bull, bear, crash and calm periods. |
| R6 | Bulk, scriptable, verifiable download | You must be able to reproduce every number. |
| R7 | Legal and technical access **from India** | You must be able to reproduce it from your own network without workarounds. |
| R8 | Raw, not pre-cleaned | The data-health audit should find real problems, not someone else's cleaned output. |
| R9 | Relevance to Gravia | 24/7, adversarial, crypto or prediction-market structure. |

---

## 2. How the facts below were checked

Legend: ✅ checked against the primary source · 🔶 from a secondary source or a search snippet of the primary docs (the docs site was not directly reachable) · ❓ not verified; will be checked in Phase 2.

My research container's network policy blocks every data host I tried: `data.binance.vision`, `api.binance.com`, `*.polymarket.com`, `api.elections.kalshi.com`, `huggingface.co`, Yahoo, Stooq, Goldsky and Polygon RPC. So **I could not download a single sample row in Phase 1.** Every data claim below comes from documentation, not from inspecting data. The data-health audit in Phase 2 is where these claims get tested.

---

## 3. Candidate datasets

### Candidate A: Binance Vision public data archive (recommended)

| Field | Detail |
|---|---|
| Source | Binance, official archive at <https://data.binance.vision/>; docs at <https://github.com/binance/binance-public-data> ✅ |
| Coverage | BTCUSDT spot from 2017-08-17 (well known; ❓ will confirm by listing the bucket). USD-M BTCUSDT perpetual from ~Sep 2019 ❓. SOL and other later listings start later, so coverage differs by asset. |
| Frequency | Klines at `1s, 1m, 3m, 5m, 15m, 30m, 1h, … 1mo` ✅; tick-level `trades` and `aggTrades` ✅ |
| Kline columns | Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base volume, Taker buy quote volume, Ignore ✅ |
| Size (arithmetic) | BTCUSDT 1-minute from 2017-08-17 to 2026-09-30 ≈ **4.80 million rows**, or ≈ **79,944 hourly candles**. Four assets ≈ 15–18M one-minute rows. |
| OHLCV | Yes ✅ |
| Volume | Yes, including **taker-buy volume**, which gives an order-flow-imbalance proxy ✅ |
| Order book | No historical L2 for spot. USD-M futures daily archives reportedly include `bookTicker` and `bookDepth` 🔶, with patchy coverage ❓ |
| Timestamps | Yes. ⚠️ Spot timestamps change from **milliseconds to microseconds from 2025-01-01** ✅ (official README). |
| Transaction data | Yes: `trades` (id, price, qty, quoteQty, time, isBuyerMaker) and `aggTrades` ✅ |
| Free | Yes ✅ |
| Download method | Plain HTTPS zip files by day or month, each with a `.CHECKSUM` (SHA-256) file ✅ |
| Rate limits | None documented for the static archive. The terms (§7) forbid scraping that "impairs platform hosting", so download politely ✅ |
| Licence | **Binance Vision Dataset Terms v1.0, last updated 26 Aug 2026: CC BY-NC-SA 4.0** ✅. Allowed: "academic research … algorithmic historical backtesting for purely personal non-production research". Forbidden: live proprietary trading, commercial products, and distributing signals for compensation. Derived work must keep CC BY-NC-SA. The repository *code* is MIT. |
| Access from India | Binance is FIU-IND registered and operating legally in India since 2024 🔶. The archive domain itself ❓; the probe script in §8 will check it. |
| Reproducibility | **High**: official source, fixed file names, checksums, and an archive-revision log. |
| Relevance to Gravia | **Medium-high**: 24/7, adversarial, crypto. It is also the settlement source of Polymarket's hourly BTC Up/Down markets 🔶, which is the bridge to prediction markets. |
| Known data-quality risks | See §5. They include the µs timestamp switch, exchange-downtime gaps, archive revisions, USDT (not USD) pricing, missing order books, and the volume regime break during Binance's zero-fee BTC promotion in 2022–23 ❓ (to look for in EDA). |

### Candidate B: Polymarket (official APIs and on-chain trade tape)

| Field | Detail |
|---|---|
| Source | Gamma API (market metadata), CLOB API (live order book, price history), Data API v2 (trades, activity), and Polygon on-chain `OrderFilled` events from the CTF Exchange 🔶 |
| Coverage | Polymarket launched in 2020; the CLOB era is later ❓. Data API v2 trade queries scoped to a market go back **~3 years at most** 🔶. Hourly BTC Up/Down markets: launch date ❓. Slugs found online suggest at least 2025. |
| Frequency | Trade-level fills. Price history: **1-minute kept ≥7 days, 5-minute ≥60 days, 30-minute ≥90 days; only 12-hour granularity is permanent** (3-hour from July 2026) 🔶. In the older CLOB endpoint, resolved markets return nothing below 12 hours ([py-clob-client issue #216](https://github.com/Polymarket/py-clob-client/issues/216)). |
| Columns | Trades: wallet, side, asset/token id, condition id, size, price, timestamp, transaction hash 🔶 |
| Size | Third-party full dump: 293M fills and 268,706 markets (claimed, ❓) |
| OHLCV | Not natively; has to be built from fills |
| Volume | Yes, from fills. Hourly BTC windows have a **median of ~$17.8k volume per window** 🔶 (updowncharts.com), so capacity is very limited. |
| Order book | **Live only** through the official API. Historical books exist only from third parties (pmxt archive, hourly snapshots, CC BY 4.0 🔶; PendulumFlow 🔶), and their quality is unverified. |
| Timestamps | Yes (seconds or block time) |
| Transaction data | **Yes, with wallet addresses.** This is what makes Polymarket special and maps to Gravia's "wallet analysis". |
| Free | Yes for reading |
| Rate limits | Gamma 4,000 req/10 s overall; Data API v2 800 req/10 s; CLOB 9,000 req/10 s; CLOB `/prices-history` 1,000 req/10 s; throttled, not rejected 🔶 |
| Fees (for backtests) | Crypto markets: taker fee = shares × 0.07 × p × (1−p); makers pay 0 and receive a 20% rebate 🔶 (Polymarket fee docs via search). The fee regime changed during 2025–26, which creates a structural break. |
| Licence / access | Polymarket Terms of Use: trading is restricted in many jurisdictions; data is viewable. ⚠️ **India: MeitY ordered an ISP-level block on 21 May 2026** under the Promotion and Regulation of Online Gaming Act 2025, and an earlier advisory targeted VPN use for blocked platforms 🔶. **Do not use a VPN to reach Polymarket.** I am not a lawyer; if unsure, ask your institute. |
| Reproducibility for you | **Low to uncertain** from India through the official APIs. The on-chain route (Polygon RPC, Google BigQuery `goog_blockchain_polygon_mainnet_us` 🔶, or the Hugging Face dump `SII-WANGZJ/Polymarket_data` (MIT claimed, "data before 2026", ~107 GB) 🔶) does not touch Polymarket's servers, but it is heavy and unverified. |
| Relevance to Gravia | **Highest**: this is their market. |
| Known data-quality risks | Bounded prices with jumps to 0 or 1 at resolution; last trade ≠ executable price; thin liquidity; wash trading; API v1→v2 and CLOB V2 migrations; settlement-rule changes (hourly = Binance candle; 5m/15m/4h = Chainlink TWAP since ~Aug 2026 🔶); Eastern-Time titles with daylight saving mapped to UTC candles; tie rule "close ≥ open counts as Up"; selection bias in which markets you study; third-party cleaning steps ("unified YES perspective") that would need auditing. |

### Candidate C: Kalshi public API

| Field | Detail |
|---|---|
| Source | Kalshi REST API (CFTC-regulated US exchange) |
| Coverage | Earliest trade reportedly 30 June 2021 🔶; there is a `GET /historical/cutoff` endpoint plus `/historical/trades` and `/historical/markets/{ticker}/candlesticks` 🔶 |
| Frequency | Candlesticks at 1 minute, 1 hour or 1 day 🔶; trade-level prints |
| Columns | Candlesticks reportedly include **yes_bid and yes_ask OHLC** besides trade-price OHLC, volume and open interest 🔶. That would give historical *quotes*, which is a real advantage for cost modelling. |
| OHLCV / volume | Yes 🔶 |
| Order book | Live only; quotes appear inside candlesticks 🔶 |
| Transaction data | Trades yes; **no wallet or trader identity** (not on-chain) |
| Free / auth | Market-data endpoints need no authentication 🔶 |
| Rate limits | Basic tier: 200 read tokens/s, most requests cost 10 tokens 🔶 |
| Licence / access | Kalshi API terms ❓. ⚠️ India: reports say MeitY is "in the process" of blocking Kalshi too 🔶. |
| Reproducibility | Medium. ⚠️ Kalshi BTC contracts settle on the **CF Benchmarks BRTI 60-second average** 🔶. That index's history is a licensed product, so you **cannot independently recompute the label**. |
| Relevance to Gravia | High (Gravia names Kalshi), but not on-chain |
| Data-quality risks | Same binary-contract issues as Polymarket; a historical/live data split at the cutoff; contract-design changes over time. |

### Candidate D: US equities or ETFs (Yahoo Finance through `yfinance`, or Stooq)

| Field | Detail |
|---|---|
| Coverage / frequency | Decades of daily data; intraday history only for a short recent window |
| OHLCV | Yes, daily |
| Order book / transactions | No |
| Licence | Yahoo's API is unofficial and its terms are personal-use oriented; it can break without notice ❓ |
| Risks | Survivorship bias, revisions of adjusted prices, corporate actions, market-closure gaps |
| Verdict | Clean for a textbook momentum study, but **least relevant to Gravia**, not 24/7, and not reproducible through any official API. Rejected. |

### Also considered and rejected as primary

* **Kaggle "Polymarket 5-minute BTC Up/Down"** and similar dumps: pre-cleaned, unclear provenance. You asked to avoid these.
* **Tardis.dev, Kaiko, CryptoLake** (historical crypto order books): the best microstructure data, but paid (Tardis has free first-of-month samples ❓). Not reproducible for free.
* **Coinbase Exchange API candles**: free, but paged at 300 candles per request ❓, has no bulk archive, and is not the settlement source of any Polymarket market.

---

## 4. Scorecard

Scores are 0–2 (2 = good), my judgement from sections 1–3.

| Requirement | A. Binance Vision | B. Polymarket | C. Kalshi | D. Yahoo equities |
|---|---|---|---|---|
| R1 timestamps | 2 | 2 | 2 | 1 |
| R2 tradeable price | 2 (trades and klines on the venue itself) | 1 (fills yes, no historical book) | 2 (bid/ask in candles 🔶) | 1 |
| R3 cost information | 2 (published fees, trades) | 1 (fees known, spread unknown) | 2 🔶 | 1 |
| R4 sample size | 2 (~80k hourly obs.) | 1 (thousands of markets; ~$18k median volume) | 1 | 1 |
| R5 regimes | 2 (2017–2026) | 0–1 (hourly markets are recent) | 1 | 2 |
| R6 bulk, verifiable download | 2 (checksums) | 1 | 1 | 0 |
| R7 access from India | 2 🔶 | **0** (blocked) | **0–1** (block reportedly coming) | 1 |
| R8 raw, not pre-cleaned | 2 | 2 (on-chain) / 1 (third-party dump) | 2 | 1 |
| R9 Gravia relevance | 1.5 (settlement source of Polymarket hourly) | 2 | 2 | 0 |
| **Total (of 18)** | **17.5** | **9–10** | **13–14** | **8** |

---

## 5. Known problems in the recommended dataset (to audit in Phase 2, not fix silently)

1. **Timestamp unit change:** spot files use microseconds from 2025-01-01 ✅. Treating them as milliseconds silently produces dates thousands of years in the future, or wrong joins.
2. **Missing minutes or hours** from exchange maintenance or outages: we detect and report them, never forward-fill silently.
3. **Archive revisions:** Binance replaced some archived kline and aggTrade files (2022-04-21 and 2022-08-08 changelogs) ✅. We record the SHA-256 and download date of every file.
4. **Overlapping daily and monthly files** can create duplicate rows.
5. **Prices are in USDT, not USD.** USDT can deviate from $1 under stress. This does not change the target, because Polymarket hourly markets settle on BTC/USDT.
6. **Volume regime breaks:** fee promotions (e.g. Binance's zero-fee BTC spot trading in 2022–23 ❓) can inflate volume, which would distort volume features.
7. **No historical spot order book:** the spread and slippage are *assumptions*, stress-tested in Phase 7. Where USD-M `bookTicker` files exist ❓, we measure the real spread for a sample.
8. **Survivorship in multi-asset tests:** pick the assets in advance (BTC, ETH, SOL, XRP) and never choose them by performance.
9. **Stale candles:** minutes with zero trades repeat the last price; count and flag them.
10. **Close-time convention:** a kline's close time is open time + interval − 1 ms. Using the close of the current hour as a feature for that same hour's target would be leakage.

---

## 6. Why not Polymarket as primary, and how it stays in the project

Polymarket is where Gravia's edge lives, and Gravia's JD asks for alpha in on-chain fills and wallets. I took it seriously. As your *primary* source today, though, it fails three of the hard requirements:

* **R7, access:** the official APIs are blocked in India by government order. A project you can only reproduce through a VPN is not reproducible, and is a legal risk you should not take.
* **R2/R3, executable prices:** there is no official historical order book, and fine-grained price history for resolved markets is gone after ~7–90 days. Costs would rest on assumptions you cannot check.
* **R4/R5, sample:** hourly markets are recent and thin (~$18k median volume per window 🔶), so there are few regimes and little capacity.

**How Polymarket still enters the project:**

1. **The target is the Polymarket settlement event.** Y = 1 if the Binance BTC/USDT 1-hour candle closes ≥ its open, which is the hourly *Bitcoin Up or Down* resolution rule 🔶.
2. **Two ways to monetise the same signal:** (a) trade the Binance perpetual with its published fees, and (b) buy a binary contract under Polymarket's published crypto fee curve. This makes the economics concrete (see §7).
3. **Gated extension (optional, Phase 7b):** if you can *legally* obtain on-chain fills without touching Polymarket's servers (Polygon RPC, BigQuery's public Polygon dataset, or the Hugging Face dump), we test whether actual Polymarket hourly prices were efficient relative to our Binance-based model. If not, the report states this as a limitation and as the next experiment.

**What would change my recommendation:** you confirm you can reach the on-chain data legally, and you prefer higher Gravia relevance over cleaner reproducibility. Then I would switch the primary to a *Polymarket hourly-market panel joined to Binance klines*, with a go/no-go data check before any modelling.

---

## 7. Arithmetic that shapes the research design (not results)

These are calculations from published fee formulas and sample sizes. **They are not findings about the data.**

**(a) Break-even accuracy for a Polymarket-style binary bet.** If you buy "Up" at price *p* per share, it pays 1 if Up and 0 otherwise. Taker fee per share = 0.07·p·(1−p) 🔶. The expected profit per share is q − p − 0.07·p(1−p), so the break-even win probability is **q\* = p + 0.07·p·(1−p)**.

| Entry price p | Fee per share | Fee as % of stake | Break-even win rate q\* |
|---|---|---|---|
| 0.50 | 0.0175 | 3.50% | **51.75%** |
| 0.51 (paying 1¢ of spread) | 0.0175 | 3.43% | **52.75%** |
| 0.55 | 0.0173 | 3.15% | 56.73% |
| 0.60 | 0.0168 | 2.80% | 61.68% |

So at the start of the hour, when the fair price is about 0.50, a model needs out-of-sample accuracy of roughly **52–53%** just to break even.

**(b) Perpetual futures.** Binance USD-M regular-tier taker fee is 0.05% per side 🔶, so a round trip costs about **0.10% of notional** before slippage and funding. Whether a 1-hour signal can beat that depends on the hourly return distribution, which we measure in Phase 3 rather than assume.

**(c) Can we detect such an edge?** The standard error of a hit rate near 50% is √(0.25/n):

| Sample | n (hours) | SE of hit rate |
|---|---|---|
| Full BTC history (2017-08-17 to 2026-09-30) | 79,944 | 0.18 pp |
| A 2-year untouched test set | ~17,500 | 0.38 pp |
| 1 year | ~8,770 | 0.53 pp |

A true 52.75% hit rate would be roughly 7 standard errors from 50% even on a 2-year test set. **If an edge that size exists, we should see it. If we don't, that is informative.** Caveat: serial dependence and regime changes lower the effective sample size, so later phases use block bootstrap rather than these i.i.d. formulas.

---

## 8. What happens next (after your confirmation)

1. **Fix data access.**
   * Option 1: allow `data.binance.vision` in this cloud environment's network settings (Custom access, keeping the default package-manager list).
   * Option 2: run the acquisition notebook on your own machine.
   * Raw data will **not** be committed to git, both for size and because the licence is CC BY-NC-SA.
   * Run `python src/check_data_access.py` on your network first; it reports which sources you can actually reach.
2. **Task 2:** define the exact research question (Y, horizon, decision time, entry, exit, sizing, no-trade region) for the confirmed dataset.
3. **Phase 2:** download with checksum verification, then the data-health audit.

### Decisions I need from you

* **D1.** Confirm the primary dataset: Binance Vision BTCUSDT, with ETH, SOL and XRP for robustness. (Alternative: the Polymarket-panel design from §6.)
* **D2.** For Phase 2, either enable `data.binance.vision` in the environment or run the download locally.
* **D3.** Keep the Polymarket on-chain extension as an optional Phase 7b, yes or no?

---

## Sources

Primary sources (✅ read directly):
* Binance public data README and Vision Dataset Terms v1.0 (26 Aug 2026): <https://github.com/binance/binance-public-data>, <https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md>

Polymarket docs (🔶 via search snippets; `docs.polymarket.com` was blocked from my environment):
* Fees: <https://docs.polymarket.com/trading/fees>
* Rate limits: <https://docs.polymarket.com/api-reference/rate-limits>
* Data API v2 and price-history retention: <https://docs.polymarket.com/api-reference/data-api/overview>, <https://docs.polymarket.com/api-reference/markets/get-prices-history>
* v1→v2 migration and the ~3-year floor: <https://docs.polymarket.com/migrate/data-api-v1-to-v2>
* Hourly BTC market rules (Binance BTC/USDT 1H candle, close ≥ open = Up): e.g. <https://polymarket.com/event/bitcoin-up-or-down-march-8-12pm-et>

Other sources (🔶):
* Resolved-market price-history limitation: <https://github.com/Polymarket/py-clob-client/issues/216>
* Third-party Polymarket on-chain dump: <https://github.com/SII-WANGZJ/Polymarket_data>
* Historical order-book archives: <https://archive.pmxt.dev/>, <https://archive.pendulumflow.com/>
* Settlement sources for 5m/15m/4h vs hourly markets: <https://genfinity.io/2026/08/12/chainlink-twap-data-streams-polymarket-crypto-markets/>, <https://x.com/chainlink/status/1966502945173717409>
* India block: <https://crypto.news/polymarket-goes-offline-in-india-after-government-enforcement-order/>, <https://www.yogonet.com/international/news/2026/05/22/121400-india-blocks-access-to-polymarket-amid-prediction-markets-crackdown>
* Kalshi historical endpoints and settlement: <https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks>, <https://predictionmarketspicks.com/articles/how-kalshi-settles-bitcoin>
* Binance in India: <https://www.dlnews.com/articles/regulation/binance-has-successfully-registered-with-the-fiu-in-india/>
* Binance futures fees: <https://www.finder.com/cryptocurrency/trading/binance-futures-fees>
* Hourly-window volume statistics: <https://www.updowncharts.com/markets/bitcoin>
* BigQuery Polygon dataset: <https://console.cloud.google.com/marketplace/product/bigquery-public-data/blockchain-analytics-polygon-mainnet-us>
