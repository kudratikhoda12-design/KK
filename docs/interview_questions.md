# Interview Preparation: Ethereum Order-Flow Alpha

The answers are based on this project's actual results. The source files are listed in `docs/research_report.md`.

## Data

**Why Binance futures?**
- Binance USDⓈ-M is the highest-volume ETH venue.
- Its public archive has years of trade-level data with the aggressor side given explicitly, plus order-book depth, funding and open interest. All of it is SHA-256 verifiable and free, so anyone can rebuild the dataset.
- Coinbase had no free historical book data, and its API, like Tardis, was unreachable from the research environment.
- A perpetual also allows native shorting, which a long/short signal needs.

**Why ETH?**
- BTC was excluded by design.
- ETH is the second most liquid crypto asset, with ~1M aggregated trades per day.
- SOL was used for replication.

**Why this time period?**
- The order-book files have gaps before March 2023; the first unbroken 30-day run starts 2023-02-10.
- So 2023-03-01 → 2026-09-30, with the last 12 months reserved as the holdout before any returns were examined.

**What are aggTrades?**
- Binance merges the fills of one taker order at one price into one aggregated trade.
- Each record has the price, quantity, the first and last underlying trade ids, the timestamp, and the side flag.
- We read 1,877,097,488 of them. 4.0 M underlying fill ids (0.216 %) are missing from the archive; they are flagged, not repaired.

**What is buyer-maker?**
- `is_buyer_maker = true` means the buyer's order was resting, so the *seller* was the aggressor.
- Buyer-initiated (taker buy) volume is therefore `is_buyer_maker = false`.
- Because the side is given, no Lee–Ready or tick-rule inference is needed.

**What does order-flow imbalance mean?**
- OFI = (taker buy volume − taker sell volume) / total volume over a window. It ranges from −1 to +1.
- It measures net demand for immediacy.
- It is strongly tied to the *same-minute* return: corr = 0.467.

**What exactly is Binance `bookDepth`?**
- Cumulative resting quantity and notional within ±1, 2, 3, 4, 5 % of the price, snapshotted about every 30 s.
- It is **not** a level-by-level book: there is no best bid/ask, spread or queue. ±1 % is about 300 ticks.
- The sign convention and the UTC time zone were verified empirically: 94 % of snapshots have bid-band average < trade VWAP < ask-band average with no clock shift, against 64 % with a 1-hour shift.
- We also found that the archive repeats one snapshot for a month (2025-04-16 → 2025-05-19). We detect this by checking whether values change.

## Statistics

**Why is correlation insufficient?**
- Pearson correlations here are around 0.005–0.009. With 1.36 M overlapping observations, naive t-stats are hugely inflated, because overlapping forward returns are autocorrelated by construction.
- The returns are heavy-tailed: 1-min excess kurtosis is 487, so a few minutes dominate Pearson.
- Correlation says nothing about the magnitude relative to costs.
- We therefore used HAC errors, non-overlapping re-estimation, rank statistics, decile spreads with a day-block bootstrap, and controls.

**Why walk-forward validation?**
- Neighbouring minutes share future price paths and regimes, so random CV leaks the future into training.
- Expanding walk-forward mimics live deployment.
- Purge plus a 1-day embargo stops training labels from overlapping validation labels.

**What is look-ahead bias, and how was it tested?**
Look-ahead bias means using information that was not available at decision time. We tested for it directly:
- Every feature was recomputed on data cut at 12 random times. All 34.8 M values were identical: no feature changes when the future is removed.
- Book snapshots must be stamped strictly before the bar close.
- Funding must be settled before decision time.
- Open interest is lagged at least 5 minutes.

**What is data snooping, and how was it handled?**
Data snooping means finding patterns because many things were tried. We recorded every attempt: 43 model configurations, 5 thresholds and 36 univariate tests.
- Holm and BH adjustments were applied to the tests.
- All choices were frozen before the single holdout evaluation.
- The holdout AUC z-score (19.6) passes a Bonferroni correction over 43 configurations (needs 3.05).
- The gross Sharpe (2.03) does *not* beat the ~3.3 expected maximum of 215 random strategies, so we do not claim gross profitability either.

**Why can AUC > 0.5 still lose money?**
- AUC measures ranking of direction, not magnitude net of costs.
- Our model is right 57 % of the time on the trades it takes, with an average gross gain of 0.70 bps.
- A round trip costs 11 bps.
- AUC 0.544 is statistically certain and economically worthless here.

## Machine learning

**Why logistic regression?**
- It is an interpretable, regularised linear baseline (L2, C = 0.1) and checks whether non-linearity adds value.
- Holdout AUC 0.534 vs 0.544 for the random forest.

**Why gradient boosting (LightGBM)?**
- It is the standard strong learner for tabular, noisy, non-linear data with missing values handled natively.
- A restrained 8-point grid picked the *smallest* model (15 leaves, 200 trees). The larger models had worse validation log loss, a sign of overfitting a weak signal.
- The random forest won the pre-declared log-loss rule by 0.00005, a tie in practice.

**Why not deep learning?**
- The signal-to-noise ratio is tiny: the best IC is about 0.06.
- Tree models already saturate at AUC ≈ 0.54, and the bottleneck is cost, not model capacity.
- A sequence model would add risk of overfitting and leakage without addressing why the strategy fails.

**How did you prevent leakage?**
- Causal features only, built on a complete minute grid.
- Backward-only as-of joins with an extra lag for open interest.
- Purge and embargo between training and validation.
- Preprocessing inside per-fold pipelines.
- Holdout rows are accessible only through a guard that requires the frozen methodology file and logs its hash.
- An automated audit runs all of these checks.

**How did you tune hyper-parameters?**
- 8 configurations, scored on mean validation log loss across 7 walk-forward folds on development data only.
- The holdout was never used for any choice.

## Trading

**What is the bid-ask spread here?**
- The ETHUSDT perp is one tick ($0.01) wide 99.5 % of the time, a mean of 0.046 bps measured on 3 days.
- Spread is negligible; fees dominate.

**What is slippage?**
- The difference between the decision price and the achieved price.
- We execute at the next minute's VWAP and add a further 1 bp at base cost (0.5 bp low, 2 bps high).

**What are the transaction costs?**
- Taker fee 4.5 bps with the BNB discount (5 bps without), plus slippage and half spread: 5.52 bps per side at base, 11 bps per round trip.

**What is funding?**
- An 8-hourly payment between longs and shorts that keeps the perpetual near spot.
- We charge the realised rate to the position held at each settlement.
- It was immaterial for a 15-minute strategy (+0.08 % of capital over the holdout year).

**How does position sizing work?**
- Each signal opens 1/15 of capital for 15 minutes; overlapping positions net out, so |position| ≤ 1 and there is no leverage.
- Average exposure was 0.22.

**Why does a predictive signal not generate alpha?**
- The predictable move is short-horizon mean reversion of about 1–3 bps.
- That is the same order of magnitude as the bounce that liquidity providers capture.
- A taker pays about 11 bps to harvest it. Break-even is 0.49 bps per side.

## Research

**What was your strongest finding?**
- Out-of-sample predictability is real and stable: AUC above 0.5 in every one of 31 out-of-sample months, a holdout CI of [0.539, 0.548], and calibrated probabilities.
- It is almost entirely mean reversion in *price*, with order flow pointing the same way (reversal).

**What failed?**
- The economic hypothesis failed: no model, feature set, horizon, threshold or cost scenario is net-profitable.
- Order-flow and book features did not add robust incremental value over price.
- The continuation rules (momentum, buy-on-OFI) lose money even before costs.

**What surprised you?**
1. The order-book archive is frozen for a month while looking complete.
2. Order flow predicts reversal, not continuation.
3. The holdout AUC was slightly higher than in development.

**What would you do differently?**
- Get L2/L3 data to model passive (maker) execution and touch-level imbalance.
- Use second-level horizons.
- Pre-register an even smaller experiment set, and measure SOL's spread for its cost model.

**How would you deploy this?**
Not as a stand-alone taker strategy. Possible uses:
- an execution-timing overlay, e.g. delaying a planned buy after heavy taker buying;
- an input to a market-making skew, if queue-position data confirm a maker edge.
Before that, a paper-trading phase with real fills.

**Why should we trust the backtest?**
- Raw data are SHA-256-verified and the pipeline is fully reproducible.
- An automated leakage audit runs, and execution happens one bar after the signal at VWAP.
- Costs are sourced or measured, and funding is included.
- The methodology was frozen and hash-logged before a single holdout evaluation.
- Random, momentum and buy-and-hold baselines were evaluated in the same engine.
- The backtest reports a *loss*. The main way it could be wrong is by being optimistic (VWAP fills, 1-bp slippage), which would make the conclusion stronger.
