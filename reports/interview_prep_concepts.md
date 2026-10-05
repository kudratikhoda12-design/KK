# Interview Preparation, Part 1: Concepts (independent of results)

Simple English. For each question: **Simple answer** (say this first) → **Deeper answer** (if they ask more) → **Likely follow-up** → **Do NOT say**.
Project-specific and adversarial questions ("why should I believe your backtest?") are in Part 2, written only after the real-data results exist.

---

## A. Data and data quality

**1. How did you check data quality?**
* **Simple:** "Before any modelling, I ran an audit for schema, timestamps, missing minutes, impossible candles, volume, duplicates and extreme moves. Every fix followed a rule I wrote down in advance."
* **Deeper:** "For example: high must be ≥ open and close; the VWAP, quote volume divided by volume, must lie inside the candle's range; each timestamp must be on a minute boundary; and every file is checked against Binance's SHA-256 checksum."
* **Follow-up:** "What did you do with a bad candle?" → "I set that minute to missing. I did not fill it or guess."
* **Do NOT say:** "The data was clean, so I just used it."

**2. How did you handle missing candles?**
* **Simple:** "I left them missing. A feature is only computed if at least 90% of its window exists; otherwise that hour is dropped."
* **Deeper:** "Forward-filling creates fake zero returns. That lowers measured volatility and can create fake patterns around outages."
* **Follow-up:** "Doesn't dropping rows bias you?" → "Only if the missingness relates to returns. Gaps are rare, and I report how many rows were dropped."
* **Do NOT say:** "I forward-filled the gaps."

**3. Why is timestamp handling important?**
* **Simple:** "Being off by one minute can let the model see the future. And Binance changed its timestamp unit from milliseconds to microseconds in 2025."
* **Deeper:** "If you read microseconds as milliseconds, dates land thousands of years in the future and joins break silently. I detect the unit per file and test it."
* **Do NOT say:** "pandas handles dates automatically."

**4. How did you handle outliers?**
* **Simple:** "I flagged them and kept them, unless they were impossible."
* **Deeper:** "A crash is real risk. Deleting it would make the strategy look safer than it is. I only remove candles that break basic rules, like high below low."
* **Do NOT say:** "I removed everything beyond 3 standard deviations." Fat tails mean many real moves are beyond 3 SD.

**5. How did you detect leakage?**
* **Simple:** "Two ways. A timing table showing exactly which candle each feature uses. And a test that replaces all future data with random numbers and checks that the features don't change."
* **Deeper:** "I ran that perturbation test in unit tests and on the real data at random times. I also scale features inside the training window, and I purge training labels that overlap the test block."
* **Do NOT say:** "I used shift(1), so there is no leakage." Shifting is necessary but not proof.

## B. Statistics

**6. What is autocorrelation?**
* **Simple:** "The correlation of a series with its own past values. For returns, it asks whether the last hour's return tells you about the next one."
* **Deeper:** "I test it with the Ljung–Box test across many lags. Returns usually show tiny autocorrelation, but *absolute* returns show strong autocorrelation."
* **Do NOT say:** "No autocorrelation means returns are independent." Squared returns can still be dependent.

**7. What is volatility clustering?**
* **Simple:** "Big moves tend to follow big moves, and calm follows calm."
* **Deeper:** "That is why |return| has strong autocorrelation even when returns have almost none. It is why I use block bootstrap and HAC standard errors instead of i.i.d. formulas."

**8. Why are financial returns hard to predict?**
* **Simple:** "If a pattern were easy and profitable, traders would use it until it disappeared. What remains is mostly noise."
* **Deeper:** "The signal-to-noise ratio is tiny, relationships change over time, tails are fat, and costs eat small edges. A 52% hit rate can still lose money after fees."

**9. Why can accuracy be misleading?**
* **Simple:** "Being right often doesn't mean making money. You can win many small trades and lose a few big ones."
* **Deeper:** "If the market goes up 52% of hours, always predicting up gives 52% accuracy with zero skill. That's why I compare against the base rate, use AUC and log-loss, and judge by net P&L."

**10. What is the multiple-testing problem?**
* **Simple:** "If you test 20 ideas at 5% significance, about one will look significant by luck."
* **Deeper:** "I correct with Holm across all 16 features, and the Deflated Sharpe Ratio adjusts for how many strategies I tried. On synthetic random-walk data, 2 of 16 features looked significant before correction and none after."
* **Do NOT say:** "p < 0.05, so it's real."

## C. Backtesting

**11. What is look-ahead bias?**
* **Simple:** "Using information in the backtest that you would not have had at the time."
* **Deeper:** "Examples: using this hour's close to trade at this hour's open, normalising with the full-sample mean, or choosing parameters on the test period."

**12. Why can't you randomly split time-series data?**
* **Simple:** "A random split trains on the future and tests on the past."
* **Deeper:** "Neighbouring hours are similar because volatility clusters, so a random split puts near-copies in both train and test and overstates skill."

**13. What is walk-forward validation?**
* **Simple:** "Train on the past, test on the next block, move forward, retrain, repeat, like a real trader would."
* **Deeper:** "I use an expanding window with 6-month blocks and purge training rows whose labels overlap the next block."

**14. How did you prevent overfitting?**
* **Simple:** "I fixed the design before seeing the data, used few interpretable features and simple models, and touched the test set only once."
* **Deeper:** "Every experiment is logged, including failures. I have a rule to prefer logistic regression unless LightGBM is clearly better. And I run kill tests: other thresholds, costs, delays, horizons, removed features, label permutation."

**15. How did you choose thresholds?**
* **Simple:** "On validation data only, from a small grid fixed in advance."
* **Deeper:** "I trade when probability exceeds 0.5 by k times the model's typical prediction spread. k comes from {0, 0.25, 0.5, 1, 1.5}, chosen by validation net Sharpe with a minimum number of trades."
* **Do NOT say:** "I picked the threshold with the best backtest."

**16. How did you model transaction costs? What is slippage?**
* **Simple:** "A 5 bp exchange fee, plus 1 bp half-spread and 1 bp slippage per side, charged every time the position changes. Slippage is the gap between the price you expect and the price you actually get."
* **Deeper:** "The fee comes from Binance's schedule; spread and slippage are assumptions because I have no historical order book. So I stress them at 2× and 3× and add extra slippage."
* **Do NOT say:** "Costs are negligible for BTC."

## D. Risk

**17. What is the Sharpe ratio?**
* **Simple:** "Average return divided by volatility, annualised. Return per unit of risk."
* **Deeper:** "It is noisy. Over half a year, its standard error is about 1.4, so a Sharpe of 1.5 can easily be luck."

**18. What is Sortino?**
* **Simple:** "Like Sharpe, but it only penalises downside volatility."

**19. What is maximum drawdown?**
* **Simple:** "The biggest fall from a peak to a trough in the equity curve."
* **Follow-up:** "Why does it matter?" → "It's the pain you must survive to collect the average return."

**20. What is VaR?**
* **Simple:** "A loss level that is exceeded only rarely. For example, 1-day 99% VaR is the loss exceeded on about 1 day in 100."
* **Deeper:** "I compute it three ways: historical, normal and Student-t. I backtest it with the Kupiec test, which counts how often losses actually exceed VaR."
* **Do NOT say:** "VaR is the maximum loss."

**21. What is Expected Shortfall?**
* **Simple:** "The average loss on the bad days beyond VaR."
* **Deeper:** "It tells you *how bad* the tail is, not just where it starts, which matters with fat tails."

**22. Why stress test a strategy?**
* **Simple:** "History shows only one path. Stress tests ask what happens if costs, volatility or the signal get worse."

## E. Machine learning

**23. Why logistic regression?**
* **Simple:** "It's simple, interpretable, gives probabilities, and is hard to overfit with few features."

**24. Why LightGBM?**
* **Simple:** "To check whether non-linear patterns or feature interactions add value. I only use it if it clearly beats logistic regression."

**25. Why not deep learning?**
* **Simple:** "With a weak signal and about 80k noisy samples, a deep model would mostly fit noise. A simpler model that fails honestly is more useful than a complex one that overfits."

**26. What is calibration?**
* **Simple:** "When the model says 55%, the event should happen about 55% of the time."
* **Deeper:** "I check it by decile, as in my credit project, and add the average future return per decile, because that is what matters for trading."

## F. Python and SQL

**27. How do you compute a rolling feature without leakage in pandas?**
* **Simple:** "`rolling` windows end at the current row, so they only look back. Then I take the value from the candle that closed at the decision time, `shift(1)` on a regular minute grid."

**28. How would you find gaps in 1-minute data in SQL?**
* **Simple:** use `LAG` to compare each timestamp with the previous one.

  ```sql
  SELECT open_time,
         LAG(open_time) OVER (ORDER BY open_time) AS prev_time,
         open_time - LAG(open_time) OVER (ORDER BY open_time) AS gap
  FROM klines
  QUALIFY gap > INTERVAL '1 minute';   -- or wrap it in a subquery where QUALIFY is unsupported
  ```

**29. How would you resample 1-minute bars to 1-hour bars in SQL?**

  ```sql
  SELECT date_trunc('hour', open_time) AS hour,
         arg_min(open, open_time) AS open, max(high) AS high,
         min(low) AS low, arg_max(close, open_time) AS close, sum(volume) AS volume,
         count(*) AS minutes_present
  FROM klines GROUP BY 1 ORDER BY 1;
  ```

  "I keep `minutes_present` so incomplete hours can be excluded instead of silently trusted."
