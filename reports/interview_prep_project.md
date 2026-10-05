# Interview Preparation, Part 2: My Project (with real results)

Simple English. For each question: **Simple answer** → **Deeper answer** → **Likely follow-up** → **Do NOT say**.
Concept questions (what is VaR, autocorrelation, walk-forward…) are in Part 1: `interview_prep_concepts.md`.

**Numbers to remember**

| What | Number |
|---|---|
| Data | 4.79M BTCUSDT minutes, 110/110 files checksum-verified |
| Test period | 2024-01 → 2026-09; 24,095 hours; run once |
| Test AUC | **0.545** (CI 0.538–0.552); accuracy 53.5% vs base rate 50.5% |
| Permutation test | p = 0.005 (200 shuffles) |
| Net Sharpe | **−5.84** (CI −6.94 to −4.77); gross +0.16; buy-and-hold +0.76 |
| Costs | 7 bp per side, 14 bp round trip; **break-even 0.19 bp per side** |
| Return per decile | −1.3 to +1.9 bp |
| Robustness | 23/23 variants negative after costs, all with AUC > 0.53 |
| ETH (same frozen pipeline) | AUC 0.546, net Sharpe −3.70 |
| Verdict | **WEAK / INCONCLUSIVE**: real signal, not tradable |

---

## I. Explaining the project

**30 seconds**
> "I extended my credit-risk project to market data. Using 4.8 million minutes of Binance Bitcoin data, I asked whether the next hour's direction can be predicted. I fixed the whole design before seeing the data. I found a real short-term reversal: after the price goes up, the next hour is slightly more likely to go down. Out of sample the AUC is 0.545 and the permutation p-value is 0.005. But when I backtested with fees and slippage, the edge was 0.19 basis points per trade against 7 basis points of cost. So my conclusion is: real signal, not tradable."

**2 minutes**, in order:
1. The credit project: PD → validation → ECL → Monte Carlo VaR → stress.
2. The same framework on market data: question, data, audit (including the Binance timestamp quirk).
3. Pre-registration, and why.
4. Features with a leakage test.
5. Walk-forward validation, with the test period touched once.
6. Result: statistically real (AUC 0.545, p = 0.005, holds on ETH) but economically dead (net Sharpe −5.8, break-even 0.19 bp).
7. Kill tests: 23 of 23 variants negative.
8. What I'd test next: binary prediction markets pay on direction, not size.

**Do NOT say:** "I built a trading strategy that…" or "I found alpha."

## A. Data and data quality

**Why Binance?**
* **Simple:** "It's the official exchange archive. It's free, has checksums, goes back to 2017, and I can legally download it from India."
* **Deeper:** "It's raw data, not a cleaned Kaggle file, so the audit found real problems. It's also the settlement source of Polymarket's hourly BTC markets."
* **Follow-up:** "Why not Polymarket itself?" → "India blocked Polymarket in May 2026, there is no official historical order book, and fine-grained price history only goes back weeks. I didn't want a project I couldn't reproduce."
* **Do NOT say:** "Because it was easy."

**Why BTCUSDT? Why 1-minute data?**
* **Simple:** "BTCUSDT is the most liquid crypto pair, so if a pattern fails there it's not because of a thin market. I used 1-minute data so features and execution times could be exact to the minute, even though I predict 1 hour ahead."
* **Follow-up:** "Then why not predict 1 minute ahead?" → "Costs. A 1-minute move is far smaller than 14 bp. My 15-minute variant already had net Sharpe −17."

**What was the most interesting data problem?**
* **Simple:** "About two weeks of December 2017 candles start 20.799 seconds after each minute, and ETH has exactly the same stretch."
* **Deeper:** "If I had snapped them down to the minute, my features would have seen 20 seconds of the future. I followed my pre-written rule and treated them as missing. It only affects 2017–18 training data."
* **Do NOT say:** "The data was clean."

**How did you handle missing data and outliers?**
* **Simple:** "0.63% of minutes were missing, and none from 2022 onward. I never filled gaps. I kept all 408 big moves because they match real events: the COVID crash, FTX, 10 October 2025."

## B. Statistics

**Your AUC is only 0.545. Isn't that basically random?**
* **Simple:** "It's small but not random. The 95% interval is 0.538–0.552, and in 200 label shuffles none came close."
* **Deeper:** "With 24,000 test hours, 0.545 is many standard errors from 0.5. But significant isn't the same as useful. I measured the money directly: about 1–2 bp per hour, which is too small."
* **Do NOT say:** "0.545 is good."

**How do you know it's not multiple testing?**
* **Simple:** "I fixed the design in advance, logged all 44 experiments, used Holm correction across 16 features, and ran a permutation test. It also replicated on ETH, which I never tuned on."

**Why block bootstrap and not normal standard errors?**
* **Simple:** "Hourly returns aren't independent. Volatility clusters: the autocorrelation of |return| is 0.31. Normal standard errors would be too small."

## C. Python / pandas

**How did you make the features leakage-free in pandas?**
* **Simple:** "I put the data on a regular 1-minute grid, used backward-looking rolling windows, and took each feature from the candle that closes exactly at the decision time. Then I *tested* it: I replaced all future data with random numbers and checked that no feature changed."

**How did you handle 4.8 million rows?**
* **Simple:** "Parquet files, vectorised pandas and NumPy, about 0.5 GB in memory. The whole pipeline runs in about 10 minutes."
* **Do NOT say:** "I used Polars or Spark." I didn't.

## D. SQL

**Did you use SQL?**
* **Simple:** "Not in this project; it's pandas on parquet. But the same steps translate directly to SQL, for example finding gaps with `LAG`, or resampling to hourly bars with `GROUP BY date_trunc`." (See Part 1, questions 28–29.)
* **Do NOT say:** that you used SQL or ClickHouse here.

## E. Machine learning

**Why classification and not regression?**
* **Simple:** "Hourly returns have kurtosis 38. A regression would be dominated by a few crash hours. I predict direction, then judge the money using the real returns."

**Logistic regression vs LightGBM: what happened?**
* **Simple:** "LightGBM passed my simplicity rule exactly at the threshold, 7 of 10 folds. On test it had slightly higher AUC (0.545 vs 0.540), but logistic regression lost less money."
* **Deeper:** "So the borderline AUC win didn't turn into money. Next time I'd require the complex model to win on the economic metric in validation too. I did **not** switch models after seeing the test."
* **Do NOT say:** "LightGBM is always better."

**Why not deep learning?**
* **Simple:** "The signal is tiny and noisy. A deep model would mostly learn noise, and even my simple models already showed that the problem is cost, not model capacity."

**What did the model actually learn?**
* **Simple:** "Reversal. The most important features were the 4-hour return, the 1-hour return, and where the price closed in the last hour's range. Up moves predict slightly lower next hours."

## F. Trading

**Why does a signal with 53.5% accuracy lose money?**
* **Simple:** "Accuracy counts how often you're right, not how much you make. The average edge is about 1–2 bp per hour, and every trade costs 7 bp to enter and 7 to exit."
* **Deeper:** "To clear 14 bp round trip the model would need P(up) above about 0.72. Its most confident predictions average 0.60."

**Could you fix it with maker orders or a VIP fee tier?**
* **Simple:** "Not realistically. The break-even cost is 0.19 bp per side, which is below even the maker fee of most tiers. Maker orders also get filled mostly when the price moves against you."
* **Do NOT say:** "With lower fees it would work." Show the number instead.

**What about Polymarket?**
* **Simple:** "That's my next experiment, not a result. A binary contract pays on direction, not size, and direction is what my model predicts. On the test data, its most confident Up calls were right 56.6% of the time, versus a break-even of about 51.75% for a contract bought at 0.50."
* **Deeper:** "But I never observed Polymarket prices. Their market makers see the same Binance data, so the price at the start of the hour may already include this. And liquidity is thin. It needs a forward test with real prices."
* **Do NOT say:** "My model would make money on Polymarket."

## G. Backtesting

**How did you make sure you didn't tune on the test set?**
* **Simple:** "The design was committed to git before I downloaded data. The model and threshold were committed before the test ran. The experiment log has exactly one TEST row."

**How did you pick the threshold?**
* **Simple:** "From a fixed grid, on validation only. Every option lost money in validation, so the rule picked the least bad one, k = 1.5 long-only. That was already a warning sign before the test."

**How did you model costs?**
* **Simple:** "5 bp Binance taker fee (sourced), plus 1 bp spread and 1 bp slippage (assumed), charged on every position change, plus actual funding history. I stressed 2× and 3×. Even at zero cost the gross Sharpe is only 0.16."

## H. Risk

**What were the risk numbers?**
* **Simple:** "Daily 99% VaR was 3.5% historical, versus 2.2% assuming a normal distribution, so the normal understates the tail by about 38%. Kupiec didn't reject the VaR model (p = 0.60). The bootstrap Monte Carlo gave a 100% chance of losing money over a year."

**Why a block bootstrap and not the credit module's Monte Carlo?**
* **Simple:** "In credit, default correlation can't be observed, so I had to assume it with a one-factor model. Here I observe the strategy's returns directly, so I resample them in blocks. That keeps the fat tails and the volatility clustering."

**What stress scenario hurt most?**
* **Simple:** "Costs: 3× costs gives Sharpe −16. Also, because it's a dip-buying strategy, it was long during the 5 August 2024 crash and lost 7.2% that day."

## I. Project defence

**What did you actually achieve if the strategy loses?**
* **Simple:** "I answered the question with evidence. I showed a real pattern exists, measured how big it is, and proved it isn't worth trading. That saves real money compared with deploying it."

**What is the strongest evidence your signal is real?**
* **Simple:** "Four independent checks: the untouched test period (AUC 0.545, CI above 0.5), a permutation test (p = 0.005), every half-year above 0.53, and ETH with the frozen design (AUC 0.546)."

**Why did performance decay over time?**
* **Simple:** "The market got calmer. My volatility features drifted a lot (PSI up to 3.5), and AUC fell from 0.567 to 0.532. Next time I'd normalise the features by volatility."

**What would you do with more data or time?**
* **Simple:** "A forward test on Polymarket hourly markets with real prices and order books; volatility-normalised features; and slower, lower-turnover versions."

## J. Adversarial questions

**"Your backtest looks profitable. Why should I believe it?"**
* **Simple:** "It isn't profitable, and I'd be suspicious if it were. Gross Sharpe is only 0.16, and net is −5.8."
* **If they point at the 2.05 gross Sharpe variant:** "That's an always-in-market version I looked at *after* the test. It trades 3,685 times a year, so its net Sharpe is −8.8. Picking it now would be exactly the data snooping I tried to avoid."

**"How do you know you didn't overfit?"**
* **Simple:** "Overfitting shows up as a model that works in validation and dies in test. Mine *predicted* almost as well in test (0.565 → 0.545) and on a new asset, so the prediction isn't overfit. The money was negative everywhere, so there was nothing to overfit to."

**"What happens if costs double?"**
* **Simple:** "Net Sharpe goes from −5.8 to −11.4. It's already dead at normal costs."

**"What if the regime changes?"**
* **Simple:** "It already did: volatility fell by about a third in the test period. The prediction weakened (AUC 0.567 → 0.532) and no regime made money, whether high or low volatility, trending or range-bound."

**"Why should I trust LightGBM?"**
* **Simple:** "You shouldn't trust it blindly, and I didn't. It only passed my simplicity rule at the threshold, and on test the simpler logistic regression lost less money."

**"What if your model doesn't work?"**
* **Simple:** "Economically it doesn't, and that's the finding. What *works* is the process: it caught the problem before any money was at risk."

**"Isn't 'no signal' a boring result?"**
* **Simple:** "It's what most honest backtests look like. And it isn't 'no signal': it's a real signal that's too small for this market structure. That's exactly the kind of thing I'd want to test in prediction markets, where you're paid on direction."

**"You used an AI assistant. How do I know you understand this?"**
* **Simple:** "Ask me anything in the code. And I can tell you where the AI was wrong and how we caught it: a timestamp check that would have flagged every row, an outlier score dividing by zero, a chart title that stated a 'fact' before any data existed, and a stage that would have overwritten BTC results with ETH ones."
* **Do NOT say:** "The AI did it." Or "I didn't use AI."

**"What is the one thing you'd change?"**
* **Simple:** "Make the model-selection rule economic, not just statistical, and use volatility-normalised features. Both are lessons from the result, and I wrote them down instead of changing the test after seeing it."
