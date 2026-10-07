# Interview defence guide

How to use: every question has (1) a 20-30 second answer, (2) the technical depth behind it, (3) the maths, (4) the follow-up an interviewer is likely to ask, with an answer. Project-specific numbers are generated from the actual result tables (re-run `python run_project.py` and they update).

**Honesty rules to repeat in any interview:** costs are hypothetical; XGBoost did *not* clearly win; the test period was harder than validation; the evidence for the inventory gain is suggestive (wide intervals).

---

## DATA

### Q1. Why this dataset?

**1. Short interview answer.** M5 is the standard public retail-demand benchmark: real Walmart item-store daily sales with prices, calendar events and SNAP flags. It is realistic (intermittent, noisy, promotional) and big enough to compare statistical and ML methods fairly.

**2. Detailed technical answer.** M5 has 30,490 item-store series over 1,941 days in 3 states and 10 stores, plus weekly prices and an event calendar. It lets me test ML with price/event features and tie forecasts to inventory decisions. The official Kaggle files need credentials, so I used two independent public Hugging Face mirrors and verified every file by SHA-256 (identical in both mirrors) plus structural checks (30,490 series x 1,941 days, 6,841,121 price rows, validation file = prefix of evaluation file). I state the limitation that I could not compare with Kaggle's own checksums.

**3. Mathematical explanation.**

Not applicable (conceptual question).

**4. Likely follow-up:** *How do you know the mirror was not altered?*

> I cannot prove byte-equality with Kaggle, but two independently uploaded copies have identical hashes and the structure matches the documented release; I report that as a limitation rather than claim certainty.

### Q2. What is the target variable?

**1. Short interview answer.** Observed daily unit sales of one item in one store, y_t. For modelling I forecast the sum of demand over the next H days (7, 14, 28) from each forecast origin.

**2. Detailed technical answer.** At origin t (end of day t) the model predicts Target_H(t) = sum of the next H days. Every model - baselines, ETS, SARIMA, XGBoost - is evaluated on exactly this quantity, so the comparison is like for like. Statistical models forecast the daily path and I sum it (negative forecasts clipped at zero); XGBoost uses the *direct* strategy, one model per horizon. H = 3 is supplementary, needed only for the 3-day lead time of the inventory layer.

**3. Mathematical explanation.**

```
Target_H(t) = y[t+1] + y[t+2] + ... + y[t+H]        (H = 7, 14, 28; H = 3 supplementary)
```

**4. Likely follow-up:** *Why predict the sum rather than each day?*

> Because inventory needs *lead-time demand* (a sum), sums are less intermittent than daily values, and it avoids error accumulation of recursive multi-step forecasts.

### Q3. Why daily demand?

**1. Short interview answer.** Lead times and replenishment decisions are in days, the data are daily, and weekly patterns (SNAP, weekends) live at the daily level. I aggregate to H-day sums for evaluation.

**2. Detailed technical answer.** Daily data keep the weekly cycle and let the simulation run day by day (orders, arrivals, stockouts). The downside is intermittency (14 of 18 series have ADI >= 1.32), which is why I added a Croston baseline and evaluate on sums.

**3. Mathematical explanation.**

Not applicable (conceptual question).

**4. Likely follow-up:** *Would weekly forecasting be better?*

> Possibly for slow movers; but weekly buckets would blur lead times of 3 days and the weekday structure. I would test it, scoring both on the same lead-time sums.

### Q4. Why did you select these products?

**1. Short interview answer.** To get a reproducible, representative mix of high, medium and low demand using only training information: one store per state, demand tiers by TRAIN-mean percentile bands, seeded random draws. Nothing was chosen for good test performance.

**2. Detailed technical answer.** Steps: chronological split first; one store per state with the largest TRAIN volume (CA_3, TX_2, WI_3); eligibility = listed every TRAIN day from d_1 (3,590 series); tiers from percentile bands of TRAIN mean sales; 2 random draws (seed 42) per store x tier, different categories preferred and no repeated item -> 18 series, 18 distinct products, categories {'FOODS': 8, 'HOUSEHOLD': 6, 'HOBBIES': 4}. Selection ran once; nothing was repeated after seeing any forecasting result. I also verified (not selected on) that the eligible items stay listed in validation/test.

**3. Mathematical explanation.**

Not applicable (conceptual question).

**4. Likely follow-up:** *Isn't 18 series too few?*

> Yes, it limits power: bootstrap and DM intervals are wide and subgroup results (4-14 series) are descriptive. It was a deliberate trade-off for transparency; the pipeline scales by changing the config.

### Q5. What are the limitations of the data?

**1. Short interview answer.** Sales are not demand: stockouts are invisible, so observed sales may underestimate true demand. Also no real costs or lead times, and my sample is old, continuously listed items.

**2. Detailed technical answer.** (1) Zero sales are 68% of cells; I can identify structural zeros (no price row) and closures, but not true zero demand vs stockout; 7 of 18 series have >= 90-day zero spells while listed. (2) No inventory records, costs or supplier lead times - all assumed and labelled hypothetical. (3) Items listed since 2011: 12 of 18 series have a significantly negative trend (median trend -14% of the mean per year). (4) One 292-day test period.

**3. Mathematical explanation.**

Not applicable (conceptual question).

**4. Likely follow-up:** *What would you do about censored demand in a real company?*

> Use stock records to flag stockout days, treat them as censored observations (Tobit / Kaplan-Meier style estimators) or exclude them from training, and measure lost sales explicitly.


## TIME SERIES

### Q6. What is stationarity?

**1. Short interview answer.** A series is (weakly) stationary if its mean, variance and autocovariance structure do not change over time. Many classical models, like ARMA, assume it.

**2. Detailed technical answer.** Weak stationarity: constant mean, constant finite variance, autocovariance that depends only on the lag. Trends, level shifts and changing variance break it. Differencing or modelling the trend restores it. In my data the ADF/KPSS pair gives: conflict: ADF stationary / KPSS non-stationary: 14, stationary (both tests agree): 3, unit root (both tests agree): 1 - i.e. 14 of 18 series show 'ADF says no unit root but KPSS rejects level-stationarity', which is typical of a slowly moving mean or structural breaks.

**3. Mathematical explanation.**

```
E[y_t] = mu,   Var(y_t) = sigma^2 < inf,   Cov(y_t, y_{t-k}) = gamma(k)   for all t
```

**4. Likely follow-up:** *What did you do about the conflict?*

> I kept both d = 0 and d = 1 (and D = 0 / 1) candidates and let validation error choose; AIC cannot compare different differencing orders.

### Q7. What is the ADF test?

**1. Short interview answer.** The Augmented Dickey-Fuller test checks the null hypothesis of a unit root (non-stationarity). A small p-value means the series looks stationary.

**2. Detailed technical answer.** It regresses the first difference on the lagged level plus lagged differences (lag order by AIC); a significantly negative coefficient on the lagged level rejects the unit root. It has low power against slow mean reversion, and structural breaks make it reject too rarely or too often, so I pair it with KPSS.

**3. Mathematical explanation.**

```
Delta y_t = a + gamma * y_{t-1} + sum_{i=1..p} delta_i * Delta y_{t-i} + e_t
H0: gamma = 0 (unit root)   vs   H1: gamma < 0 (stationary)
```

**4. Likely follow-up:** *What if ADF and KPSS disagree?*

> Then the evidence is inconclusive: ADF rejects a unit root (H0) while KPSS rejects stationarity (its H0). That points to a trend/regime change; I treat differencing as a modelling choice to be validated.

### Q8. What is the KPSS test?

**1. Short interview answer.** KPSS reverses the null: H0 is that the series is stationary around a mean (or trend). A small p-value says non-stationary. I use it with ADF for a two-sided view.

**2. Detailed technical answer.** It builds the partial sums of residuals from the series' mean and compares their variability (LM statistic) with a long-run variance estimate; large partial-sum wandering means a unit-root-like component.

**3. Mathematical explanation.**

```
S_t = sum_{i<=t} e_i,   KPSS = (1/T^2) * sum_t S_t^2 / sigma_LR^2      (reject stationarity if large)
```

**4. Likely follow-up:** *Why use both tests?*

> Because their nulls are opposite: agreement gives confidence (3 'stationary' and 1 'unit root' series here); disagreement flags structure that needs modelling.

### Q9. What is the ACF?

**1. Short interview answer.** The autocorrelation function shows the correlation of the series with its own lags. Spikes at lag 7, 14, 21 suggest weekly seasonality.

**2. Detailed technical answer.** ACF(k) = corr(y_t, y_(t-k)); under no autocorrelation the 95% band is about +-1.96/sqrt(n). In my TRAIN data lag-7 autocorrelation exceeds the band for 15 of 18 series (mean 0.16, max 0.49) - present but modest, matching the low seasonal strength (median F_s 0.09).

**3. Mathematical explanation.**

```
rho(k) = Cov(y_t, y_{t-k}) / Var(y_t)        band = +-1.96 / sqrt(n)
```

**4. Likely follow-up:** *How did you use it for SARIMA?*

> After seasonal differencing the ACF has a spike near -0.5 at lag 7, the signature of a seasonal MA(1) term - or of over-differencing a weakly seasonal series; hence candidates with and without D = 1.

### Q10. What is the PACF?

**1. Short interview answer.** The partial autocorrelation is the correlation at lag k after removing the effect of the intermediate lags. It suggests the AR order: a sharp cut-off after lag p indicates AR(p).

**2. Detailed technical answer.** PACF(k) is the coefficient of y_(t-k) in an AR(k) regression. For a pure MA process the PACF decays gradually instead of cutting off: the seasonally differenced TRAIN series show median PACF -0.48, -0.33, -0.24, -0.18 at the seasonal lags 7, 14, 21, 28 (outside the 95% band at all four lags in 18 of 18 series; `eda_09_acf_pacf_seasonally_differenced.png`), which points to a seasonal MA term.

**3. Mathematical explanation.**

```
PACF(k) = last coefficient phi_kk of the AR(k) fit  y_t = phi_k1 y_{t-1} + ... + phi_kk y_{t-k} + e_t
```

**4. Likely follow-up:** *Why look at PACF if you pick the model by validation?*

> It narrows the candidate set to a handful of structures (small and fast) so I do not search blindly; validation error then decides.

### Q11. Why seasonal period 7?

**1. Short interview answer.** Daily retail data have a weekly cycle, so the natural seasonal period is 7. I confirm it with the ACF and with Holt-Winters AICc rather than assume it.

**2. Detailed technical answer.** AICc of Holt-Winters is lower than SES for 15 of 18 series. But seasonal strength is weak (median 0.09), and for 7/14/28-day *sums* the additive seasonal indices cancel, so weekly seasonality barely changes those forecasts (test WAPE SES 37.6% vs Holt-Winters 37.7% at 7 days).

**3. Mathematical explanation.**

```
seasonal period m = 7 : s_{t+h} = s_{t+h-7*ceil(h/7)}  (additive weekly index)
```

**4. Likely follow-up:** *What about yearly seasonality?*

> With only ~3.7 training years, month effects rest on 3-4 observations; the model sees month/week-of-year/Christmas features instead, and I flag the estimates as indicative.

### Q12. What is differencing?

**1. Short interview answer.** Subtracting the previous (or seasonal) value to remove trend or seasonality. d is the number of ordinary differences, D the number of seasonal ones.

**2. Detailed technical answer.** First difference (1-B)y removes a stochastic trend; seasonal difference (1-B^7)y removes a stable weekly pattern. Over-differencing injects negative autocorrelation (an MA(1) with theta = -1): independent noise differenced at lag 7 has autocorrelation -0.5 at lag 7. The seasonally differenced TRAIN series have a lag-7 autocorrelation of -0.48 (range -0.52 to -0.41); but removing a stable weekly pattern from noisy data gives the same -0.5, so the spike alone cannot tell a needed seasonal difference from an unnecessary one - validation error has to decide.

**3. Mathematical explanation.**

```
(1 - B) y_t = y_t - y_{t-1}        (1 - B^7) y_t = y_t - y_{t-7}
If y_t is white noise: corr((1-B^7)y_t, (1-B^7)y_{t-7}) = -0.5
```

**4. Likely follow-up:** *Which differencing did the data choose?*

> Validation error picked structures per series: (0,1,1)(0,1,1)7 x5, (1,0,1)(0,1,1)7 x4, (1,0,1)(1,0,1)7+c x4, (1,0,0)(1,0,0)7+c x3, (1,1,1)(0,1,1)7 x2; 11 of 18 use seasonal differencing although seasonal strength is weak (median F_s 0.09); with a seasonal MA(1) term the model behaves like a seasonal exponential smoother, so the weekly pattern is learned gradually rather than assumed fixed.

### Q13. What is SARIMA?

**1. Short interview answer.** SARIMA(p,d,q)(P,D,Q)_s combines autoregressive and moving-average terms for the series and for its seasonal lags, with differencing; it captures autocorrelation and weekly seasonality in one linear model.

**2. Detailed technical answer.** I used five small candidates with s = 7 and chose per series on validation error (AIC/BIC logged). SARIMA was the best model on validation at horizons [7, 14, 28] but ranked 4 / 2 / 2 of 9 on test at 7 / 14 / 28 days (test WAPE 38.0% / 33.7% / 31.6%), illustrating validation optimism. Parameters are fixed after fitting and the Kalman state is updated each day (`extend`), verified identical to a full re-filter.

**3. Mathematical explanation.**

```
Phi(B^s) phi(B) (1-B)^d (1-B^s)^D y_t = c + Theta(B^s) theta(B) e_t,   s = 7
```

**4. Likely follow-up:** *How did you avoid leakage when rolling the model?*

> Parameters are estimated on the fitting window only; each day the state absorbs one new observation; a perturbation test shows forecasts at origin t are unchanged if later data are replaced.

### Q14. What is exponential smoothing?

**1. Short interview answer.** A family of models whose forecast is a weighted average of past values with exponentially decaying weights: SES (level), Holt (level + trend), Holt-Winters (level + trend + seasonality).

**2. Detailed technical answer.** I use SES, Holt with damped trend (undamped trends explode over 28 steps) and additive Holt-Winters (multiplicative is impossible with zeros). Parameters are small (median alpha 0.05): slow-adapting. A Holt-Winters -> Holt -> SES fallback exists (0 fallbacks occurred).

**3. Mathematical explanation.**

```
l_t = alpha*y_t + (1-alpha)*l_{t-1}                (SES: forecast = l_t)
Holt (damped): b_t = beta*(l_t-l_{t-1}) + (1-beta)*phi*b_{t-1};  yhat_{t+h} = l_t + (phi+...+phi^h) b_t
HW additive: yhat_{t+h} = l_t + (phi+..+phi^h) b_t + s_{t+h-7*ceil(h/7)}
```

**4. Likely follow-up:** *Why is SES competitive here?*

> A plausible reason: demand is noisy with weak weekly seasonality (median seasonal strength 0.09), so a slowly adapting level (median SES alpha 0.05) is hard to beat; Holt had the lowest test WAPE at 7 days (SES second, 0.04 points behind).

### Q15. What is Croston's method and why SBA?

**1. Short interview answer.** Croston forecasts intermittent demand by smoothing demand sizes and inter-demand intervals separately; the Syntetos-Boylan correction removes its bias.

**2. Detailed technical answer.** 14 of 18 series are intermittent/lumpy (ADI >= 1.32), so I included Croston-SBA (alpha = 0.1) as a baseline. It did not beat ETS or the moving average here (test WAPE 40.7% / 33.6% at 7 / 28 days).

**3. Mathematical explanation.**

```
z_t = z_{t-1} + a*(y_t - z_{t-1}),  p_t = p_{t-1} + a*(q_t - p_{t-1})   (only when y_t > 0)
forecast = (1 - a/2) * z_t / p_t           (SBA bias correction)
```

**4. Likely follow-up:** *Why did it not win?*

> Its constant-rate forecast ignores the level shifts and the weekly/SNAP pattern, and 28-day sums are far less intermittent than daily values.


## MACHINE LEARNING

### Q16. Why XGBoost?

**1. Short interview answer.** It handles non-linear effects and interactions between lags, calendar and price on tabular data, is fast and robust with little tuning, and has an exact, fast explainer (TreeSHAP). Deep learning was unnecessary for 18 series.

**2. Detailed technical answer.** Gradient-boosted trees were the backbone of the M5 winners. I use one *global* model per horizon (pooled over series) so splits are shared. Honest result: it is competitive, not dominant - best pooled test WAPE at 14 days (32.8%) and 28 days (29.8%), but never significantly better than the other non-naive models in the Diebold-Mariano tests, and significantly worse than Holt, HoltWinters, SES at 7 days.

**3. Mathematical explanation.**

```
F(x) = sum_{m=1..M} eta * f_m(x),   f_m = regression tree fitted to the gradient of the squared-error loss
```

**4. Likely follow-up:** *Why not a neural network?*

> No compelling methodological reason: small data, a need for explainability and a risk of unreproducible gains. I would revisit with thousands of series.

### Q17. Why regression?

**1. Short interview answer.** The target is a continuous non-negative quantity (sum of units), so regression with a squared-error objective gives the conditional mean, which is what inventory planning needs.

**2. Detailed technical answer.** Squared error estimates E[Target | features]; MAE would estimate the median; quantile loss would give a service-level quantile (an extension I recommend). Predictions are clipped at zero.

**3. Mathematical explanation.**

```
minimise sum_i (Target_i - F(x_i))^2   ->   F(x) ~ E[Target | x]
```

**4. Likely follow-up:** *Would a Tweedie/Poisson objective be better?*

> Plausibly for intermittent counts; I kept squared error for interpretability, consistency with RMSE and SHAP in units; it is a natural next experiment.

### Q18. How were the lag features created?

**1. Short interview answer.** As the sales k days before the first forecast day: lag_1 is the last observed day, lag_7 the same weekday last week, up to lag_28.

**2. Detailed technical answer.** Row = (series, forecast origin t). lag_k(t) = y[t+1-k], so lags are measured from the first forecast day t+1. A unit test compares 200 random rows with explicit loops; the audit re-checks 400 rows on the real panel.

**3. Mathematical explanation.**

```
lag_k(t) = y[t + 1 - k],  k in {1, 7, 14, 21, 28}
```

**4. Likely follow-up:** *Why measure lags from t+1 rather than t?*

> Because then lag_7 aligns with the weekday of the first forecast day (same as seasonal naive) and the origin day's own value is available, which the rolling windows also include.

### Q19. What are rolling features?

**1. Short interview answer.** Summary statistics of the most recent window: mean and standard deviation over the last 7/14/28 days, ending at the forecast origin.

**2. Detailed technical answer.** rolling_mean_w(t) = mean(y[t-w+1..t]) and the sample standard deviation (ddof = 1), computed with exact sliding windows. They carry level and volatility; SHAP shows they dominate the model (74% of attribution at 7 days, 73% at 28 days).

**3. Mathematical explanation.**

```
rolling_mean_w(t) = (1/w) * sum_{j=0..w-1} y[t-j]        rolling_std_w(t) = sqrt( sum (y[t-j]-mean)^2 / (w-1) )
```

**4. Likely follow-up:** *Why is the std useful?*

> It tells the model how volatile the item is, which matters when demand is lumpy. SHAP: rolling_std_28 carries 6.4% of the 28-day model's attribution (correlation of feature value and SHAP +0.77) but 4.6% at 7 days (correlation +0.07). For count data the standard deviation rises with the mean (median within-series correlation of the two 28-day windows 0.80), so credit is shared with the level and I do not read it as a separate effect.

### Q20. How can rolling features cause leakage?

**1. Short interview answer.** If the window includes future days (centred windows, off-by-one) or statistics are computed on the whole series, the feature secretly contains the target.

**2. Detailed technical answer.** Typical bugs: centred rolling windows (t-3..t+3), forgetting to shift so that today's value predicts today, scaling or encoding with global statistics, building targets whose windows overlap the validation period. I wrote a negative-control unit test: a deliberately centred-window feature is caught by the perturbation test, which proves the audit has power.

**3. Mathematical explanation.**

```
SAFE:    mean(y[t-6..t])      LEAKY:  mean(y[t-3..t+3])  (uses y[t+1], y[t+2], y[t+3])
```

**4. Likely follow-up:** *How do you know there is no leakage in your final model?*

> 40/40 automated checks pass, including perturbation tests: replace all data after a random cut-off with wild values and verify that features, model forecasts, sigma and policy decisions up to the cut-off are identical.

### Q21. How did you prevent leakage overall?

**1. Short interview answer.** Chronological split, features that only use data up to the forecast origin, training rows cut so their targets end before the next period, hyper-parameters chosen on validation only, sigma from validation residuals, and automated perturbation tests.

**2. Detailed technical answer.** (1) Split by date; (2) history features end at the origin; (3) the only features with later timestamps are calendar counts (SNAP, events, Christmas) that are deterministic functions of the published calendar and verified independent of sales; (4) XGBoost rows satisfy origin + H <= fit_end - 1; (5) ETS/SARIMA/XGBoost are refit on train+validation only after selection; (6) the test period is evaluated once; (7) the inventory policy uses only information at the end of day t.

**3. Mathematical explanation.**

Not applicable (conceptual question).

**4. Likely follow-up:** *Is using future calendar information not leakage?*

> No: holidays and SNAP days are published long in advance, exactly like a retailer's planning calendar; I documented them as 'known-ahead' features and tested that they do not depend on sales. Using *future prices or sales* would be leakage.

### Q22. Direct or recursive multi-step forecasting?

**1. Short interview answer.** Direct: one model per horizon, trained on the target 'sum of the next H days'. It avoids error accumulation and matches the inventory need.

**2. Detailed technical answer.** Recursive forecasting would feed predictions back as lags and compound errors; for sums it is also awkward. Direct models cost more training runs (4 here) but each is tuned for its horizon (chosen max_depth by horizon: {3: 6, 7: 3, 14: 6, 28: 5}).

**3. Mathematical explanation.**

Not applicable (conceptual question).

**4. Likely follow-up:** *Why not one model with horizon as a feature?*

> Possible, and it shares data across horizons, but separate models are easier to explain and tune; with 4 horizons the cost is small.

### Q23. How did you tune XGBoost without overfitting?

**1. Short interview answer.** Modestly and only on validation: 24 configurations per horizon (default + random draws), early stopping on validation MAE, then a refit on train+validation with the chosen number of trees. The test period was never touched.

**2. Detailed technical answer.** Validation MAE improved by 1.1% (H=3), 1.3% (H=7), 4.9% (H=14), 5.5% (H=28) versus the default configuration, so the tuning gains were modest; the validation score of the winner is optimistic because it is the minimum of 24 draws.

**3. Mathematical explanation.**

Not applicable (conceptual question).

**4. Likely follow-up:** *Why not cross-validation?*

> Time-series CV needs expanding windows with gaps; with one pooled model and a few thousand rows a single chronological hold-out is simpler and sufficient - I accept the extra variance.


## EVALUATION

### Q24. Why MAE?

**1. Short interview answer.** It is in the units of demand, easy to explain, and robust to occasional spikes. It is optimal for the median.

**2. Detailed technical answer.** MAE answers 'on average how many units am I off?'. At 28 days pooled test MAE is 14.9 units for XGBoost and 16.1 for the 28-day moving average.

**3. Mathematical explanation.**

```
MAE = (1/n) * sum |y_i - f_i|
```

**4. Likely follow-up:** *Why do you also report RMSE?*

> Because the two can rank models differently when spikes matter - and in this project they do.

### Q25. Why RMSE?

**1. Short interview answer.** It squares errors, so it punishes large misses (the demand spikes that cause stockouts) and is the natural scale for safety stock, since sigma is a root-mean-square error.

**2. Detailed technical answer.** RMSE rewards the conditional mean. Rankings differ from MAE: at 7 days the best RMSE is HoltWinters while the best WAPE is Holt. My sigma_L is the RMSE of validation errors, so a biased model gets a bigger safety stock.

**3. Mathematical explanation.**

```
RMSE = sqrt( (1/n) * sum (y_i - f_i)^2 ) ;  RMSE^2 = Var(error) + Bias^2
```

**4. Likely follow-up:** *When is RMSE misleading?*

> When a few huge misses dominate: test RMSE is 1.2-2.2x validation RMSE while MAE grows only 1.1-1.5x. At 7 days the worst 5% of windows produce 65-80% of the test squared error against 47-55% in validation; the lost units of the forecast-driven policy are concentrated in November-December (66% of them in 21% of the test days).

### Q26. Why sMAPE?

**1. Short interview answer.** It is a scale-free percentage error that is bounded and handles zeros better than MAPE. I report it but do not rely on it.

**2. Detailed technical answer.** sMAPE = mean(200*|e|/(|y|+|f|)) with 0/0 := 0. Weakness for intermittent data: a forecast of exactly zero when actual is zero scores perfectly, so methods that output exact zeros collect free perfect scores - at 28 days the best sMAPE belongs to MovingAverage (38%), which ranks 3 of 9 on WAPE; it forecasts exactly zero in 7.5% of the 28-day test windows (the actual is zero in 6.9%) whereas Croston, Holt, HoltWinters, SARIMA, SES, XGBoost never do.

**3. Mathematical explanation.**

```
sMAPE = (100/n) * sum 2|y_i - f_i| / (|y_i| + |f_i|)      (0/0 := 0; range 0..200)
```

**4. Likely follow-up:** *Then why report it?*

> Because it is standard in forecasting competitions and lets readers compare; I flag its bias towards zero forecasts.

### Q27. Why WAPE?

**1. Short interview answer.** It is total absolute error divided by total actual demand - a volume-weighted, zero-safe percentage that business people understand ('we miss about a third of volume').

**2. Detailed technical answer.** WAPE = MAE / mean demand. It is my headline metric: 37.6% / 32.8% / 29.8% for the best models at 7/14/28 days. Because it is volume-weighted it favours models that do well on the high-volume series (best 28-day model there: XGBoost), which is why I also run an equal-weighted DM test.

**3. Mathematical explanation.**

```
WAPE = 100 * sum |y_i - f_i| / sum |y_i|
```

**4. Likely follow-up:** *What if total demand is zero?*

> WAPE is undefined for that group (the code returns a missing value instead of a number); I only compute it on pooled groups with positive demand.

### Q28. Why not MAPE?

**1. Short interview answer.** MAPE divides by the actual, which is zero on many days (68% of cells), so it is undefined or explosive, and it penalises over-forecasts more than under-forecasts.

**2. Detailed technical answer.** |y - f|/|y| is unbounded for over-forecasts and capped at 100% for under-forecasts, which biases model selection toward under-forecasting; sMAPE/WAPE/MASE avoid division by small actuals.

**3. Mathematical explanation.**

```
MAPE = (100/n) * sum |y_i - f_i| / |y_i|     undefined when y_i = 0
```

**4. Likely follow-up:** *Would MASE be better?*

> MASE (scaled by in-sample naive error) is a good scale-free alternative and I would add it; WAPE was chosen for business readability.

### Q29. Why chronological validation?

**1. Short interview answer.** Because time series are ordered and autocorrelated: shuffling lets the model see the future, inflating accuracy. Chronological splits mimic real deployment.

**2. Detailed technical answer.** Train 2011-01-29-2014-10-17, validation 2014-10-18-2015-08-04, test 2015-08-05-2016-05-22. Within each period I use *rolling origins* (every day), so each model is scored on 264-290 overlapping windows per series. Overlap means the effective sample is about n/H, which the DM test accounts for with a HAC variance (lag H-1).

**3. Mathematical explanation.**

```
Origins t = a-1 ... b-1-H  (window t+1..t+H inside the period)
DM = mean(d) / sqrt(LRV(d)/n),  d_t = L(e_A,t) - L(e_B,t),  LRV with lag H-1
```

**4. Likely follow-up:** *What did the DM test tell you?*

> XGBoost is not significantly better than ETS/SARIMA/MA at 14/28 days (smallest Holm p = 0.42) and is significantly worse than Holt, HoltWinters, SES at 7 days; XGBoost and SARIMA beat naive and seasonal naive significantly at all three horizons.


## EXPLAINABILITY

### Q30. What is SHAP?

**1. Short interview answer.** SHAP assigns each feature a contribution to one prediction using Shapley values from game theory, so contributions add up exactly to the difference between the prediction and the average prediction.

**2. Detailed technical answer.** For trees, TreeSHAP computes them exactly and fast. I explain the final train+validation models on test-period origins (4,000 sampled rows) and verify additivity (max error 1.4e-04). Findings: rolling means dominate (74% at 7 days); SNAP-day counts 4.5%; price 3.7%.

**3. Mathematical explanation.**

```
phi_i = sum_{S subset F\{i}} |S|! (|F|-|S|-1)! / |F|!  *  [ f(S u {i}) - f(S) ]
sum_i phi_i + base_value = prediction
```

**4. Likely follow-up:** *Why SHAP rather than feature importance?*

> Impurity importances are global and biased; SHAP is local (per prediction), additive and consistent, and supports dependence plots and individual waterfalls.

### Q31. What does a positive SHAP value mean?

**1. Short interview answer.** It means that, for that prediction, the feature's value pushes the forecast above the average prediction (the base value); a negative value pushes it below.

**2. Detailed technical answer.** Example: for the 7-day model the base value is about 15.4 units; a high 28-day rolling mean gives a large positive SHAP (correlation of feature value and SHAP 0.97). The magnitude is in units of the forecast.

**3. Mathematical explanation.**

```
prediction = base_value + sum_i phi_i ;   phi_i > 0 raises, phi_i < 0 lowers the prediction
```

**4. Likely follow-up:** *Can a positive SHAP value be read as 'the feature increases demand'?*

> No - it describes how the *model* uses the feature, not what happens in the world if the feature is changed.

### Q32. Is SHAP causal?

**1. Short interview answer.** No. SHAP explains the fitted model's behaviour. Correlated features share credit, and a feature can proxy for something unobserved.

**2. Detailed technical answer.** Example in this project: `price` has a negative SHAP relation (corr -0.81), but item prices change rarely (median 2 changes in TRAIN), so it most likely separates cheap, fast-selling items from expensive, slow ones (Spearman correlation of mean price and mean sales across the 18 series -0.69) - not a price elasticity. Causal claims would need experiments or causal modelling.

**3. Mathematical explanation.**

Not applicable (conceptual question).

**4. Likely follow-up:** *How would you estimate price elasticity?*

> With within-item price variation and controls (or a randomised price test) - e.g. a log-log regression with item fixed effects - rather than from SHAP.


## INVENTORY

### Q33. What is safety stock?

**1. Short interview answer.** Extra stock held to protect against uncertainty in demand (and lead time) during the replenishment lead time.

**2. Detailed technical answer.** SS = z * sigma_L, where sigma_L is the standard deviation of the lead-time demand forecast error. In my Policy B, sigma_L is the RMSE of the model's out-of-sample (validation) errors of the L-day sum; in Policy A it is the textbook sqrt(L) x trailing daily std. Using real forecast errors captures autocorrelation that the i.i.d. formula misses.

**3. Mathematical explanation.**

```
SS = z * sigma_L ;   i.i.d. daily demand:  sigma_L = sqrt(L) * sigma_daily
```

**4. Likely follow-up:** *What did using validation errors change?*

> With the same 28-day-mean point forecast as Policy A, only the sigma estimate changed and total cost moved -7.3% at L=7 (95%, MEDIUM) - 86% of the headline change (-8.5%).

### Q34. What is the reorder point (ROP)?

**1. Short interview answer.** The inventory position at which you place a new order: expected demand during the lead time plus safety stock.

**2. Detailed technical answer.** I check the inventory position (on hand + on order) at the end of each day and order up to S = ROP + Q when it falls to the ROP. Orders arrive L+1 days after the end-of-day order so the protection interval is exactly the L forecast days.

**3. Mathematical explanation.**

```
ROP_t = mu_L(t) + z * sigma_L ;  order if IP_t <= ROP_t, quantity = ceil(ROP_t + Q_t - IP_t)
```

**4. Likely follow-up:** *Why inventory position and not on-hand?*

> Because outstanding orders already cover part of the demand; using on-hand alone would order again and again during the lead time.

### Q35. Why use z?

**1. Short interview answer.** z converts a target service level (a probability) into a number of standard deviations of safety stock, assuming the lead-time error is roughly normal.

**2. Detailed technical answer.** z is the normal quantile: 90% -> 1.282, 95% -> 1.645, 98% -> 2.054 (prescribed in the project). Normality is an approximation: for the H-day sums the standardised validation residuals have skewness -0.08 to 0.45; the larger problem was that sigma estimated on validation understated test errors, so the realised cycle service level of a nominal 95% was 85% (B) / 75% (A).

**3. Mathematical explanation.**

```
P(D_L <= mu_L + z*sigma_L) = alpha  =>  z = Phi^{-1}(alpha)
z(90%) = 1.282, z(95%) = 1.645, z(98%) = 2.054
```

**4. Likely follow-up:** *Do empirical quantiles fix that?*

> Not systematically: I ran the empirical-quantile variant for every model, lead time and service level (81 cells). Total cost changed by a median of 1.7% (range -7.5% to +7.0%), fill rate by at most 0.8 points and cycle service level by a median of 1.4 points (mean change +0.3), so it did not repair the shortfall - the dominant issue is the shift in error size between validation and test.

### Q36. What is service level?

**1. Short interview answer.** The probability (or share) of demand you can satisfy from stock. It is a design target, not a guarantee.

**2. Detailed technical answer.** I report several: cycle service level (share of replenishment cycles without a stockout - what z targets), fill rate (units served / units demanded), and in-stock day rate. At 95% target and L=7: cycle service level 75% (A) / 85% (B), fill rate 96.0% / 96.3%, in-stock days 98.8% / 99.1%.

**3. Mathematical explanation.**

```
Type-1 (cycle) SL = P(no stockout in a cycle) ;  Type-2 (fill rate) = 1 - E[shortage per cycle] / E[demand per cycle]
```

**4. Likely follow-up:** *Which should the business use?*

> Fill rate is closer to customer experience and cost; cycle service level is what the z-formula controls. Choose from the cost of a lost sale, then monitor the realised value.

### Q37. What is EOQ?

**1. Short interview answer.** The Economic Order Quantity balances ordering cost against holding cost to minimise total cost per unit time: Q* = sqrt(2 D S / H).

**2. Detailed technical answer.** Derivation: cost per year = (D/Q)*S + (Q/2)*H; minimising over Q gives Q*. I use D = 365 x (28-day forecast)/28, S = $1.00 per order, H = 25% of unit cost per year (hypothetical), giving cycle lengths of 12-65 days on TRAIN data. Assumptions: constant demand, no shortages in the EOQ itself - the ROP/safety stock handles uncertainty.

**3. Mathematical explanation.**

```
TC(Q) = (D/Q)*S + (Q/2)*H   =>   dTC/dQ = 0   =>   Q* = sqrt(2*D*S/H)
```

**4. Likely follow-up:** *Why EOQ with variable demand?*

> It is a standard (s, S)/(s, Q) building block; with a forecast that updates daily, Q* updates too. A more exact approach would optimise (s, S) numerically or use a cost-aware simulation.

### Q38. What happens when lead time increases?

**1. Short interview answer.** Both the expected lead-time demand and its uncertainty grow, so the reorder point and safety stock rise and total cost increases.

**2. Detailed technical answer.** mu_L grows ~ L and sigma_L ~ sqrt(L) for i.i.d. demand (faster if errors are autocorrelated). In the simulation the mean total cost of Policy B (averaged over other factors) is 559 / 679 / 836 for L = 3 / 7 / 14 days - a 40% range, versus 8% across the three service levels.

**3. Mathematical explanation.**

```
mu_L = L * mu_d ;  sigma_L = sqrt(L) * sigma_d (i.i.d.) ;  ROP = mu_L + z*sigma_L  -> increases with L
```

**4. Likely follow-up:** *What can a retailer do about it?*

> Shorten or stabilise the lead time (supplier agreements, closer sourcing). In my results the lead-time range of mean cost (40%) exceeded both the service-level range (8%) and the largest cost difference between the forecast-driven and the historical policy (14%).

### Q39. What happens when the service level increases?

**1. Short interview answer.** z rises, safety stock and holding cost increase, and stockouts fall - with diminishing returns because z grows faster at high service levels.

**2. Detailed technical answer.** z goes 1.282 -> 1.645 -> 2.054 for 90/95/98%; the extra safety stock per point of service increases. Whether it pays depends on the stockout penalty: in my simulation the cost-minimising level on the 90/95/98% grid is: LOW 90%, MEDIUM 98% (the grid edge), HIGH 98% (the grid edge). Mean total cost of Policy B moves only 8% from 90% to 98%.

**3. Mathematical explanation.**

```
SS(alpha) = z(alpha) * sigma_L ;  d SS / d alpha = sigma_L / phi(z)  (rises steeply as alpha -> 1)
```

**4. Likely follow-up:** *How would you pick the service level?*

> From the critical ratio: alpha* ~ p / (p + h*T_cycle) where p is the cost of a lost unit and h*T_cycle the holding cost of a leftover unit for one cycle - then verify by simulation and monitor the realised level.


## BUSINESS

### Q40. Why does better forecasting not necessarily mean lower inventory cost?

**1. Short interview answer.** Because forecast accuracy metrics are symmetric and average over all periods, while inventory cost is asymmetric (lost sales vs holding) and driven by bias and tail errors during lead times, plus how uncertainty is turned into safety stock.

**2. Detailed technical answer.** In my results the lowest-WAPE model was the cheapest in only 2 of 27 cells; the Spearman correlation between error rank and cost rank was +0.47 (LOW penalty), -0.08 (MEDIUM), -0.26 (HIGH). Reasons: (1) asymmetric costs favour quantile-type forecasts; (2) noisy forecasters get large sigma and hold more stock; (3) bias (all models under-forecast on average in the test year) goes with lost sales (Spearman -0.81 between test bias and lost units across models at L=7, p = 0.014; an association across 8 non-naive models, not proof); (4) sigma estimated on validation is optimistic for the models with the smallest validation sigma (XGBoost, SARIMA, MovingAverage), which are also the three with the largest test/validation RMSE ratio at 7 days (a winner's-curse pattern, not proof of cause).

**3. Mathematical explanation.**

```
Optimal order-up-to for asymmetric cost (newsvendor): S* = F^{-1}( p / (p + h) )   -> a QUANTILE of demand, not the mean
```

**4. Likely follow-up:** *So should you ignore accuracy?*

> No - accuracy is necessary, but the objective should be the decision: evaluate with the inventory simulation, train quantile/cost-aware models, and control bias.

### Q41. What assumptions did you make?

**1. Short interview answer.** Hypothetical costs and lead times, lost sales, constant lead time, no minimum order size or perishability, observed sales as demand, independence across items, constant sigma over the test period.

**2. Detailed technical answer.** Costs: unit cost 70% of price, holding 25%/year, order cost $1, stockout penalty 0.25x / 1x / 3x unit margin (LOW/MEDIUM/HIGH); lead times 3/7/14 days; initial inventory cover L+14 days. EOQ cycles on TRAIN data: 12-65 days; the critical-ratio argument that set the scenarios used TRAIN data only. I never present these as real business costs.

**3. Mathematical explanation.**

Not applicable (conceptual question).

**4. Likely follow-up:** *How sensitive are conclusions to them?*

> Strongly for dollar amounts, less so for direction: B is cheaper than A in 21 of 27 grid cells; the factors rank by how much they move the mean cost of B: stockout-penalty scenario (85%), then lead time (40%), then service level (8%).

### Q42. What would you change in a real company?

**1. Short interview answer.** Use real costs and lead-time distributions, stockout/inventory records to handle censored demand, promotions and price plans as drivers, rolling re-estimation of sigma, quantile/cost-aware forecasts and a controlled pilot before rollout.

**2. Detailed technical answer.** Concretely: (1) estimate the cost of a lost sale from margin and substitution data; (2) model stochastic lead times; (3) re-estimate sigma weekly on rolling errors and monitor realised service; (4) train quantile models for the lead-time demand; (5) respect pack sizes/MOQs/shelf life; (6) forecast hierarchically (store-department-item) to help intermittent items; (7) A/B-test the policy on a subset of stores.

**3. Mathematical explanation.**

Not applicable (conceptual question).

**4. Likely follow-up:** *What is the first thing you would do?*

> Measure the real stockout cost and lead-time distribution: in my simulation moving between the LOW and HIGH penalty scenario changed the mean total cost of Policy B by 85%, whereas the choice among the non-naive forecasting models changed it by a median of 10% (MEDIUM penalty; up to 54% with the HIGH penalty) - the penalty assumption mattered more than the model choice.

### Q43. What are the limitations of the project?

**1. Short interview answer.** Hypothetical costs, censored demand, only 18 series and one test year, and wide uncertainty on the inventory gain. The ML model did not clearly beat simple methods.

**2. Detailed technical answer.** Specific numbers: bootstrap intervals for B vs A include zero in 27 of 27 cells; the test year was harder than validation for every model (RMSE ratio 1.17-2.23x); the model with the best validation WAPE was the cheapest on test in 0 of 27 cells (median regret 10% with the MEDIUM penalty); the realised cycle service level at the 95% target and L = 7 was 85% (B) / 75% (A). Mitigations: chronological protocol, leakage audit, honest reporting.

**3. Mathematical explanation.**

Not applicable (conceptual question).

**4. Likely follow-up:** *What are you most proud of?*

> The discipline: a leakage audit with a negative control, a frozen protocol, reporting results that went against the 'ML wins' narrative, and a clear answer that accuracy is not business value.

### Q44. Why did you use the Diebold-Mariano test?

**1. Short interview answer.** To check whether differences in forecast accuracy between two models are statistically meaningful rather than noise.

**2. Detailed technical answer.** It tests whether the mean loss differential is zero, with a long-run variance that accounts for overlapping forecast errors (lag H-1), the Harvey-Leybourne-Newbold small-sample correction and Holm adjustment across comparisons. I aggregate across series with equal weights (scaled losses) to respect cross-sectional dependence. Power is low at long horizons because the effective sample is about n/H.

**3. Mathematical explanation.**

```
d_t = |e_A,t|/scale - |e_B,t|/scale ;  DM = mean(d)/sqrt(LRV/n) ;  HLN: DM* = DM * sqrt((n+1-2H+H(H-1)/n)/n),  t_{n-1}
```

**4. Likely follow-up:** *Why can XGBoost lead on WAPE but not on DM?*

> WAPE is volume-weighted (busy series dominate) while the DM panel loss gives each series equal weight. By group, the best 28-day model is XGBoost for the 6 high-volume series, Holt for the 6 medium and Holt for the 6 low-volume ones (XGBoost for the 4 regular and MovingAverage for the 14 intermittent series), so the ranking depends on which series carry the weight.

