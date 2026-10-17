"""
Core Markov-chain machinery for the 3-state credit-rating model
(0 = Good, 1 = Watchlist, 2 = Default, absorbing).

Implements:
  * a through-the-cycle (TTC) quarterly transition matrix,
  * the Belkin-Suchower-Forest / Vasicek single-factor threshold model that
    conditions the TTC matrix on a systematic macro factor Z, and
  * generic multi-period (matrix-power) transition and default-probability
    utilities, plus cohort (empirical) estimation from panel data.

Reference for the macro-conditioning approach: Belkin, Suchower & Forest
(1998), "A one-parameter representation of credit risk and transition
matrices", CreditMetrics Monitor; Wilson (1997); Trueck & Rachev (2009),
"Rating Based Modeling of Credit Risk", ch. 3 & 5.
"""
import numpy as np
from scipy.stats import norm

STATES = ["Good", "Watchlist", "Default"]
GOOD, WATCH, DEFAULT = 0, 1, 2


def ttc_quarterly_matrix():
    """Stylised through-the-cycle quarterly transition matrix, calibrated so
    that the implied *annual* matrix (see implied_annual_matrix) is broadly
    in line with publicly reported average one-year corporate default rates
    by broad credit-quality band (investment-grade-like ~0.1-0.3%, higher-
    yield/"watchlist"-like ~4-6%), as commonly cited in rating-agency
    studies (e.g. S&P/Moody's annual default studies)."""
    P = np.array([
        [0.9820, 0.0175, 0.0005],
        [0.0650, 0.9220, 0.0130],
        [0.0000, 0.0000, 1.0000],
    ])
    assert np.allclose(P.sum(axis=1), 1.0)
    return P


def implied_annual_matrix(P_quarterly):
    return np.linalg.matrix_power(P_quarterly, 4)


def _thresholds(P_row):
    """Convert a row of unconditional probabilities (Good/Watchlist/Default
    order) into Vasicek thresholds on the latent credit-quality axis, ordered
    worst (Default) to best (Good)."""
    p_default, p_watch, p_good = P_row[DEFAULT], P_row[WATCH], P_row[GOOD]
    c_default = norm.ppf(np.clip(p_default, 1e-10, 1 - 1e-10))
    c_watch_or_below = norm.ppf(np.clip(p_default + p_watch, 1e-10, 1 - 1e-10))
    return c_default, c_watch_or_below


def conditional_matrix(P_ttc, Z, rho=0.15):
    """Return the 3x3 transition matrix conditional on systematic macro
    factor Z (standard normal; Z>0 = benign, Z<0 = stressed), given
    asset/systematic correlation rho."""
    P = np.zeros_like(P_ttc)
    scale = np.sqrt(1 - rho)
    shift = np.sqrt(rho) * Z
    for i in (GOOD, WATCH):
        c_d, c_w = _thresholds(P_ttc[i])
        cum_d = norm.cdf((c_d - shift) / scale)
        cum_w = norm.cdf((c_w - shift) / scale)
        P[i, DEFAULT] = cum_d
        P[i, WATCH] = cum_w - cum_d
        P[i, GOOD] = 1 - cum_w
    P[DEFAULT] = [0.0, 0.0, 1.0]
    # numerical floor/renormalisation
    P = np.clip(P, 0, 1)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def multiperiod_matrix(P, n):
    return np.linalg.matrix_power(P, n)


def cumulative_default_prob(P, max_n, start_state):
    """Cumulative default probability for horizons 1..max_n quarters,
    starting from `start_state`."""
    probs = []
    for n in range(1, max_n + 1):
        Pn = multiperiod_matrix(P, n)
        probs.append(Pn[start_state, DEFAULT])
    return np.array(probs)


def estimate_transition_matrix_cohort(panel_long, state_col="state",
                                       id_col="borrower_id", time_col="quarter"):
    """Cohort (empirical frequency) estimator: pools one-period transitions
    across all borrowers and all quarters in the given panel slice."""
    panel_long = panel_long.sort_values([id_col, time_col])
    panel_long["next_state"] = panel_long.groupby(id_col)[state_col].shift(-1)
    valid = panel_long.dropna(subset=["next_state"])
    valid = valid[valid[state_col] != DEFAULT]  # default is absorbing/exit
    counts = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            counts[i, j] = ((valid[state_col] == i) & (valid["next_state"] == j)).sum()
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P_hat = counts / row_sums
    P_hat[DEFAULT] = [0.0, 0.0, 1.0]
    return P_hat, counts
