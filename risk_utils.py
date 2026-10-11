"""
risk_utils.py
=============
Core statistical engine for the Market Risk Statistical Validation project.

Everything here is CLASSICAL statistics (distributions, hypothesis testing,
OLS regression, confidence intervals, VaR + backtesting) — no machine
learning is used anywhere in this module. This module is imported by both
main.ipynb (the analysis notebook) and app.py (the Streamlit dashboard) so
the logic is written once and shown twice.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# 1. DATA & RETURNS
# ---------------------------------------------------------------------------

def download_prices(tickers, start, end):
    """
    Download adjusted close prices for a list of tickers using yfinance.
    `tickers` should include the market index (e.g. '^NSEI') plus the
    individual stocks (e.g. 'RELIANCE.NS', 'TCS.NS').
    Returns a wide DataFrame: columns = tickers, index = date.
    """
    import yfinance as yf
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all")


def log_returns(prices):
    """Log returns: r_t = ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def descriptive_stats(returns: pd.Series) -> pd.Series:
    """Mean, std (daily + annualised), skewness, excess kurtosis for one return series."""
    return pd.Series({
        "mean_daily": returns.mean(),
        "std_daily": returns.std(),
        "annualised_vol": returns.std() * np.sqrt(TRADING_DAYS),
        "skewness": stats.skew(returns),
        "excess_kurtosis": stats.kurtosis(returns),  # Fisher definition: 0 = normal
    })


# ---------------------------------------------------------------------------
# 2. DISTRIBUTION ANALYSIS — Normal vs Student-t
# ---------------------------------------------------------------------------

def fit_normal(returns):
    mu, sigma = stats.norm.fit(returns)
    return {"loc": mu, "scale": sigma}


def fit_student_t(returns):
    """Fit a Student-t: returns (df, loc, scale). Low df => fat tails."""
    df, loc, scale = stats.t.fit(returns)
    return {"df": df, "loc": loc, "scale": scale}


def goodness_of_fit_normal(returns):
    """
    Jarque-Bera test for normality.
    H0: returns are normally distributed (skew=0, excess kurtosis=0).
    A significant p-value (< 0.05) => reject normality.
    """
    jb_stat, jb_p = stats.jarque_bera(returns)
    return {"jarque_bera_stat": jb_stat, "p_value": jb_p, "reject_normality_at_5pct": jb_p < 0.05}


# ---------------------------------------------------------------------------
# 3. CALM vs STRESSED REGIME — Hypothesis Testing
# ---------------------------------------------------------------------------

def classify_regime(returns: pd.Series,window: int = 21,method: str = "median"):
    """
    Classify each day as 'low_vol' or 'high_vol' based on
    trailing realised volatility (rolling standard deviation).

    This is a SLOW, structural regime definition based on the
    recent volatility environment. It is used for hypothesis
    testing in Section 3 and later as the reference regime
    against which a faster daily stress-flagging rule is evaluated.

    method='median'
        -> below/at median = low_vol
        -> above median = high_vol

    method='tercile'
        -> bottom third = low_vol
        -> top third = high_vol
        -> middle third dropped
    """

    roll_vol = returns.rolling(window).std()
    roll_vol = roll_vol.dropna()

    if method == "median":
        cutoff = roll_vol.median()

        regime = pd.Series(
            np.where(
                roll_vol > cutoff,
                "high_vol",
                "low_vol"
            ),
            index=roll_vol.index
        )

    elif method == "tercile":
        q1, q2 = roll_vol.quantile([1 / 3, 2 / 3])

        regime = pd.Series(
            index=roll_vol.index,
            dtype=object
        )

        regime[roll_vol <= q1] = "low_vol"
        regime[roll_vol >= q2] = "high_vol"

        regime = regime.dropna()

    else:
        raise ValueError(
            "method must be 'median' or 'tercile'"
        )

    return regime, roll_vol


def compare_regimes(returns: pd.Series, regime: pd.Series):
    """
    Compare returns across low- and high-volatility regimes using:

    - Welch's t-test:
        Tests whether the mean return differs between regimes.
        Allows unequal variances.

    - Mann-Whitney U test:
        Non-parametric comparison of the two return samples.
        Used as the primary inference because Section 2 rejected
        the Normality assumption.

    - Levene's test:
        Tests whether the two regimes have equal variances.

    - 95% confidence interval:
        Gives the uncertainty range for the difference in mean returns.

    Regimes:
        'low_vol'  -> lower-volatility regime
        'high_vol' -> higher-volatility regime
    """

    # Align return observations with available regime labels
    aligned = returns.loc[returns.index.isin(regime.index)]
    regime_aligned = regime.reindex(aligned.index)

    # Split returns into the two volatility regimes
    low_vol = aligned[regime_aligned == "low_vol"].dropna()
    high_vol = aligned[regime_aligned == "high_vol"].dropna()

    # 1. Welch's t-test
    # H0: mean(low_vol) = mean(high_vol)
    # Allows unequal variances
    t_stat, t_p = stats.ttest_ind(
        low_vol,
        high_vol,
        equal_var=False
    )

    # 2. Mann-Whitney U test
    # Primary non-parametric comparison
    u_stat, u_p = stats.mannwhitneyu(
        low_vol,
        high_vol,
        alternative="two-sided"
    )

    # 3. Levene's test for equality of variance
    # H0: variances are equal
    lev_stat, lev_p = stats.levene(
        low_vol,
        high_vol
    )

    # 4. Difference in means + 95% CI
    # Defined as:
    #  mean(low_vol) - mean(high_vol)
    diff = low_vol.mean() - high_vol.mean()

    var_low = low_vol.var(ddof=1)
    var_high = high_vol.var(ddof=1)

    n_low = len(low_vol)
    n_high = len(high_vol)

    # Standard error for unequal variances
    se = np.sqrt(
        var_low / n_low +
        var_high / n_high
    )

    # Welch-Satterthwaite degrees of freedom
    df_welch = (
        (var_low / n_low + var_high / n_high) ** 2
        /
        (
            (var_low / n_low) ** 2 / (n_low - 1)
            +
            (var_high / n_high) ** 2 / (n_high - 1)
        )
    )

    # Critical value for 95% CI
    t_critical = stats.t.ppf(
        0.975,
        df=df_welch
    )

    ci_low = diff - t_critical * se
    ci_high = diff + t_critical * se

    return {
        "n_low_vol": n_low,
        "n_high_vol": n_high,

        "mean_low_vol": low_vol.mean(),
        "mean_high_vol": high_vol.mean(),

        "t_test_stat": t_stat,
        "t_test_p": t_p,

        "mannwhitney_stat": u_stat,
        "mannwhitney_p": u_p,

        "levene_stat": lev_stat,
        "levene_p": lev_p,

        "mean_diff": diff,
        "mean_diff_95ci": (ci_low, ci_high),
    }


# ---------------------------------------------------------------------------
# 4. MARKET MODEL REGRESSION (classical OLS, NOT machine learning)
# ---------------------------------------------------------------------------

def market_model_regression(stock_returns: pd.Series, market_returns: pd.Series):
    """
    Classical single-factor market-model regression:
        R_i = alpha + beta * R_m + epsilon

    This is OLS estimated via statsmodels (closed-form, classical inference),
    not a fitted ML model. Returns alpha, beta, R^2, standard errors,
    p-values and 95% confidence intervals — i.e. the full classical
    inferential output, not just point estimates.

    Standard errors are HC1 (heteroskedasticity-robust / White) rather than
    classical OLS SEs. Daily equity returns exhibit volatility clustering
    (variance is not constant over time), which violates the homoskedasticity
    assumption behind classical OLS SEs and tends to understate them —
    making beta look more statistically significant than it actually is.
    Robust SEs correct for that without changing the point estimates of
    alpha/beta themselves (only their estimated uncertainty).

    Note: this is a MARKET MODEL, not CAPM, because no risk-free rate /
    excess-return adjustment is applied. It answers "how sensitive is this
    stock's return to the market's return", which is what beta/systematic
    risk means in a risk-management context.
    """
    df = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    df.columns = ["stock", "market"]

    X = sm.add_constant(df["market"])
    model = sm.OLS(df["stock"], X).fit(cov_type="HC1")

    return {
        "alpha": model.params["const"],
        "beta": model.params["market"],
        "r_squared": model.rsquared,
        "alpha_se": model.bse["const"],
        "beta_se": model.bse["market"],
        "alpha_p": model.pvalues["const"],
        "beta_p": model.pvalues["market"],
        "alpha_ci95": tuple(model.conf_int().loc["const"]),
        "beta_ci95": tuple(model.conf_int().loc["market"]),
        "n_obs": int(model.nobs),
        "se_type": "HC1 (heteroskedasticity-robust)",
        "model": model,  # full statsmodels result object, for residual diagnostics if needed
    }


# ---------------------------------------------------------------------------
# 5. STRESS-DAY FLAGGING + TYPE I / II ERROR ANALYSIS
# ---------------------------------------------------------------------------

def flag_stress_days(returns: pd.Series, percentile: float = 5.0):
    """
    FAST/reactive rule: flag a day as 'predicted stress' if its return falls
    below the given lower percentile of the historical return distribution
    (e.g. percentile=5 -> flag days in the worst 5% of single-day returns).
    """
    threshold = np.percentile(returns, percentile)
    flagged = returns < threshold
    return flagged, threshold


def confusion_matrix_metrics(actual_stress: pd.Series, predicted_stress: pd.Series):
    """
    actual_stress: boolean series, e.g. regime == 'stressed' from classify_regime
    predicted_stress: boolean series, e.g. from flag_stress_days

    Both series are aligned on their common index before comparison.
    Returns TP/FP/FN/TN counts and the standard classification metrics,
    used here as a STATISTICAL evaluation of a detection rule
    (not as an ML model score).
    """
    idx = actual_stress.index.intersection(predicted_stress.index)
    a = actual_stress.reindex(idx).astype(bool)
    p = predicted_stress.reindex(idx).astype(bool)

    tp = int(((a) & (p)).sum())
    fp = int((~a & p).sum())
    fn = int((a & ~p).sum())
    tn = int((~a & ~p).sum())

    recall = tp / (tp + fn) if (tp + fn) else np.nan          # sensitivity: catches true stress
    specificity = tn / (tn + fp) if (tn + fp) else np.nan     # avoids false alarms
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan             # Type I error rate
    fnr = fn / (fn + tp) if (fn + tp) else np.nan             # Type II error rate

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "recall_sensitivity": recall, "specificity": specificity,
        "precision": precision,
        "type_I_error_rate_FPR": fpr, "type_II_error_rate_FNR": fnr,
    }


# ---------------------------------------------------------------------------
# 6. VALUE AT RISK
# ---------------------------------------------------------------------------

def historical_var(returns: pd.Series, alpha: float = 0.05):
    """Empirical (historical) VaR: the alpha-quantile of the empirical return distribution."""
    return -np.percentile(returns, alpha * 100)


def parametric_var(returns: pd.Series, alpha: float = 0.05):
    """Parametric (Normal) VaR: VaR = -(mu + z_alpha * sigma)."""
    mu, sigma = returns.mean(), returns.std()
    z = stats.norm.ppf(alpha)
    return -(mu + z * sigma)


# ---------------------------------------------------------------------------
# 7. VaR BACKTESTING — Kupiec Proportion-of-Failures (POF) test
# ---------------------------------------------------------------------------

def count_violations(returns: pd.Series, var_estimate: float):
    """
    In-sample violation count: a violation = a day where the realised loss
    exceeds a SINGLE fixed VaR estimate (estimated from the same data being
    tested). Kept for illustration, but see `rolling_out_of_sample_var` below
    for the backtest that should actually be reported — this in-sample
    version has look-ahead bias (the VaR estimate 'knows' about every day it's
    being tested against, including future ones).
    """
    violations = returns < -var_estimate
    return int(violations.sum()), len(returns)


def rolling_out_of_sample_var(returns: pd.Series, window: int = 250,
                               alpha: float = 0.05, method: str = "historical"):
    """
    Out-of-sample VaR backtest with NO look-ahead bias.

    For each day t (t > window), the VaR estimate used to test day t is
    computed using ONLY returns[t-window : t] — i.e. only information
    available strictly before day t. The model is then asked "would this
    have flagged day t as a violation?" using data it could not have seen
    when day t actually happened. This is the walk-forward procedure the
    Kupiec test should be run against, not a single full-sample estimate.

    method: 'historical' or 'parametric'

    Returns a DataFrame indexed by date with columns:
    ['var_estimate', 'actual_return', 'violation']
    """
    if method not in ("historical", "parametric"):
        raise ValueError("method must be 'historical' or 'parametric'")

    values = returns.values
    dates = returns.index
    records = []

    for i in range(window, len(values)):
        trailing_window = pd.Series(values[i - window:i])
        if method == "historical":
            var_est = historical_var(trailing_window, alpha)
        else:
            var_est = parametric_var(trailing_window, alpha)

        actual = values[i]
        violation = actual < -var_est
        records.append((dates[i], var_est, actual, violation))

    return pd.DataFrame(records, columns=["date", "var_estimate", "actual_return", "violation"]).set_index("date")


def kupiec_pof_test(n_violations: int, n_obs: int, alpha: float = 0.05):
    """
    Kupiec (1995) unconditional-coverage / Proportion-of-Failures test.

    H0: the model's violation rate equals the expected rate alpha.
    LR_POF = -2 ln[ (1-alpha)^(T-x) * alpha^x ] + 2 ln[ (1-x/T)^(T-x) * (x/T)^x ]
    LR_POF ~ chi-square(1) under H0.

    A p-value < 0.05 means we reject H0: the VaR model is mis-calibrated
    (too many or too few violations relative to what it promised).
    """
    x, T = n_violations, n_obs
    pi_hat = x / T if T else np.nan

    if x == 0:
        # avoid log(0): use the limiting form (no violations observed)
        log_l0 = T * np.log(1 - alpha)
        log_l1 = T * np.log(1 - pi_hat) if pi_hat < 1 else 0
    elif x == T:
        # avoid log(0): symmetric limiting form (every day violated)
        log_l0 = T * np.log(alpha)
        log_l1 = T * np.log(pi_hat) if pi_hat > 0 else 0
    else:
        log_l0 = (T - x) * np.log(1 - alpha) + x * np.log(alpha)
        log_l1 = (T - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)

    lr_stat = -2 * (log_l0 - log_l1)
    p_value = stats.chi2.sf(lr_stat, df=1)

    return {
        "observed_violations": x,
        "expected_violations": round(alpha * T, 2),
        "observed_rate": pi_hat,
        "expected_rate": alpha,
        "LR_stat": lr_stat,
        "p_value": p_value,
        "reject_model_at_5pct": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# 8. TRANSLATE VaR INTO A CURRENCY LOSS ESTIMATE
# ---------------------------------------------------------------------------

def var_to_exposure_loss(var_estimate: float, exposure_amount: float):
    """
    Translate a VaR percentage into an estimated rupee (or any-currency) loss
    for a given exposure. E.g. var_estimate=0.025, exposure_amount=4e7
    ("Rs 4 crore") -> estimated one-day loss at that confidence level.
    Pure arithmetic — included because a risk number only becomes useful to
    a risk manager once it's translated into a monetary figure.
    """
    return var_estimate * exposure_amount
