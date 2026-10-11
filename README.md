# Market Risk Statistical Validation Framework

A classical-statistics market-risk framework for analysing return distributions, volatility regimes, systematic market exposure, stress-detection trade-offs, and Value-at-Risk model calibration.

The project is deliberately focused on **classical statistical inference and risk-model validation** rather than machine learning. The central question is not simply *“Can I calculate VaR?”* but:

> **Can I statistically evaluate whether a market-risk model is appropriate and calibrated when tested on unseen observations?**

---

## Project Overview

The framework uses the **NIFTY 50** as the market benchmark and four large-cap Indian equities:

- `RELIANCE.NS`
- `TCS.NS`
- `HDFCBANK.NS`
- `INFY.NS`

Analysis period:

- **1 January 2019 – 30 December 2024**
- Daily observations
- Data pulled from Yahoo Finance through `yfinance`

The workflow is:

```text
Market Prices
      │
      ▼
   Log Returns
      │
      ├───────────────┬────────────────┐
      ▼               ▼                ▼
Distribution      Volatility        Market Model
Analysis           Regimes              │
      │               │                 ▼
      │               │             Beta / R²
      └───────┬───────┘
              ▼
       Stress Detection
              │
              ▼
             VaR
              │
              ▼
   Walk-Forward Backtesting
              │
              ▼
        Kupiec POF Test
              │
              ▼
      Model Validation
```

The core theme is:

$$
\boxed{\text{Understand Risk} \rightarrow \text{Measure Risk} \rightarrow \text{Validate Risk}}
$$

---

# Key Findings

## 1. Return distribution is strongly non-Normal

For NIFTY 50:

- Jarque-Bera statistic: **≈ 30,022**
- Jarque-Bera p-value: **< 1e-300**
- Excess kurtosis: **≈ 21.88**
- Student-t fitted degrees of freedom: **≈ 3.15**

The evidence strongly rejects the Normality assumption and indicates substantial heavy-tail behaviour.

This motivates explicit investigation of the effect of distributional assumptions on VaR.

---

## 2. Low- vs high-volatility regimes differ mainly in dispersion

A 21-day trailing realised-volatility measure is split at its sample median into:

- `low_vol`
- `high_vol`

This creates **727 observations in each regime**.

The median split is a **relative volatility-regime definition**, not a claim that 50% of trading days were crisis days.

Results:

- Levene's test for equal variance: **p ≈ 3.27e-26**
- Welch t-test for mean difference: **p = 0.926**
- Mann-Whitney U test: **p = 0.121**
- 95% CI for low-vol minus high-vol mean return:

$$
(-0.00126,\;0.00115)
$$

The important finding is:

$$
\boxed{\text{High volatility does not necessarily mean lower average return}}
$$

The regime distinction appears primarily through **dispersion**, not return direction.

---

## 3. Systematic market exposure differs substantially across stocks

The classical market model is:

$$
R_{i,t}=\alpha_i+\beta_iR_{m,t}+\epsilon_{i,t}
$$

Estimated Betas:

| Stock | Beta | 95% Beta CI | R² |
|---|---:|---:|---:|
| RELIANCE | 1.093 | 0.998 – 1.188 | 0.478 |
| TCS | 0.707 | 0.613 – 0.801 | 0.292 |
| HDFCBANK | 1.076 | 0.994 – 1.158 | 0.577 |
| INFY | 0.839 | 0.749 – 0.928 | 0.307 |

TCS and INFY have confidence intervals entirely below 1, supporting lower-than-market systematic sensitivity.

RELIANCE and HDFCBANK have point estimates above 1, but their confidence intervals include values consistent with a market Beta of 1.

HDFCBANK has the highest \(R^2\), so the market factor explains the largest share of its return variation. TCS has the lowest \(R^2\), leaving the largest share unexplained by this one-factor market model.

The key distinction is:

$$
\boxed{\text{Volatility} \neq \text{Beta} \neq R^2}
$$
---

## 4. A single-day shock is not the same as a volatility regime

A fast stress rule flags the worst X% of daily returns and is compared with the slower `high_vol` regime.

The results show a strong sensitivity-selectivity trade-off:

| Flag threshold | Recall | Specificity | Type I / FPR | Type II / FNR |
|---:|---:|---:|---:|---:|
| Worst 10% | 14.58% | 94.38% | 5.62% | 85.42% |
| Worst 5% | 9.35% | 99.20% | 0.80% | 90.65% |
| Worst 1% | 2.06% | 100.00% | 0.00% | 97.94% |

This demonstrates:

$$
\boxed{\text{Extreme return} \neq \text{high-volatility regime}}
$$

The analysis is explicitly **in-sample and retrospective**. It evaluates agreement between two risk-state definitions; it is not presented as a live early-warning system.

---

# VaR Model Validation

Two VaR approaches are evaluated:

### Historical VaR

Uses the empirical lower-tail quantile:

$$
VaR_{\alpha}^{Hist}
=
-\operatorname{Quantile}_{\alpha}(R)
$$

No parametric distribution is imposed.

### Parametric Normal VaR

Uses:

$$
VaR_{\alpha}^{Normal}
=
-(\mu+z_{\alpha}\sigma)
$$

with a Normal distribution assumption.

VaR is additionally translated into a monetary loss using an illustrative portfolio exposure.

---

## 95% VaR

Reference full-sample VaR magnitudes:

| Method | VaR |
|---|---:|
| Historical | 1.5477% |
| Parametric-Normal | 1.8666% |

At 95%, the Normal-parametric estimate happens to be more conservative.

### Walk-forward out-of-sample validation

The backtest uses a **250-day rolling estimation window**. Each day's VaR is estimated using only the preceding observations, then tested against the next unseen return.

Out-of-sample period:

- **1,224 trading days**
- **16 January 2020 – 30 December 2024**

| Method | Violations | Expected | Observed Rate | Kupiec p-value | Decision |
|---|---:|---:|---:|---:|---|
| Historical | 66 | 61.2 | 5.39% | 0.5340 | Not rejected |
| Parametric-Normal | 59 | 61.2 | 4.82% | 0.7717 | Not rejected |

At 95%, neither model is rejected by the **Kupiec Proportion-of-Failures (POF) test**.

This means the observed exception frequencies are statistically consistent with the intended 5% violation rate.

It does **not** prove that either model is universally correct; Kupiec tests unconditional coverage, not violation independence or clustering.

---

## 99% VaR and the fat-tail effect

At the stricter 99% confidence level:

| Method | VaR |
|---|---:|
| Historical | 3.1314% |
| Parametric-Normal | 2.6612% |

The ordering reverses:

$$
95\%:\quad VaR_{Normal}>VaR_{Historical}
$$

$$
99\%:\quad VaR_{Historical}>VaR_{Normal}
$$

This is important because the effect of heavy tails becomes more relevant deeper in the distribution.

### 99% walk-forward validation

| Method | Violations | Expected | Observed Rate | Kupiec LR | p-value | Decision |
|---|---:|---:|---:|---:|---:|---|
| Historical | 21 | 12.24 | 1.72% | 5.2156 | 0.0224 | **Reject** |
| Parametric-Normal | 28 | 12.24 | 2.29% | 15.0256 | 0.0001 | **Reject** |

Both models are rejected at 99%, but the Normal specification is rejected substantially more strongly.

The result supports the broader finding that:

$$
\boxed{\text{Extreme-tail calibration is harder than moderate-tail calibration}}
$$

It also shows that Historical VaR is not a perfect solution: a 250-day window contains very little empirical information in the 1% tail, so finite-sample tail uncertainty remains significant.

---

# Methodology

| Section | Technique | Question |
|---|---|---|
| 1 | Log returns + descriptive statistics | What do the return series look like? |
| 2 | Normal vs Student-t + Q-Q + Jarque-Bera | Are returns Normal or heavy-tailed? |
| 3 | 21-day rolling volatility + Welch + Mann-Whitney + Levene | Do return characteristics differ across low/high-volatility regimes? |
| 4 | Classical OLS market model + HC1 robust SEs | How sensitive is each stock to market movements? |
| 5 | Percentile stress flag + confusion matrix | What is the false-alarm vs missed-event trade-off? |
| 6 | Historical VaR + Normal Parametric VaR | What is the estimated one-day downside threshold? |
| 7 | Walk-forward out-of-sample VaR + Kupiec POF | Is the realised violation frequency consistent with the stated confidence level? |
| 7b | 99% VaR comparison + backtest | Does model behaviour deteriorate deeper in the tail? |
| 7c | Consolidated validation summary | How do VaR models compare across confidence levels? |

---

# Statistical Design Notes

## Market Model vs CAPM

The regression is deliberately called a **market model**, not CAPM:

$$
R_{stock,t}=\alpha+\beta R_{market,t}+\epsilon_t
$$

No risk-free rate or excess-return transformation is applied.

Beta measures systematic market sensitivity, while \(R^2\) measures the explanatory power of the NIFTY factor.

---

## HC1 Robust Inference

The OLS point estimates are classical, but inference uses **HC1 heteroskedasticity-robust standard errors**.

This is motivated by the observed time-varying volatility of daily equity returns.

HC1 changes:

- standard errors
- test statistics
- p-values
- confidence intervals

but not the OLS point estimates of \(\alpha\) and \(\beta\).

---

## Low- vs High-Volatility Regimes

The regime definition is:

$$
\sigma_t = Std(R_{t-20},...,R_t)
$$

using a 21-day rolling window.

The sample median separates:

- `low_vol`
- `high_vol`

This is a **relative statistical regime definition**, not an absolute crisis classification.

The key distinction is:

$$
\boxed{\text{Shock detection} \neq \text{Regime identification}}
$$

---

## Section 5 vs Section 7

### Section 5

The stress-day confusion matrix is an **in-sample, retrospective comparison** because both the percentile threshold and the volatility regime label are estimated from the full dataset.

It asks:

> Across the full historical sample, how well does a fast single-day extreme-return rule track the slower high-volatility regime?

It is not a real-time forecasting test.

### Section 7

The VaR backtest is genuinely **out-of-sample and walk-forward**:

1. Use only the preceding 250 observations.
2. Estimate that day's VaR.
3. Compare it with the next unseen return.
4. Record whether a violation occurred.
5. Move the window forward by one day.
6. Repeat.

This avoids look-ahead bias.

---

# Project Structure

```text
.
├── risk_utils.py        # Central statistical/risk engine
├── main.ipynb           # Complete written analysis and results
├── app.py               # Interactive Streamlit dashboard
├── build_notebook.py    # Reproducible notebook-generation helper
├── requirements.txt
└── README.md
```

`risk_utils.py` is the central calculation layer used by both the notebook and dashboard. The notebook provides the analytical narrative; the Streamlit app provides the interactive presentation.

---

# Dashboard

Run:

```bash
streamlit run app.py
```

The dashboard contains five tabs:

1. **Distribution**
2. **Low vs High Volatility**
3. **Beta**
4. **Error Trade-off**
5. **VaR & Backtest**

The dashboard allows the user to configure the market universe, date range, volatility window, VaR confidence level, stress-flag percentile, exposure, and rolling VaR estimation window.

---

# Notebook

Run:

```bash
jupyter notebook main.ipynb
```

The notebook contains the full analysis:

1. Data and log returns
2. Distribution analysis
3. Low/high-volatility regime analysis
4. Market-model regression
5. Stress-day Type I/Type II analysis
6. Historical and Normal Parametric VaR
7. Out-of-sample Kupiec backtesting
8. 99% tail analysis
9. Consolidated VaR validation summary
10. Conclusions

---

# Data

- **Market index:** NIFTY 50 (`^NSEI`)
- **Stocks:** `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`
- **Source:** Yahoo Finance through `yfinance`
- **Frequency:** Daily
- **Price basis:** Adjusted prices via `auto_adjust=True`
- **Period:** 2019-01-01 to 2024-12-31

The stock set provides a simple cross-section across large-cap Indian companies and different sector exposures:

- Reliance — conglomerate / energy
- TCS — information technology
- HDFC Bank — private banking
- Infosys — information technology

---

# Scope

The project deliberately uses **classical statistical methods**.

Included:

- Probability distributions
- Statistical inference
- Hypothesis testing
- Classical OLS regression
- Robust standard errors
- Volatility-regime analysis
- Value-at-Risk
- Out-of-sample backtesting
- Kupiec POF validation

Deliberately outside scope:

- Machine-learning models
- GARCH / stochastic-volatility models
- Monte Carlo simulation
- Copulas
- Advanced derivatives pricing
- Regulatory capital modelling

The goal is not to maximise the number of techniques. The goal is to demonstrate that each statistical method is connected to a meaningful market-risk question and can be interpreted correctly.

---

# Why This Project

Built as a **Quantitative Risk Modelling (QRM)** flagship project.

The central theme is:

$$
\boxed{\text{Risk is multidimensional}}
$$

because:

$$
\text{Return performance}
\neq
\text{Volatility}
\neq
\text{Systematic exposure}
\neq
\text{Tail risk}
$$

The project therefore combines:

$$
\text{Distribution analysis}
\rightarrow
\text{Regime analysis}
\rightarrow
\text{Systematic risk}
\rightarrow
\text{Stress detection}
\rightarrow
\text{VaR}
\rightarrow
\text{Out-of-sample validation}
$$

The final emphasis is not merely on producing a risk number, but on asking:

> **Can the risk model be trusted when it is tested against data it did not use to estimate itself?**

---

# Key Design Principle

```text
                   Market Data
                        │
                        ▼
                   Log Returns
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     Distribution   Low/High       Market Model
       Analysis     Volatility      Beta / R²
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                 Stress Detection
                        │
                        ▼
                       VaR
                        │
                        ▼
             Walk-Forward Backtest
                        │
                        ▼
                  Kupiec POF Test
                        │
                        ▼
                  Model Validation
```

The same statistical engine powers the notebook and dashboard:

```text
                 risk_utils.py
                  /        \
                 /          \
                ▼            ▼
          main.ipynb       app.py
          analytical      interactive
           analysis        dashboard
```

This centralisation keeps the statistical calculations consistent between the written analysis and the interactive application.
