"""
app.py
======
Streamlit dashboard for the Market Risk Statistical Validation project.

This file handles layout, visualization, and display. All statistical
calculations live in risk_utils.py so the notebook and dashboard use the
same analytical engine.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import streamlit as st

import risk_utils as ru


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Market Risk Statistical Validation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD CUSTOM CSS
# ============================================================

try:
    with open("style.css", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )
except FileNotFoundError:
    st.warning(
        "style.css was not found. The dashboard will use the default "
        "Streamlit styling."
    )


# ============================================================
# VISUAL DESIGN REGISTER
# ============================================================

CHART_BG = "#fffdf9"
TEXT = "#2b211b"
MUTED = "#786b61"
GRID = "#e7dfd7"

ACCENT = "#6f4e37"
ACCENT_DARK = "#513724"
SECONDARY = "#8b7564"
RISK = "#9a4a3f"


def style_chart(ax):
    """Apply the common visual language to all Matplotlib charts."""

    ax.set_facecolor(CHART_BG)

    # Remove unnecessary chart borders.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#cfc4b9")
    ax.spines["bottom"].set_color("#cfc4b9")

    # Minimal ticks.
    ax.tick_params(
        colors=MUTED,
        labelsize=9,
        length=0,
    )

    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

    # Light horizontal grid for quantitative reading.
    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_axisbelow(True)

    # Title styling.
    ax.title.set_color(TEXT)
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("600")

    # Legend styling.
    legend = ax.get_legend()

    if legend is not None:
        legend.get_frame().set_facecolor(CHART_BG)
        legend.get_frame().set_edgecolor("#ddd3ca")
        legend.get_frame().set_alpha(0.95)

        for text in legend.get_texts():
            text.set_color(TEXT)
            text.set_fontsize(8.5)


def format_pvalue(p):
    """Display very small p-values without misleading rounding."""

    if pd.isna(p):
        return "N/A"

    if p == 0:
        return "<1e-300"

    if p < 0.0001:
        return f"{p:.2e}"

    return f"{p:.4f}"


def verdict_pill(passed: bool, pass_text: str, fail_text: str) -> str:
    """Small HTML pill for plain-English statistical verdicts."""

    cls, label = (
        ("pass", pass_text)
        if passed
        else ("fail", fail_text)
    )

    return f'<span class="verdict {cls}">{label}</span>'


def section_note(text: str):
    """Display plain-English framing at the top of each tab."""

    st.markdown(
        f'<div class="section-note">{text}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.title("Market Risk Statistical Validation")

st.markdown(
    '<p class="app-subtitle">'
    "Statistical analysis of return distributions, volatility regimes, "
    "systematic market exposure, stress detection, and VaR model calibration."
    "</p>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — CONFIGURATION
# ============================================================

st.sidebar.header("Configuration")

market_ticker = st.sidebar.text_input(
    "Market index ticker",
    "^NSEI",
)

stock_input = st.sidebar.text_input(
    "Stock tickers (comma-separated)",
    "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS",
)

stock_tickers = [
    ticker.strip()
    for ticker in stock_input.split(",")
    if ticker.strip()
]

start_date = st.sidebar.date_input(
    "Start date",
    pd.to_datetime("2019-01-01"),
)

end_date = st.sidebar.date_input(
    "End date",
    pd.to_datetime("2024-12-31"),
)

st.sidebar.markdown("---")

regime_window = st.sidebar.slider(
    "Rolling volatility window (days)",
    10,
    60,
    21,
)

var_alpha_pct = st.sidebar.select_slider(
    "VaR confidence level",
    options=[90, 95, 99],
    value=95,
)

flag_pct = st.sidebar.select_slider(
    "Stress-flag percentile",
    options=[1, 5, 10],
    value=5,
)

run = st.sidebar.button(
    "Run analysis",
    type="primary",
    use_container_width=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if run:
    st.session_state.analysis_started = True


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    """Download and cache price data."""

    return ru.download_prices(
        tickers,
        start,
        end,
    )


if not st.session_state.analysis_started:
    st.info(
        "Set your configuration in the sidebar and click "
        "**Run analysis** to begin."
    )
    st.stop()


if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()


with st.spinner("Downloading price data..."):

    all_tickers = [market_ticker] + stock_tickers

    try:
        prices = load_data(
            all_tickers,
            str(start_date),
            str(end_date),
        )

        prices = prices.rename(
            columns={market_ticker: "MARKET"}
        )

    except Exception as e:
        st.error(
            "Data download failed — check tickers and internet connection. "
            f"Details: {e}"
        )
        st.stop()


if "MARKET" not in prices.columns:
    st.error(
        "The selected market ticker could not be loaded. "
        "Check the market index ticker."
    )
    st.stop()


returns = ru.log_returns(prices)
market_returns = returns["MARKET"]


if market_returns.empty:
    st.error("No market return observations were available.")
    st.stop()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1 · Distribution",
        "2 · Low vs High Volatility",
        "3 · Beta",
        "4 · Error Trade-off",
        "5 · VaR & Backtest",
    ]
)


# ============================================================
# TAB 1 — DISTRIBUTION
# ============================================================

with tab1:

    section_note(
        "Do returns follow a Normal distribution, or are extreme moves more "
        "common than a Normal model would predict? This matters because the "
        "Normal Parametric VaR analysis later in the framework assumes normality."
    )

    normal_params = ru.fit_normal(market_returns)
    t_params = ru.fit_student_t(market_returns)
    jb = ru.goodness_of_fit_normal(market_returns)

    c1, c2 = st.columns([1.3, 1])

    with c1:

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10.5, 3.8),
            facecolor=CHART_BG,
        )

        for ax in axes:
            style_chart(ax)

        x = np.linspace(
            market_returns.min(),
            market_returns.max(),
            500,
        )

        # ----------------------------------------------------
        # Empirical distribution vs fitted distributions
        # ----------------------------------------------------

        axes[0].hist(
            market_returns,
            bins=60,
            density=True,
            alpha=0.42,
            color="#c9d8e6",
            edgecolor="none",
            label="Empirical",
        )

        axes[0].plot(
            x,
            stats.norm.pdf(x, **normal_params),
            color=ACCENT,
            linewidth=2.0,
            label="Normal fit",
        )

        axes[0].plot(
            x,
            stats.t.pdf(x, **t_params),
            color=SECONDARY,
            linewidth=2.0,
            label="Student-t fit",
        )

        axes[0].set_title(
            "Empirical vs fitted distributions",
            loc="left",
        )

        axes[0].set_xlabel("Daily log return")
        axes[0].set_ylabel("Density")

        axes[0].legend()

        # ----------------------------------------------------
        # Q-Q plot vs Normal
        # ----------------------------------------------------

        stats.probplot(
            market_returns,
            dist="norm",
            plot=axes[1],
        )

        axes[1].set_title(
            "Normal Q-Q diagnostic",
            loc="left",
        )

        axes[1].set_xlabel("Theoretical quantiles")
        axes[1].set_ylabel("Ordered values")

        # Style scipy-generated Q-Q elements.
        if len(axes[1].lines) >= 1:
            axes[1].lines[0].set_color(ACCENT)
            axes[1].lines[0].set_linewidth(1.8)

        if len(axes[1].lines) >= 2:
            axes[1].lines[1].set_color(SECONDARY)
            axes[1].lines[1].set_markersize(3.5)
            axes[1].lines[1].set_markeredgewidth(0)

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

    with c2:

        st.metric(
            "Jarque-Bera p-value (market)",
            format_pvalue(jb["p_value"]),
        )

        st.markdown(
            "Reject normality at 5%: "
            + verdict_pill(
                jb["reject_normality_at_5pct"],
                "Yes — fat-tailed",
                "No — looks Normal",
            ),
            unsafe_allow_html=True,
        )

        st.metric(
            "Student-t fit — degrees of freedom",
            f"{t_params['df']:.2f}",
        )

        st.caption(
            "Lower degrees of freedom indicate heavier tails relative "
            "to a Normal distribution."
        )

    st.markdown("##### Descriptive statistics — all series")

    descriptive_display = returns.apply(
        ru.descriptive_stats
    ).T

    st.dataframe(
        descriptive_display,
        use_container_width=True,
    )


# ============================================================
# TAB 2 — LOW VS HIGH VOLATILITY
# ============================================================

with tab2:

    section_note(
        "Days are classified into 'low-volatility' or 'high-volatility' "
        "regimes using trailing realised volatility. The median split is a "
        "relative volatility-regime definition, not a claim that half the "
        "trading days were crisis or financially stressed days. The analysis "
        "asks whether return behaviour differs across these volatility environments."
    )

    regime, rolling_vol = ru.classify_regime(
        market_returns,
        window=regime_window,
        method="median",
    )

    result = ru.compare_regimes(
        market_returns,
        regime,
    )

    fig, ax = plt.subplots(
        figsize=(10, 3.4),
        facecolor=CHART_BG,
    )

    style_chart(ax)

    rolling_vol.plot(
        ax=ax,
        color=ACCENT,
        linewidth=1.25,
        label=f"{regime_window}-day rolling volatility",
    )

    ax.axhline(
        rolling_vol.median(),
        color=RISK,
        linestyle="--",
        linewidth=1.25,
        label="Median low/high-vol cutoff",
    )

    ax.set_title(
        "Trailing realised volatility",
        loc="left",
    )

    ax.set_ylabel("Rolling volatility")
    ax.set_xlabel("Date")

    ax.legend()

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Welch t-test p-value (mean)",
        format_pvalue(result["t_test_p"]),
    )

    c2.metric(
        "Mann-Whitney p-value (distribution)",
        format_pvalue(result["mannwhitney_p"]),
    )

    c3.metric(
        "Levene p-value (variance)",
        format_pvalue(result["levene_p"]),
    )

    st.markdown(
        "**Reading this:** returns are non-Normal (Tab 1), so the "
        "Mann-Whitney U test provides the primary non-parametric comparison "
        "of the low- and high-volatility return samples. The Welch t-test "
        "provides a complementary test of whether their mean returns differ "
        "while allowing unequal variances. "
        f"95% CI for mean difference "
        f"(low-vol − high-vol): "
        f"({result['mean_diff_95ci'][0]:.5f}, "
        f"{result['mean_diff_95ci'][1]:.5f})."
    )

    st.markdown(
        "Variance differs sharply between regimes: "
        + verdict_pill(
            result["levene_p"] < 0.05,
            "Confirmed",
            "Not confirmed",
        ),
        unsafe_allow_html=True,
    )

    regime_counts = regime.value_counts()

    c1, c2 = st.columns(2)

    c1.metric(
        "Low-vol observations",
        int(regime_counts.get("low_vol", 0)),
    )

    c2.metric(
        "High-vol observations",
        int(regime_counts.get("high_vol", 0)),
    )


# ============================================================
# TAB 3 — BETA
# ============================================================

with tab3:

    section_note(
        "Classical market-model OLS regression: "
        "R_stock = alpha + beta · R_market + error. "
        "Beta measures systematic sensitivity to NIFTY 50 movements, "
        "while R² measures how much of the stock's return variation "
        "is explained by the market factor."
    )

    rows = []

    for ticker in stock_tickers:

        if ticker not in returns.columns:
            continue

        res = ru.market_model_regression(
            returns[ticker],
            market_returns,
        )

        rows.append(
            {
                "stock": ticker,
                "alpha": res["alpha"],
                "alpha_p": res["alpha_p"],
                "beta": res["beta"],
                "beta_p": res["beta_p"],
                "beta_ci_low": res["beta_ci95"][0],
                "beta_ci_high": res["beta_ci95"][1],
                "r_squared": res["r_squared"],
            }
        )

    if not rows:
        st.warning(
            "No stock series were available for market-model regression."
        )
    else:

        reg_table = pd.DataFrame(rows).set_index("stock")

        c1, c2 = st.columns([1, 1.2])

        with c1:

            fig, ax = plt.subplots(
                figsize=(6.2, 3.8),
                facecolor=CHART_BG,
            )

            style_chart(ax)

            beta = reg_table["beta"]

            beta_err = np.vstack(
                [
                    beta - reg_table["beta_ci_low"],
                    reg_table["beta_ci_high"] - beta,
                ]
            )

            ax.bar(
                reg_table.index,
                beta,
                yerr=beta_err,
                capsize=4,
                width=0.58,
                color=ACCENT,
                edgecolor="none",
                error_kw={
                    "elinewidth": 1.2,
                    "ecolor": TEXT,
                    "capthick": 1.2,
                },
            )

            ax.axhline(
                1.0,
                color=RISK,
                linestyle="--",
                linewidth=1.3,
                label="Market beta = 1",
            )

            ax.set_title(
                "Systematic market exposure",
                loc="left",
            )

            ax.set_ylabel("Beta")

            ax.legend()

            plt.xticks(
                rotation=20,
                ha="right",
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True,
            )

        with c2:

            reg_display = reg_table.copy()

            for col in [
                "alpha",
                "beta",
                "beta_ci_low",
                "beta_ci_high",
                "r_squared",
            ]:

                if col in reg_display.columns:
                    reg_display[col] = reg_display[col].map(
                        lambda x: f"{x:.4f}"
                    )

            for col in [
                "alpha_p",
                "beta_p",
            ]:

                if col in reg_display.columns:
                    reg_display[col] = reg_display[col].map(
                        format_pvalue
                    )

            st.dataframe(
                reg_display,
                use_container_width=True,
            )

        st.caption(
            "Inference uses HC1 heteroskedasticity-robust standard errors "
            "because daily equity returns exhibit volatility clustering. "
            "The OLS point estimates of alpha and beta are unchanged; "
            "robust inference adjusts their estimated uncertainty. A 95% "
            "beta CI entirely below or above 1 indicates evidence that "
            "systematic market sensitivity differs from the benchmark."
        )


# ============================================================
# TAB 4 — ERROR TRADE-OFF
# ============================================================

with tab4:

    section_note(
        "A fast, reactive rule flags a day as predicted stress if its return "
        "falls among the worst X% of daily returns. This tab compares that "
        "single-day shock flag against the slower high-volatility regime from "
        "Tab 2 and shows the trade-off between false alarms and missed "
        "high-volatility observations."
    )

    flagged, threshold = ru.flag_stress_days(
        market_returns,
        percentile=flag_pct,
    )

    actual = (
        regime.reindex(flagged.index) == "high_vol"
    ).fillna(False)

    metrics = ru.confusion_matrix_metrics(
        actual,
        flagged,
    )

    cm_df = pd.DataFrame(
        [
            [metrics["TP"], metrics["FN"]],
            [metrics["FP"], metrics["TN"]],
        ],
        index=[
            "Flagged",
            "Not flagged",
        ],
        columns=[
            "Reference: High-vol",
            "Reference: Low-vol",
        ],
    )

    c1, c2 = st.columns([1, 1.3])

    with c1:

        st.markdown(
            "##### Fast stress flag vs high-volatility regime"
        )

        st.dataframe(
            cm_df,
            use_container_width=True,
        )

        st.caption(
            f"Fast rule: flagging the worst {flag_pct}% of daily returns "
            f"(return threshold: {threshold:.4%})."
        )

    with c2:

        m1, m2 = st.columns(2)

        m1.metric(
            "Recall (high-vol captured)",
            f"{metrics['recall_sensitivity']:.1%}",
        )

        m2.metric(
            "Specificity (low-vol not flagged)",
            f"{metrics['specificity']:.1%}",
        )

        m3, m4 = st.columns(2)

        m3.metric(
            "Type I error (false alarms)",
            f"{metrics['type_I_error_rate_FPR']:.1%}",
        )

        m4.metric(
            "Type II error (missed high-vol)",
            f"{metrics['type_II_error_rate_FNR']:.1%}",
        )

    st.caption(
        "The fast rule detects single-day return shocks, while the reference "
        "state is the slower 21-day high-volatility regime. A stricter threshold "
        "(lower percentile) produces fewer alerts, typically reducing false "
        "alarms but increasing the risk of missing observations that belong to "
        "the high-volatility regime. This is a retrospective, in-sample "
        "risk-management comparison rather than a live early-warning test."
    )


# ============================================================
# TAB 5 — VAR & BACKTEST
# ============================================================

with tab5:

    section_note(
        "Value-at-Risk (VaR) estimates a loss threshold at a chosen confidence "
        "level. This tab compares Historical VaR with Normal Parametric VaR, "
        "translates the estimate into a rupee figure, and evaluates out-of-sample "
        "violation-rate calibration using a walk-forward backtest."
    )

    alpha = 1 - var_alpha_pct / 100

    hvar = ru.historical_var(
        market_returns,
        alpha,
    )

    pvar = ru.parametric_var(
        market_returns,
        alpha,
    )

    c1, c2 = st.columns(2)

    c1.metric(
        f"{var_alpha_pct}% Historical VaR (daily)",
        f"{hvar:.4%}",
    )

    c2.metric(
        f"{var_alpha_pct}% Normal Parametric VaR (daily)",
        f"{pvar:.4%}",
    )

    # --------------------------------------------------------
    # Reference comparison
    # --------------------------------------------------------

    if pvar > hvar:

        st.markdown(
            "At this confidence level, "
            + verdict_pill(
                True,
                "Normal VaR is more conservative",
                "Historical VaR is more conservative",
            ),
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            "At this confidence level, "
            + verdict_pill(
                True,
                "Historical VaR is more conservative",
                "Normal VaR is more conservative",
            ),
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Translate to monetary loss
    # --------------------------------------------------------

    st.markdown("##### Translate into an estimated loss")

    exposure = st.number_input(
        "Illustrative total exposure (Rs)",
        min_value=0.0,
        value=4_00_00_000.0,
        step=10_00_000.0,
        format="%.0f",
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Est. 1-day loss — Historical",
        f"Rs {ru.var_to_exposure_loss(hvar, exposure):,.0f}",
    )

    c2.metric(
        "Est. 1-day loss — Normal Parametric",
        f"Rs {ru.var_to_exposure_loss(pvar, exposure):,.0f}",
    )

    # --------------------------------------------------------
    # Walk-forward backtest
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "##### Out-of-sample backtest (no look-ahead bias)"
    )

    st.caption(
        "Each day's VaR is estimated using only the preceding rolling window, "
        "then tested only against the next unseen day. This is a walk-forward "
        "out-of-sample evaluation, not a full-sample self-test."
    )

    window = st.slider(
        "Rolling estimation window (trading days)",
        100,
        500,
        250,
        step=10,
    )

    with st.spinner("Running walk-forward backtest..."):

        bt_hist = ru.rolling_out_of_sample_var(
            market_returns,
            window=window,
            alpha=alpha,
            method="historical",
        )

        bt_para = ru.rolling_out_of_sample_var(
            market_returns,
            window=window,
            alpha=alpha,
            method="parametric",
        )

    # --------------------------------------------------------
    # Rolling VaR chart
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 3.8),
        facecolor=CHART_BG,
    )

    style_chart(ax)

    bt_hist["actual_return"].plot(
        ax=ax,
        color="#9fb8c9",
        alpha=0.60,
        linewidth=0.7,
        label="Realised return",
    )

    (-bt_hist["var_estimate"]).plot(
        ax=ax,
        color=ACCENT,
        linewidth=1.7,
        label="Rolling Historical VaR",
    )

    (-bt_para["var_estimate"]).plot(
        ax=ax,
        color=SECONDARY,
        linewidth=1.7,
        label="Rolling Normal VaR",
    )

    ax.axhline(
        0,
        color="#cfc4b9",
        linewidth=0.8,
    )

    ax.set_title(
        f"Walk-forward {var_alpha_pct}% VaR vs realised returns",
        loc="left",
    )

    ax.set_ylabel("Return / VaR")
    ax.set_xlabel("Date")

    ax.legend()

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Violation summary
    # --------------------------------------------------------

    st.markdown(
        "##### Out-of-sample violation summary"
    )

    summary_data = []

    for label, bt in [
        ("Historical VaR", bt_hist),
        ("Normal Parametric VaR", bt_para),
    ]:

        violations = int(
            bt["violation"].sum()
        )

        n_obs = len(bt)

        expected = alpha * n_obs
        observed_rate = (
            violations / n_obs
            if n_obs
            else np.nan
        )

        summary_data.append(
            {
                "Method": label,
                "Violations": violations,
                "Expected": expected,
                "Observed rate": observed_rate,
                "Expected rate": alpha,
            }
        )

    summary_display = pd.DataFrame(
        summary_data
    )

    summary_display["Expected"] = (
        summary_display["Expected"]
        .map(lambda x: f"{x:.2f}")
    )

    summary_display["Observed rate"] = (
        summary_display["Observed rate"]
        .map(lambda x: f"{x:.2%}")
    )

    summary_display["Expected rate"] = (
        summary_display["Expected rate"]
        .map(lambda x: f"{x:.2%}")
    )

    st.dataframe(
        summary_display,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Kupiec POF
    # --------------------------------------------------------

    st.markdown(
        "##### Kupiec Proportion-of-Failures test"
    )

    st.caption(
        "A p-value below 0.05 means the observed violation frequency is "
        "statistically inconsistent with the model's stated confidence level."
    )

    for label, bt in [
        ("Historical VaR", bt_hist),
        ("Normal Parametric VaR", bt_para),
    ]:

        x = int(
            bt["violation"].sum()
        )

        T = len(bt)

        kupiec = ru.kupiec_pof_test(
            x,
            T,
            alpha=alpha,
        )

        c1, c2 = st.columns([1, 3])

        with c1:

            st.markdown(
                f"**{label}**"
            )

            st.caption(
                f"{T} out-of-sample days"
            )

            st.markdown(
                verdict_pill(
                    not kupiec["reject_model_at_5pct"],
                    "Not rejected by Kupiec",
                    "Miscalibration detected",
                ),
                unsafe_allow_html=True,
            )

        with c2:

            kupiec_display = pd.DataFrame(
                [
                    {
                        "Observed violations": kupiec[
                            "observed_violations"
                        ],
                        "Expected violations": kupiec[
                            "expected_violations"
                        ],
                        "Observed rate": (
                            kupiec["observed_rate"]
                        ),
                        "Expected rate": (
                            kupiec["expected_rate"]
                        ),
                        "Kupiec LR": kupiec["LR_stat"],
                        "p-value": kupiec["p_value"],
                    }
                ]
            )

            kupiec_display["Observed rate"] = (
                kupiec_display["Observed rate"]
                .map(lambda x: f"{x:.2%}")
            )

            kupiec_display["Expected rate"] = (
                kupiec_display["Expected rate"]
                .map(lambda x: f"{x:.2%}")
            )

            kupiec_display["Kupiec LR"] = (
                kupiec_display["Kupiec LR"]
                .map(lambda x: f"{x:.4f}")
            )

            kupiec_display["p-value"] = (
                kupiec_display["p-value"]
                .map(format_pvalue)
            )

            st.dataframe(
                kupiec_display,
                use_container_width=True,
            )
