"""
Corn & Soybean Demand Monitor — Streamlit dashboard.

Run with:
    streamlit run demand_monitor/app.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from data import load_crop_data, load_real_prices
from model import (
    BASE_YEAR,
    CORN_ELASTICITY,
    CORN_OLS_INTERCEPT,
    CORN_OLS_SLOPE,
    SOYBEAN_ELASTICITY,
    SOYBEAN_OLS_INTERCEPT,
    SOYBEAN_OLS_SLOPE,
    compute_g,
)
from shapley import run_shapley, verify_decomp

# ---------------------------------------------------------------------------
# AEI brand palette
# ---------------------------------------------------------------------------
AEI = {
    "green":      "#609D42",   # primary brand green
    "navy":       "#001425",   # navy
    "orange":     "#F28C28",   # burnt orange
    "gray":       "#A7A9AC",   # light gray
    "teal":       "#3B8B8B",   # teal
    "red":        "#C0504D",   # soft red
    "dark_green": "#508428",   # deeper green
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AEI Corn & Soybean Demand Monitor",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      [data-testid="stAppViewContainer"] > .main {{ padding-top: 1rem; }}
      header[data-testid="stHeader"] {{
          background: {AEI["navy"]};
          height: 4px;
      }}
      .aei-title {{
          color: {AEI["navy"]};
          font-size: 1.7rem;
          font-weight: 700;
          margin-bottom: 0;
      }}
      .aei-sub {{
          color: {AEI["gray"]};
          font-size: 0.85rem;
          margin-top: 0;
      }}
      .scenario-box {{
          background: rgba(96,157,66,0.09);
          border-left: 4px solid {AEI["green"]};
          border-radius: 4px;
          padding: 0.75rem 1rem;
          margin-bottom: 0.5rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<p class="aei-title">🌽 Corn & Soybean Demand Monitor</p>'
    f'<p class="aei-sub">Ag Economic Insights &nbsp;|&nbsp; '
    f'What USDA supply &amp; usage data say about corn and soybean prices &nbsp;|&nbsp; '
    f'Prices adjusted to 2025 dollars</p>',
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_all(crop: str, elasticity: float, b0: float, b1: float):
    df          = load_crop_data(crop)
    real_prices = load_real_prices(crop)
    df["price_real"] = df["year"].map(real_prices)
    results     = run_shapley(df, elasticity, b0, b1, BASE_YEAR)
    return df, results


corn_df, corn_res = load_all("corn",     CORN_ELASTICITY,    CORN_OLS_INTERCEPT,    CORN_OLS_SLOPE)
soy_df,  soy_res  = load_all("soybeans", SOYBEAN_ELASTICITY, SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE)


# ---------------------------------------------------------------------------
# Synced slider + number input helper
# ---------------------------------------------------------------------------

def _synced_input(
    label: str,
    ss_key: str,
    min_val: float,
    max_val: float,
    step: float,
    init_val: float,
    fmt: str = "%.0f",
) -> float:
    """
    Render a paired slider and number_input that stay in sync via session_state.

    Moving the slider updates the text box; typing in the text box updates the
    slider position. Returns the current value.
    """
    if ss_key not in st.session_state:
        st.session_state[ss_key] = float(init_val)

    _sldr = f"_sldr_{ss_key}"
    _num  = f"_num_{ss_key}"

    def _from_slider():
        st.session_state[ss_key] = float(st.session_state[_sldr])
        # Delete the number_input state so it reinitializes from value= on next render
        st.session_state.pop(_num, None)

    def _from_input():
        val = float(st.session_state[_num])
        st.session_state[ss_key]  = val
        st.session_state[_sldr]   = int(val)

    st.markdown(f"**{label}**")
    st.slider(
        "", min_value=int(min_val), max_value=int(max_val),
        value=int(st.session_state[ss_key]), step=int(step),
        key=_sldr, on_change=_from_slider, label_visibility="collapsed",
    )
    st.number_input(
        "", min_value=float(min_val), max_value=float(max_val),
        value=float(st.session_state[ss_key]), step=float(step), format=fmt,
        key=_num, on_change=_from_input, label_visibility="collapsed",
    )
    return float(st.session_state[ss_key])


# ---------------------------------------------------------------------------
# Sidebar — Scenario / "What-If" Tool
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔧 What-If Scenario")
    st.caption(
        "Enter your own supply and usage numbers to see "
        "what price the model implies."
    )

    scen_crop = st.radio("Crop", ["Corn", "Soybeans"], horizontal=True)

    if scen_crop == "Corn":
        ref       = corn_res.iloc[-1]
        e         = CORN_ELASTICITY
        b0, b1    = CORN_OLS_INTERCEPT, CORN_OLS_SLOPE
        df_full   = corn_df
        res_full  = corn_res
        # Slider ranges: ±20% of historical min/max, rounded to nearest 100
        _all_s = corn_df["supply"].dropna()
        _all_u = corn_df["usage"].dropna()
    else:
        ref       = soy_res.iloc[-1]
        e         = SOYBEAN_ELASTICITY
        b0, b1    = SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE
        df_full   = soy_df
        res_full  = soy_res
        _all_s = soy_df["supply"].dropna()
        _all_u = soy_df["usage"].dropna()

    s_lo = math.floor(_all_s.min() * 0.80 / 100) * 100
    s_hi = math.ceil( _all_s.max() * 1.20 / 100) * 100
    u_lo = math.floor(_all_u.min() * 0.80 / 100) * 100
    u_hi = math.ceil( _all_u.max() * 1.20 / 100) * 100

    default_s = float(round(ref["supply"]))
    default_u = float(round(ref["usage"]))

    # Use crop-prefixed keys so Corn and Soybeans remember values independently
    scen_supply = _synced_input(
        "Total Supply (mil. bu)",
        f"supply_{scen_crop}", s_lo, s_hi, 50, default_s,
    )
    scen_usage = _synced_input(
        "Total Usage (mil. bu)",
        f"usage_{scen_crop}", u_lo, u_hi, 50, default_u,
    )

    # Ratio-method price prediction (the active model method)
    G_scen     = compute_g(scen_supply, scen_usage, e)
    base_row   = df_full[df_full["year"] == BASE_YEAR].iloc[0]
    G_base_val = compute_g(float(base_row["supply"]), float(base_row["usage"]), e)
    P_base_val = float(df_full[df_full["year"] == BASE_YEAR]["price_real"].values[0])
    ratio_pred = P_base_val * (G_scen / G_base_val)

    st.markdown("---")
    st.markdown("**Model-implied price**")
    st.markdown(
        f'<div class="scenario-box">'
        f'<span style="font-size:1.4rem; font-weight:700; color:{AEI["navy"]}">'
        f"${ratio_pred:.2f}<small>/bu</small></span><br>"
        f'<span style="color:{AEI["gray"]}; font-size:0.78rem;">'
        f"G index: {G_scen:.4f}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    last_actual     = float(ref["price_real"])
    delta_vs_actual = ratio_pred - last_actual
    sign  = "+" if delta_vs_actual >= 0 else ""
    color = AEI["green"] if delta_vs_actual >= 0 else AEI["red"]
    st.markdown(
        f'<span style="color:{color}; font-weight:600;">'
        f"{sign}${delta_vs_actual:.2f} vs last actual ({int(ref['year'])})"
        f"</span>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        f"Model: P = P₂₀₀₉ × (G / G₂₀₀₉) &nbsp;|&nbsp; "
        f"Base year: {BASE_YEAR} &nbsp;|&nbsp; Real 2025 $/bu"
    )

# Auto-switch the main content tab to match the sidebar crop selection
_tab_idx = 0 if scen_crop == "Corn" else 1
components.html(
    f"""
    <script>
    (function() {{
        var attempt = 0;
        function clickTab() {{
            var tabs = window.parent.document.querySelectorAll('button[role="tab"]');
            if (tabs.length > {_tab_idx}) {{
                tabs[{_tab_idx}].click();
            }} else if (attempt < 15) {{
                attempt++;
                setTimeout(clickTab, 100);
            }}
        }}
        clickTab();
    }})();
    </script>
    """,
    height=0,
)


# ---------------------------------------------------------------------------
# Scenario row computation (Shapley for the hypothetical year)
# ---------------------------------------------------------------------------

def compute_scenario_row(
    results: pd.DataFrame,
    scenario_supply: float,
    scenario_usage: float,
    scenario_price: float,
    elasticity: float,
    b0_: float,
    b1_: float,
) -> dict:
    """
    Compute supply/usage Shapley impacts for the hypothetical scenario year,
    using the same logic as shapley.py.  Returns a dict with farmer_table keys.
    """
    prev   = results.iloc[-1]
    S_tm1  = float(prev["supply"])
    U_tm1  = float(prev["usage"])
    P_prev = float(prev["price_real"])
    y_tm1  = float(prev["pred_price"])

    # K constant: normalises IV to 100 in the base year
    base_rows = results[results["year"] == BASE_YEAR]
    base_r    = base_rows.iloc[0] if not base_rows.empty else results.iloc[0]
    K         = 100.0 / (float(base_r["G"]) * float(base_r["price_real"]))

    G_t  = compute_g(scenario_supply, scenario_usage, elasticity)
    y_t  = b0_ + b1_ * K * scenario_price * G_t

    GD   = compute_g(S_tm1,           scenario_usage, elasticity)
    GS   = compute_g(scenario_supply, U_tm1,          elasticity)
    yD   = b0_ + b1_ * K * scenario_price * GD
    yS   = b0_ + b1_ * K * scenario_price * GS

    dy        = y_t - y_tm1
    dPrice    = scenario_price - P_prev
    resid     = dPrice - dy
    d_shap    = 0.5 * ((yD - y_tm1) + (y_t - yS))
    s_shap    = 0.5 * ((yS - y_tm1) + (y_t - yD))
    abs_total = abs(d_shap) + abs(s_shap)

    if abs_total > 1e-12:
        demand_impact = d_shap + resid * abs(d_shap) / abs_total
        supply_impact = s_shap + resid * abs(s_shap) / abs_total
    else:
        demand_impact = resid / 2.0
        supply_impact = resid / 2.0

    scen_year = int(prev["year"]) + 1
    return {
        "year_label":                  f"{scen_year} (Scenario)",
        "Actual Price ($/bu)":         round(scenario_price, 2),
        "Model Price ($/bu)":          round(scenario_price, 2),
        "Change vs Prior Year ($/bu)": round(dPrice, 2),
        "Supply Drove ($/bu)":         round(supply_impact, 2),
        "Usage Drove ($/bu)":          round(demand_impact, 2),
    }


# Pre-compute scenario rows for whichever crop the sidebar is set to
if scen_crop == "Corn":
    _corn_scen_row = compute_scenario_row(
        corn_res, scen_supply, scen_usage, ratio_pred,
        CORN_ELASTICITY, CORN_OLS_INTERCEPT, CORN_OLS_SLOPE,
    )
    _soy_scen_row = None
else:
    _corn_scen_row = None
    _soy_scen_row = compute_scenario_row(
        soy_res, scen_supply, scen_usage, ratio_pred,
        SOYBEAN_ELASTICITY, SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE,
    )


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def fmt2(x):
    return f"${x:.2f}"


def price_chart(results: pd.DataFrame) -> go.Figure:
    """Time series: actual vs model-predicted price."""
    fig = go.Figure()
    fig.add_scatter(
        x=results["year"], y=results["price_real"].round(2),
        mode="lines+markers", name="Actual",
        line=dict(color=AEI["green"], width=2.5),
        marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>Actual: $%{y:.2f}/bu<extra></extra>",
    )
    fig.add_scatter(
        x=results["year"], y=results["pred_price"].round(2),
        mode="lines+markers", name="Model",
        line=dict(color=AEI["teal"], width=2, dash="dash"),
        marker=dict(size=5),
        hovertemplate="<b>%{x}</b><br>Model: $%{y:.2f}/bu<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="Marketing Year",
        yaxis_title="Real Price (2025 $/bu)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
        height=360,
        margin=dict(t=10, b=40, l=60, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, linecolor=AEI["gray"]),
        yaxis=dict(gridcolor="#EBEBEB", linecolor=AEI["gray"]),
    )
    return fig


def shapley_chart(results: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart decomposing the year-over-year price change.
    Supply bars: navy. Usage bars: green. Total-change diamonds: orange.
    """
    fig = go.Figure()
    fig.add_bar(
        x=results["year"], y=results["supply_impact"].round(2),
        name="Supply effect",
        marker_color=AEI["navy"],
        hovertemplate="<b>%{x}</b><br>Supply: %{y:+.2f}/bu<extra></extra>",
    )
    fig.add_bar(
        x=results["year"], y=results["demand_impact"].round(2),
        name="Usage effect",
        marker_color=AEI["green"],
        hovertemplate="<b>%{x}</b><br>Usage: %{y:+.2f}/bu<extra></extra>",
    )
    fig.add_scatter(
        x=results["year"], y=results["dPrice"].round(2),
        mode="markers", name="Total change",
        marker=dict(
            color=AEI["orange"], size=8, symbol="diamond",
            line=dict(width=1, color=AEI["navy"]),
        ),
        hovertemplate="<b>%{x}</b><br>Total: %{y:+.2f}/bu<extra></extra>",
    )
    fig.add_hline(y=0, line_width=1, line_color=AEI["gray"])
    fig.update_layout(
        barmode="relative",
        xaxis_title="Marketing Year",
        yaxis_title="Change in Real Price (2025 $/bu)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
        height=360,
        margin=dict(t=10, b=40, l=60, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, linecolor=AEI["gray"]),
        yaxis=dict(gridcolor="#EBEBEB", linecolor=AEI["gray"], zeroline=False),
    )
    return fig


def scenario_ellipse_chart(
    results: pd.DataFrame,
    scenario_supply: float,
    scenario_usage: float,
    scenario_price: float,
    elasticity: float,
) -> go.Figure:
    """
    Supply vs Usage scatter with:
      - Iso-price contour lines (dotted) — each line is a (S, U) locus that
        implies the same model price, derived from S/U = r_base*(P/P_base)^ε.
      - 1σ / 2σ confidence ellipses from the covariance of historical supply
        and usage (eigenvector decomposition).
      - Historical data points labeled by year.
      - An orange ★ at the subscriber's scenario.
    """
    hist        = results.dropna(subset=["supply", "usage", "price_real"])
    supply_vals = hist["supply"].values
    usage_vals  = hist["usage"].values
    years       = hist["year"].astype(int).astype(str).values

    mean_s = np.mean(supply_vals)
    mean_u = np.mean(usage_vals)
    cov    = np.cov(supply_vals, usage_vals)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    theta  = np.linspace(0, 2 * np.pi, 300)
    circle = np.array([np.cos(theta), np.sin(theta)])

    def make_ellipse(n_sigma):
        scaled = eigenvectors @ np.diag(n_sigma * np.sqrt(eigenvalues)) @ circle
        return scaled[0] + mean_s, scaled[1] + mean_u

    x_2sig, y_2sig = make_ellipse(2)
    x_1sig, y_1sig = make_ellipse(1)

    # Iso-price lines: U = S / r_target, where r_target = r_base*(P/P_base)^ε
    base_rows = hist[hist["year"] == BASE_YEAR]
    base_r    = base_rows.iloc[0] if not base_rows.empty else hist.iloc[0]
    r_base    = float(base_r["supply"]) / float(base_r["usage"])
    P_base    = float(base_r["price_real"])

    prices_hist = hist["price_real"].values
    p_lo = np.floor(prices_hist.min() * 2) / 2
    p_hi = np.ceil( prices_hist.max() * 2) / 2
    price_levels = np.linspace(p_lo, p_hi, 6)

    # Tight axis bounds — just enough padding around data + scenario point
    all_s = np.append(supply_vals, scenario_supply)
    all_u = np.append(usage_vals,  scenario_usage)
    s_pad = (supply_vals.max() - supply_vals.min()) * 0.04
    u_pad = (usage_vals.max()  - usage_vals.min())  * 0.04
    x_min = all_s.min() - s_pad
    x_max = all_s.max() + s_pad
    y_min = all_u.min() - u_pad
    y_max = all_u.max() + u_pad

    s_line = np.array([x_min, x_max])

    fig = go.Figure()

    # Iso-price lines (background layer)
    for P_target in price_levels:
        r_target = r_base * (P_target / P_base) ** elasticity
        u_line   = s_line / r_target
        if u_line.min() > y_max * 1.05 or u_line.max() < y_min * 0.95:
            continue
        fig.add_scatter(
            x=s_line, y=u_line,
            mode="lines",
            line=dict(color=AEI["gray"], width=1, dash="dot"),
            name=f"${P_target:.1f}/bu",
            legendgroup="iso",
            showlegend=True,
            hovertemplate=f"Iso-price: ${P_target:.1f}/bu<extra></extra>",
        )

    # 2σ ellipse
    fig.add_scatter(
        x=x_2sig, y=y_2sig,
        mode="lines",
        line=dict(color="rgba(96,157,66,0.45)", width=1.5, dash="dot"),
        fill="toself",
        fillcolor="rgba(96,157,66,0.06)",
        name="2σ region",
        hoverinfo="skip",
    )

    # 1σ ellipse
    fig.add_scatter(
        x=x_1sig, y=y_1sig,
        mode="lines",
        line=dict(color="rgba(96,157,66,0.80)", width=1.5),
        fill="toself",
        fillcolor="rgba(96,157,66,0.15)",
        name="1σ region",
        hoverinfo="skip",
    )

    # Historical data points
    fig.add_scatter(
        x=supply_vals, y=usage_vals,
        mode="markers+text",
        text=years,
        textposition="top center",
        textfont=dict(size=8, color=AEI["gray"]),
        marker=dict(color=AEI["navy"], size=7, opacity=0.8),
        name="Historical",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Supply: %{x:,.0f} mil. bu<br>"
            "Usage: %{y:,.0f} mil. bu"
            "<extra></extra>"
        ),
    )

    # Scenario star
    scenario_label = _ellipse_region_label(
        scenario_supply, scenario_usage, mean_s, mean_u, eigenvalues, eigenvectors
    )
    fig.add_scatter(
        x=[scenario_supply], y=[scenario_usage],
        mode="markers+text",
        text=[f"Your scenario ({scenario_label})"],
        textposition="bottom center",
        textfont=dict(size=9, color=AEI["orange"]),
        marker=dict(
            color=AEI["orange"], size=16, symbol="star",
            line=dict(color=AEI["navy"], width=1.5),
        ),
        name="Your scenario",
        hovertemplate=(
            "<b>Your Scenario</b><br>"
            f"Supply: {scenario_supply:,.0f} mil. bu<br>"
            f"Usage: {scenario_usage:,.0f} mil. bu<br>"
            f"Implied price: ${scenario_price:.2f}/bu<br>"
            f"<i>{scenario_label}</i>"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        xaxis=dict(
            title="Total Supply (mil. bu)",
            range=[x_min, x_max],
            showgrid=False,
            linecolor=AEI["gray"],
            tickformat=",d",
        ),
        yaxis=dict(
            title="Total Usage (mil. bu)",
            range=[y_min, y_max],
            gridcolor="#EBEBEB",
            linecolor=AEI["gray"],
            tickformat=",d",
        ),
        height=480,
        margin=dict(t=20, b=50, l=70, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="closest",
    )
    return fig


def _ellipse_region_label(
    s: float,
    u: float,
    mean_s: float,
    mean_u: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> str:
    """Plain-English label for where (supply, usage) falls relative to ellipses."""
    delta       = np.array([s - mean_s, u - mean_u])
    transformed = eigenvectors.T @ delta
    std_dist    = np.sqrt(np.sum((transformed / np.sqrt(eigenvalues)) ** 2))
    if std_dist <= 1.0:
        return "within 1σ — historically normal"
    elif std_dist <= 2.0:
        return "within 2σ — somewhat unusual"
    else:
        return "outside 2σ — historically rare"


# ---------------------------------------------------------------------------
# Farmer-friendly summary table
# ---------------------------------------------------------------------------

def farmer_table(
    results: pd.DataFrame,
    scenario_row: dict | None = None,
) -> pd.DataFrame:
    """
    Clean, plain-English summary table. If scenario_row is provided it is
    appended as the final row, styled distinctly.
    """
    tbl = results[[
        "year", "price_real", "pred_price",
        "dPrice", "supply_impact", "demand_impact",
    ]].copy()

    tbl["price_real"]    = tbl["price_real"].round(2)
    tbl["pred_price"]    = tbl["pred_price"].round(2)
    tbl["dPrice"]        = tbl["dPrice"].round(2)
    tbl["supply_impact"] = tbl["supply_impact"].round(2)
    tbl["demand_impact"] = tbl["demand_impact"].round(2)

    tbl.rename(columns={
        "year":           "Year",
        "price_real":     "Actual Price ($/bu)",
        "pred_price":     "Model Price ($/bu)",
        "dPrice":         "Change vs Prior Year ($/bu)",
        "supply_impact":  "Supply Drove ($/bu)",
        "demand_impact":  "Usage Drove ($/bu)",
    }, inplace=True)

    tbl = tbl.set_index("Year")

    if scenario_row is not None:
        label    = scenario_row["year_label"]
        scen_df  = pd.DataFrame(
            [{k: v for k, v in scenario_row.items() if k != "year_label"}],
            index=[label],
        )
        scen_df.index.name = "Year"
        tbl = pd.concat([tbl, scen_df])

    return tbl


# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------

def render_tab(
    crop_label: str,
    results: pd.DataFrame,
    elasticity: float,
    scenario_supply: float | None = None,
    scenario_usage: float | None = None,
    scenario_price: float | None = None,
    scenario_row: dict | None = None,
) -> None:
    check = verify_decomp(results)
    if not check["pass"]:
        st.warning(f"Decomposition check failed (max error = {check['max_resid_error']:.1e})")

    # ---- Key metric strip (latest year) ----
    latest = results.iloc[-1]
    prev   = results.iloc[-2]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Latest Year", int(latest["year"]))
    m2.metric(
        "Actual Price",
        fmt2(latest["price_real"]),
        delta=f"{latest['dPrice']:+.2f} vs {int(prev['year'])}",
    )
    m3.metric("Model Price", fmt2(latest["pred_price"]))
    m4.metric(
        "Supply Effect",
        f"{latest['supply_impact']:+.2f}",
        help="How much of last year's price change came from supply shifts ($/bu, real 2025$)",
    )
    m5.metric(
        "Usage Effect",
        f"{latest['demand_impact']:+.2f}",
        help="How much of last year's price change came from usage shifts ($/bu, real 2025$)",
    )

    st.markdown("&nbsp;")

    # ---- Price and decomposition charts ----
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Actual vs Model Price** — real 2025 $/bu")
        st.plotly_chart(price_chart(results), width="stretch")
    with col_r:
        st.markdown("**What Moved the Price?**")
        st.plotly_chart(shapley_chart(results), width="stretch")

    # ---- Scenario supply/usage chart ----
    if scenario_supply is not None and scenario_usage is not None and scenario_price is not None:
        st.markdown("---")
        st.markdown("**Where Does Your Scenario Fall?** — Supply & Usage in Context")
        st.caption(
            "Dotted gray lines are iso-price contours — each line connects every "
            "(Supply, Usage) combination that implies the same model price. "
            "The shaded regions show 1σ (darker) and 2σ (lighter) of the "
            "historical supply/usage relationship. "
            "The ★ marks your scenario."
        )
        st.plotly_chart(
            scenario_ellipse_chart(
                results, scenario_supply, scenario_usage, scenario_price, elasticity
            ),
            width="stretch",
        )

    # ---- Summary table ----
    st.markdown("---")
    st.markdown("**Year-by-Year Summary**")
    st.caption(
        '"Supply Drove" = how much supply changes alone moved the price. '
        '"Usage Drove" = same for usage. '
        "Both sum to the total price change. Positive = price up. "
        + ("Last row is your hypothetical scenario." if scenario_row else "")
    )
    tbl = farmer_table(results, scenario_row)

    def _color_impact(val):
        if val > 0:
            return f"color: {AEI['green']}; font-weight: 600"
        elif val < 0:
            return f"color: {AEI['red']}; font-weight: 600"
        return ""

    st.dataframe(
        tbl.style
        .format({
            "Actual Price ($/bu)":         "${:.2f}",
            "Model Price ($/bu)":          "${:.2f}",
            "Change vs Prior Year ($/bu)": lambda v: f"{v:+.2f}",
            "Supply Drove ($/bu)":         lambda v: f"{v:+.2f}",
            "Usage Drove ($/bu)":          lambda v: f"{v:+.2f}",
        })
        .map(_color_impact, subset=["Supply Drove ($/bu)", "Usage Drove ($/bu)"]),
        use_container_width=True,
    )

    st.caption(
        f"Elasticity: {elasticity} &nbsp;|&nbsp; Base year: {BASE_YEAR} &nbsp;|&nbsp; "
        "Supply & use in million bushels"
    )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

tab_corn, tab_soy = st.tabs(["🌽 Corn", "🫘 Soybeans"])

with tab_corn:
    render_tab(
        "corn", corn_res, CORN_ELASTICITY,
        scenario_supply = scen_supply if scen_crop == "Corn"     else None,
        scenario_usage  = scen_usage  if scen_crop == "Corn"     else None,
        scenario_price  = ratio_pred  if scen_crop == "Corn"     else None,
        scenario_row    = _corn_scen_row,
    )

with tab_soy:
    render_tab(
        "soybeans", soy_res, SOYBEAN_ELASTICITY,
        scenario_supply = scen_supply if scen_crop == "Soybeans" else None,
        scenario_usage  = scen_usage  if scen_crop == "Soybeans" else None,
        scenario_price  = ratio_pred  if scen_crop == "Soybeans" else None,
        scenario_row    = _soy_scen_row,
    )
