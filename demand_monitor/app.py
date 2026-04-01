"""
Corn & Soybean Demand Monitor — Streamlit dashboard.

Run with:
    streamlit run demand_monitor/app.py
"""

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
    "green":      "#609D42",   # primary brand green  → actual price, supply effect
    "navy":       "#001425",   # navy                 → supply bars, data points
    "orange":     "#F28C28",   # burnt orange         → total change marker, scenario star
    "gray":       "#A7A9AC",   # light gray           → gridlines / secondary text
    "teal":       "#3B8B8B",   # teal                 → predicted price line
    "red":        "#C0504D",   # soft red             → negative / warning
    "steel":      "#4F81BD",   # steel blue           → unused (kept for reference)
    "dark_green": "#508428",   # deeper green         → unused (kept for reference)
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

# Inject thin brand-navy top bar and tighten padding
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
    f'Constant elasticity of demand model &nbsp;|&nbsp; '
    f'Base year: {BASE_YEAR} &nbsp;|&nbsp; Prices in real 2025 dollars</p>',
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
# Sidebar — Scenario / "What-If" Tool
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔧 What-If Scenario")
    st.caption(
        "Plug in your own supply and usage numbers — "
        "the model instantly shows the implied price."
    )

    scen_crop = st.radio("Crop", ["Corn", "Soybeans"], horizontal=True)

    # Pull defaults from the most recent year with full data
    if scen_crop == "Corn":
        ref = corn_res.iloc[-1]
        e, b0, b1 = CORN_ELASTICITY, CORN_OLS_INTERCEPT, CORN_OLS_SLOPE
        df_full  = corn_df
    else:
        ref = soy_res.iloc[-1]
        e, b0, b1 = SOYBEAN_ELASTICITY, SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE
        df_full  = soy_df

    default_s = float(round(ref["supply"]))
    default_u = float(round(ref["usage"]))

    st.markdown("**Adjust supply & use** (mil. bu)")
    scen_supply = st.number_input(
        "Total Supply", min_value=100.0, max_value=900_000.0,
        value=default_s, step=50.0, format="%.0f",
        help="Beginning stocks + production + imports (million bushels)",
    )
    scen_usage = st.number_input(
        "Total Use", min_value=100.0, max_value=900_000.0,
        value=default_u, step=50.0, format="%.0f",
        help="Feed + FSI/Crush + exports (million bushels)",
    )

    G_scen     = compute_g(scen_supply, scen_usage, e)
    base_row   = df_full[df_full["year"] == BASE_YEAR].iloc[0]
    G_base_val = compute_g(float(base_row["supply"]), float(base_row["usage"]), e)
    P_base_val = float(df_full[df_full["year"] == BASE_YEAR]["price_real"].values[0])
    ratio_pred = P_base_val * (G_scen / G_base_val)

    st.markdown("---")
    st.markdown("**Model result (ratio method)**")
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

    with st.expander("➕ Add a current spot price (OLS method)"):
        scen_spot = st.number_input(
            "Current spot price (2025 real $/bu)",
            min_value=0.50, max_value=50.0,
            value=last_actual, step=0.10, format="%.2f",
            help="Enter in real 2025 dollars. Use spot price ÷ your deflator if converting from nominal.",
        )
        K_val   = 100.0 / (G_base_val * P_base_val)
        IV_scen = K_val * G_scen * scen_spot
        ols_pred = b0 + b1 * IV_scen
        st.markdown(
            f"OLS predicted: **${ols_pred:.2f}/bu** "
            f"(S/D Index: {IV_scen:.1f})"
        )

    st.markdown("---")
    st.caption(
        "Ratio method: P = P₂₀₀₉ × (G / G₂₀₀₉)\n\n"
        "OLS method: P = b₀ + b₁ × (K × G × Pspot)\n\n"
        "Both return real 2025 $/bu."
    )

# Auto-switch the main content tab to match the sidebar crop selection.
# Uses a retry loop because Streamlit renders the tabs after this component.
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
# Chart helpers
# ---------------------------------------------------------------------------

def fmt2(x):
    """Format a float as a dollar-rounded-to-2 string."""
    return f"${x:.2f}"


def price_chart(results: pd.DataFrame) -> go.Figure:
    """Time series of actual vs model-predicted price."""
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
    Bar chart decomposing the year-over-year price change into supply and usage
    contributions. Supply bars are navy, usage bars are green.
    Orange diamonds show the total actual price change.
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
    Supply vs Usage scatterplot with:
      - Iso-price contour lines (dotted gray) — each line connects all
        (Supply, Usage) combinations that imply the same model price.
        Because G = (S/U)^(1/ε), these are straight lines through the origin;
        in the data window they appear as upward-sloping parallel lines.
      - 1σ and 2σ confidence ellipses from the covariance of historical
        supply and usage (eigenvector decomposition).
      - Historical data points labeled by year.
      - An orange ★ at the subscriber's scenario.

    Equal-unit axis scaling (scaleratio=1) ensures iso-price lines with
    slope ~1 appear near 45°, matching the guardrail-writeup charts.
    """
    hist        = results.dropna(subset=["supply", "usage", "price_real"])
    supply_vals = hist["supply"].values
    usage_vals  = hist["usage"].values
    years       = hist["year"].astype(int).astype(str).values

    # --- Covariance and eigenvector decomposition ---
    mean_s = np.mean(supply_vals)
    mean_u = np.mean(usage_vals)
    std_s  = np.std(supply_vals, ddof=1)
    std_u  = np.std(usage_vals,  ddof=1)
    cov    = np.cov(supply_vals, usage_vals)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    theta  = np.linspace(0, 2 * np.pi, 300)
    circle = np.array([np.cos(theta), np.sin(theta)])

    def make_ellipse(n_sigma):
        scaled = eigenvectors @ np.diag(n_sigma * np.sqrt(eigenvalues)) @ circle
        return scaled[0] + mean_s, scaled[1] + mean_u

    x_2sig, y_2sig = make_ellipse(2)
    x_1sig, y_1sig = make_ellipse(1)

    # --- Iso-price lines ---
    # Ratio method: P / P_base = (G / G_base) = ((S/U) / r_base)^(1/ε)
    # For iso-price at P_target:
    #   r_target = r_base * (P_target / P_base)^ε      (note: ε < 0)
    #   iso-price line: U = S / r_target
    base_rows = hist[hist["year"] == BASE_YEAR]
    base_row  = base_rows.iloc[0] if not base_rows.empty else hist.iloc[0]
    r_base    = float(base_row["supply"]) / float(base_row["usage"])
    P_base    = float(base_row["price_real"])

    prices_hist = hist["price_real"].values
    p_lo = np.floor(prices_hist.min() * 2) / 2   # nearest $0.50 below min
    p_hi = np.ceil(prices_hist.max()  * 2) / 2   # nearest $0.50 above max
    price_levels = np.linspace(p_lo, p_hi, 6)

    s_lo = supply_vals.min() * 0.97
    s_hi = supply_vals.max() * 1.03
    u_lo = usage_vals.min()  * 0.95
    u_hi = usage_vals.max()  * 1.05
    s_line = np.array([s_lo, s_hi])

    fig = go.Figure()

    # Draw iso-price lines first so they sit behind the data
    for P_target in price_levels:
        r_target = r_base * (P_target / P_base) ** elasticity
        u_line   = s_line / r_target
        # Skip lines that fall entirely outside the visible usage range
        if u_line.min() > u_hi * 1.1 or u_line.max() < u_lo * 0.9:
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

    # 2σ ellipse — outer, light fill
    fig.add_scatter(
        x=x_2sig, y=y_2sig,
        mode="lines",
        line=dict(color="rgba(96,157,66,0.45)", width=1.5, dash="dot"),
        fill="toself",
        fillcolor="rgba(96,157,66,0.06)",
        name="2σ region",
        hoverinfo="skip",
    )

    # 1σ ellipse — inner, slightly stronger fill
    fig.add_scatter(
        x=x_1sig, y=y_1sig,
        mode="lines",
        line=dict(color="rgba(96,157,66,0.80)", width=1.5),
        fill="toself",
        fillcolor="rgba(96,157,66,0.15)",
        name="1σ region",
        hoverinfo="skip",
    )

    # Historical data points labeled by year
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
            color=AEI["orange"],
            size=16,
            symbol="star",
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

    # Equal-unit axis scaling: 1 mil bu on y-axis = 1 mil bu on x-axis.
    # This ensures iso-price lines (slope ≈ 1) appear near 45°.
    fig.update_layout(
        xaxis_title="Total Supply (mil. bu)",
        yaxis_title="Total Usage (mil. bu)",
        height=520,
        margin=dict(t=20, b=50, l=70, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            linecolor=AEI["gray"],
            tickformat=",d",
        ),
        yaxis=dict(
            gridcolor="#EBEBEB",
            linecolor=AEI["gray"],
            tickformat=",d",
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right", x=1,
        ),
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
    """
    Return a plain-English label for where (supply, usage) falls relative
    to the historical 1σ / 2σ ellipses.
    """
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

def farmer_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, plain-English summary of year-by-year supply, usage, and price
    drivers. Columns use farmer-friendly language.
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

    return tbl.set_index("Year")


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
        delta=None,
        help="How much of last year's price change came from supply shifts ($/bu, real 2025$)",
    )
    m5.metric(
        "Usage Effect",
        f"{latest['demand_impact']:+.2f}",
        delta=None,
        help="How much of last year's price change came from usage shifts ($/bu, real 2025$)",
    )

    st.markdown("&nbsp;")

    # ---- Charts ----
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Actual vs Model Price** — real 2025 $/bu")
        st.plotly_chart(price_chart(results), width="stretch")
    with col_r:
        st.markdown("**What Moved the Price?**")
        st.plotly_chart(shapley_chart(results), width="stretch")

    # ---- Scenario supply/usage chart (shown only when scenario is active) ----
    if scenario_supply is not None and scenario_usage is not None and scenario_price is not None:
        st.markdown("---")
        st.markdown("**Where Does Your Scenario Fall?** — Supply & Usage in Context")
        st.caption(
            "Dotted gray lines are iso-price contours — each line connects every "
            "(Supply, Usage) combination that implies the same model price. "
            "The shaded regions show 1σ (darker) and 2σ (lighter) of the historical "
            "supply/usage relationship. "
            "The ★ marks your scenario from the sidebar."
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
        '"Supply Drove" = how much supply changes alone moved the price that year. '
        '"Usage Drove" = same for usage. '
        "Both sum to the total actual price change. Positive = price went up."
    )
    tbl = farmer_table(results)

    def _color_impact(val):
        """Green for positive price impacts, red for negative."""
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
        "corn",
        corn_res,
        CORN_ELASTICITY,
        scenario_supply = scen_supply if scen_crop == "Corn"     else None,
        scenario_usage  = scen_usage  if scen_crop == "Corn"     else None,
        scenario_price  = ratio_pred  if scen_crop == "Corn"     else None,
    )

with tab_soy:
    render_tab(
        "soybeans",
        soy_res,
        SOYBEAN_ELASTICITY,
        scenario_supply = scen_supply if scen_crop == "Soybeans" else None,
        scenario_usage  = scen_usage  if scen_crop == "Soybeans" else None,
        scenario_price  = ratio_pred  if scen_crop == "Soybeans" else None,
    )
