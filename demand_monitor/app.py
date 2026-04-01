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
# AEI brand palette (from chart_helpers.py)
# ---------------------------------------------------------------------------
AEI = {
    "green":      "#609D42",   # primary brand green  → actual price, supply tightening
    "navy":       "#001425",   # navy                 → supply impact bars
    "orange":     "#F28C28",   # burnt orange         → demand impact bars
    "gray":       "#A7A9AC",   # light gray           → gridlines / secondary text
    "teal":       "#3B8B8B",   # teal                 → predicted price
    "red":        "#C0504D",   # soft red             → negative / warning
    "steel":      "#4F81BD",   # steel blue           → S/U ratio
    "dark_green": "#508428",   # deeper green         → baseline reference
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

# Inject a thin brand-green top bar and tighten default padding
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
    f'<p class="aei-sub">American Enterprise Institute &nbsp;|&nbsp; '
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


corn_df,  corn_res  = load_all("corn",     CORN_ELASTICITY,    CORN_OLS_INTERCEPT,    CORN_OLS_SLOPE)
soy_df,   soy_res   = load_all("soybeans", SOYBEAN_ELASTICITY, SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE)


# ---------------------------------------------------------------------------
# Sidebar — Scenario / "What-If" Tool
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"### 🔧 What-If Scenario")
    st.caption(
        "Plug in your own supply and usage numbers — "
        "the model instantly shows the implied price."
    )

    scen_crop = st.radio("Crop", ["Corn", "Soybeans"], horizontal=True)

    # Pull defaults from the most recent year with full data
    if scen_crop == "Corn":
        ref = corn_res.iloc[-1]
        e, b0, b1 = CORN_ELASTICITY, CORN_OLS_INTERCEPT, CORN_OLS_SLOPE
        df_full   = corn_df
        res_full  = corn_res
        unit      = "mil. bu"
        crop_key  = "corn"
    else:
        ref = soy_res.iloc[-1]
        e, b0, b1 = SOYBEAN_ELASTICITY, SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE
        df_full   = soy_df
        res_full  = soy_res
        unit      = "mil. bu"
        crop_key  = "soybeans"

    default_s = float(round(ref["supply"]))
    default_u = float(round(ref["usage"]))

    st.markdown(f"**Adjust supply & use** ({unit})")
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

    scen_su = scen_supply / scen_usage
    G_scen  = compute_g(scen_supply, scen_usage, e)

    # Ratio-method prediction (no spot price needed)
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
        f"S/U Ratio: {scen_su:.4f} &nbsp;|&nbsp; G index: {G_scen:.4f}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    last_actual = float(ref["price_real"])
    delta_vs_actual = ratio_pred - last_actual
    sign = "+" if delta_vs_actual >= 0 else ""
    color = AEI["green"] if delta_vs_actual >= 0 else AEI["red"]
    st.markdown(
        f'<span style="color:{color}; font-weight:600;">'
        f"{sign}${delta_vs_actual:.2f} vs last actual ({int(ref['year'])})"
        f"</span>",
        unsafe_allow_html=True,
    )

    # Optional OLS with spot price
    with st.expander("➕ Add a current spot price (OLS method)"):
        scen_spot = st.number_input(
            "Current spot price (2025 real $/bu)",
            min_value=0.50, max_value=50.0,
            value=last_actual, step=0.10, format="%.2f",
            help="Enter in real 2025 dollars. Use spot price ÷ your deflator if converting from nominal.",
        )
        K_val  = 100.0 / (G_base_val * P_base_val)
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


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def fmt2(x):
    """Format a float to 2 decimal places as a dollar string."""
    return f"${x:.2f}"


def price_chart(results: pd.DataFrame) -> go.Figure:
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
    fig = go.Figure()
    fig.add_bar(
        x=results["year"], y=results["supply_impact"].round(2),
        name="Supply effect",
        marker_color=AEI["navy"],
        hovertemplate="<b>%{x}</b><br>Supply: %{y:+.2f}/bu<extra></extra>",
    )
    fig.add_bar(
        x=results["year"], y=results["demand_impact"].round(2),
        name="Demand effect",
        marker_color=AEI["orange"],
        hovertemplate="<b>%{x}</b><br>Demand: %{y:+.2f}/bu<extra></extra>",
    )
    fig.add_scatter(
        x=results["year"], y=results["dPrice"].round(2),
        mode="markers", name="Total change",
        marker=dict(color=AEI["green"], size=8, symbol="diamond",
                    line=dict(width=1, color=AEI["dark_green"])),
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


def su_chart(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=results["year"], y=results["su_ratio"].round(4),
        mode="lines+markers", name="S/U Ratio",
        line=dict(color=AEI["steel"], width=2),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(79,129,189,0.09)",   # steel blue at ~9% opacity (Plotly 6 compatible)
        hovertemplate="<b>%{x}</b><br>S/U: %{y:.4f}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="Marketing Year",
        yaxis_title="Supply ÷ Total Use",
        height=280,
        margin=dict(t=10, b=40, l=60, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, linecolor=AEI["gray"]),
        yaxis=dict(gridcolor="#EBEBEB", linecolor=AEI["gray"]),
        showlegend=False,
    )
    return fig


def scenario_ellipse_chart(
    results: pd.DataFrame,
    scenario_su: float,
    scenario_price: float,
) -> go.Figure:
    """
    Scatter of historical S/U ratio vs real price, with 1σ and 2σ confidence
    ellipses derived from the covariance matrix (eigenvector decomposition).

    The orange star marks the subscriber's scenario from the sidebar.
    Points inside 1σ represent the most 'normal' market conditions;
    outside 2σ are historically unusual combinations.
    """
    hist = results.dropna(subset=["su_ratio", "price_real"])
    su_vals    = hist["su_ratio"].values
    price_vals = hist["price_real"].values
    years      = hist["year"].astype(int).astype(str).values

    # --- Covariance and eigenvector decomposition ---
    mean_su    = np.mean(su_vals)
    mean_price = np.mean(price_vals)
    cov        = np.cov(su_vals, price_vals)           # 2×2 covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)    # eigenvalues in ascending order

    # Parametric ellipse: unit circle transformed by sqrt(eigenvalues) and eigenvectors
    theta  = np.linspace(0, 2 * np.pi, 300)
    circle = np.array([np.cos(theta), np.sin(theta)])  # (2, 300)

    def make_ellipse(n_sigma):
        """Return (x, y) arrays for an n-sigma ellipse."""
        scaled    = eigenvectors @ np.diag(n_sigma * np.sqrt(eigenvalues)) @ circle
        return scaled[0] + mean_su, scaled[1] + mean_price

    x_2sig, y_2sig = make_ellipse(2)
    x_1sig, y_1sig = make_ellipse(1)

    fig = go.Figure()

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
        x=su_vals, y=price_vals,
        mode="markers+text",
        text=years,
        textposition="top center",
        textfont=dict(size=8, color=AEI["gray"]),
        marker=dict(color=AEI["navy"], size=7, opacity=0.8),
        name="Historical",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "S/U: %{x:.4f}<br>"
            "Price: $%{y:.2f}/bu"
            "<extra></extra>"
        ),
    )

    # Scenario star
    # Determine whether scenario is inside/outside ellipses for the label
    scenario_label = _ellipse_region_label(
        scenario_su, scenario_price, mean_su, mean_price, eigenvalues, eigenvectors
    )
    fig.add_scatter(
        x=[scenario_su], y=[scenario_price],
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
            f"<b>Your Scenario</b><br>"
            f"S/U: {scenario_su:.4f}<br>"
            f"Price: ${scenario_price:.2f}/bu<br>"
            f"<i>{scenario_label}</i>"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        xaxis_title="Supply-to-Use Ratio",
        yaxis_title="Real Price (2025 $/bu)",
        height=440,
        margin=dict(t=20, b=50, l=60, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, linecolor=AEI["gray"]),
        yaxis=dict(gridcolor="#EBEBEB", linecolor=AEI["gray"]),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right", x=1,
        ),
        hovermode="closest",
    )
    return fig


def _ellipse_region_label(
    su: float,
    price: float,
    mean_su: float,
    mean_price: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> str:
    """
    Return a plain-English label indicating whether the point falls inside
    the 1σ, 2σ, or outside both ellipses.
    """
    # Project the point into the eigenvector basis and compute Mahalanobis distance
    delta       = np.array([su - mean_su, price - mean_price])
    transformed = eigenvectors.T @ delta                      # rotate into principal axes
    # Normalize by the semi-axis lengths (sqrt of eigenvalues)
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
    Return a clean, farmer-friendly summary with plain-English column names.
    Supply/demand effects tell the grower what drove the price that year.
    """
    tbl = results[[
        "year", "price_real", "pred_price",
        "dPrice", "supply_impact", "demand_impact", "su_ratio",
    ]].copy()

    # Round everything sensibly
    tbl["price_real"]    = tbl["price_real"].round(2)
    tbl["pred_price"]    = tbl["pred_price"].round(2)
    tbl["dPrice"]        = tbl["dPrice"].round(2)
    tbl["supply_impact"] = tbl["supply_impact"].round(2)
    tbl["demand_impact"] = tbl["demand_impact"].round(2)
    tbl["su_ratio"]      = tbl["su_ratio"].round(4)

    tbl.rename(columns={
        "year":           "Year",
        "price_real":     "Actual Price ($/bu)",
        "pred_price":     "Model Price ($/bu)",
        "dPrice":         "Change vs Prior Year ($/bu)",
        "supply_impact":  "Supply Drove ($/bu)",
        "demand_impact":  "Demand Drove ($/bu)",
        "su_ratio":       "Supply-to-Use Ratio",
    }, inplace=True)

    return tbl.set_index("Year")


# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------

def render_tab(
    crop_label: str,
    results: pd.DataFrame,
    elasticity: float,
    scenario_su: float | None = None,
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
        "Demand Effect",
        f"{latest['demand_impact']:+.2f}",
        delta=None,
        help="How much came from demand shifts ($/bu, real 2025$)",
    )

    st.markdown("&nbsp;")

    # ---- Charts ----
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"**Actual vs Model Price** — real 2025 $/bu")
        st.plotly_chart(price_chart(results), width="stretch")
    with col_r:
        st.markdown("**What Moved the Price?** — Shapley decomposition")
        st.plotly_chart(shapley_chart(results), width="stretch")

    st.markdown("**Supply-to-Use Ratio**")
    st.plotly_chart(su_chart(results), width="stretch")

    # ---- Scenario confidence ellipse ----
    if scenario_su is not None and scenario_price is not None:
        st.markdown("---")
        st.markdown("**Where Does Your Scenario Fall?** — Historical S/U vs Price")
        st.caption(
            "The shaded regions show 1σ (darker) and 2σ (lighter) of the historical "
            "supply-to-use / price relationship, derived from the eigenvectors of the "
            "covariance matrix. "
            "The ★ marks your scenario from the sidebar. "
            "Points outside the 2σ ellipse represent historically unusual market conditions."
        )
        st.plotly_chart(
            scenario_ellipse_chart(results, scenario_su, scenario_price),
            width="stretch",
        )

    # ---- Summary table ----
    st.markdown("---")
    st.markdown("**Year-by-Year Summary**")
    st.caption(
        '"Supply Drove" = how much supply changes alone moved the price that year. '
        '"Demand Drove" = same for usage/demand. '
        "Both sum to the total actual price change. Positive = price went up."
    )
    tbl = farmer_table(results)
    st.dataframe(
        tbl.style
        .format({
            "Actual Price ($/bu)":       "${:.2f}",
            "Model Price ($/bu)":        "${:.2f}",
            "Change vs Prior Year ($/bu)": lambda v: f"{v:+.2f}",
            "Supply Drove ($/bu)":       lambda v: f"{v:+.2f}",
            "Demand Drove ($/bu)":       lambda v: f"{v:+.2f}",
            "Supply-to-Use Ratio":       "{:.4f}",
        })
        .background_gradient(
            subset=["Supply Drove ($/bu)", "Demand Drove ($/bu)"],
            cmap="RdYlGn", vmin=-4, vmax=4,
        ),
        width="stretch",
        height=None,
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
        scenario_su    = scen_su    if scen_crop == "Corn" else None,
        scenario_price = ratio_pred if scen_crop == "Corn" else None,
    )

with tab_soy:
    render_tab(
        "soybeans",
        soy_res,
        SOYBEAN_ELASTICITY,
        scenario_su    = scen_su    if scen_crop == "Soybeans" else None,
        scenario_price = ratio_pred if scen_crop == "Soybeans" else None,
    )
