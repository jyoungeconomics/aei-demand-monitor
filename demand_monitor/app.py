"""
Corn & Soybean Demand Monitor — Streamlit dashboard.

Run with:
    streamlit run demand_monitor/app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
          background: {AEI["green"]}18;
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
        fillcolor=f"{AEI['steel']}18",
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

def render_tab(crop_label: str, results: pd.DataFrame, elasticity: float) -> None:
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
        st.plotly_chart(price_chart(results), use_container_width=True)
    with col_r:
        st.markdown("**What Moved the Price?** — Shapley decomposition")
        st.plotly_chart(shapley_chart(results), use_container_width=True)

    st.markdown("**Supply-to-Use Ratio**")
    st.plotly_chart(su_chart(results), use_container_width=True)

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
        use_container_width=True,
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
    render_tab("corn", corn_res, CORN_ELASTICITY)

with tab_soy:
    render_tab("soybeans", soy_res, SOYBEAN_ELASTICITY)
