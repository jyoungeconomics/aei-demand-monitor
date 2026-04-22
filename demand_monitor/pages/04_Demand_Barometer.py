"""
Demand Barometer — IV (Index Value) Tracker

Shows the relative strength of demand for corn/soybeans vs. the 2009 base year.
IV combines supply/usage tightness with absolute price to quantify demand environment.
"""

import streamlit as st
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import AEI, corn_res, soy_res, corn_df, soy_df

st.set_page_config(page_title="Demand Barometer", page_icon="📊", layout="wide")

# Get the selected crop from session state (set by main app.py sidebar)
scen_crop = st.session_state.get("scen_crop", "Corn")

# Select data for this crop
if scen_crop == "Corn":
    results = corn_res
    df = corn_df
    crop_label = "🌽 Corn"
else:
    results = soy_res
    df = soy_df
    crop_label = "🫛 Soybeans"

# Get current year and IV data
latest = results.iloc[-1]
current_iv = latest["IV"]
current_year = int(latest["year"])

# Define demand strength zones
def get_strength_label(iv_value):
    if iv_value < 75:
        return "Weak Demand"
    elif iv_value < 100:
        return "Below Normal"
    elif iv_value < 125:
        return "Normal Demand"
    elif iv_value < 150:
        return "Above Normal"
    else:
        return "Strong Demand"

def get_strength_color(iv_value):
    if iv_value < 75:
        return AEI["red"]
    elif iv_value < 100:
        return AEI["yellow"]
    elif iv_value < 125:
        return AEI["green"]
    elif iv_value < 150:
        return AEI["orange"]
    else:
        return AEI["dark_green"]

strength_label = get_strength_label(current_iv)
strength_color = get_strength_color(current_iv)

# ============================================================================
# Page Layout
# ============================================================================

st.markdown(f"## {crop_label} Demand Barometer")

# Info card
st.info(
    "**IV (Index Value)** measures the relative strength of demand for this crop vs. the 2009 baseline. "
    "It combines supply/usage tightness with absolute price level to quantify the overall demand environment. "
    "IV = 100 in the base year (2009)."
)

# ============================================================================
# Current IV Gauge + Large Metric
# ============================================================================

col_metric, col_gauge = st.columns([1, 1.5])

with col_metric:
    st.metric(
        label="Current IV",
        value=f"{current_iv:.1f}",
        delta=f"{current_year} vs. 2009 baseline",
        delta_color="off",
    )
    st.markdown(
        f"<p style='text-align: center; font-size: 1.2em; color: {strength_color}; font-weight: bold;'>"
        f"{strength_label}</p>",
        unsafe_allow_html=True,
    )

with col_gauge:
    # Create speedometer gauge
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_iv,
            title={"text": "Demand Strength Index"},
            delta={"reference": 100, "suffix": " vs. 2009"},
            gauge={
                "axis": {"range": [0, 200]},
                "bar": {"color": strength_color},
                "steps": [
                    {"range": [0, 75], "color": f"rgba({int(AEI['red'][1:3], 16)}, {int(AEI['red'][3:5], 16)}, {int(AEI['red'][5:7], 16)}, 0.1)"},
                    {"range": [75, 100], "color": f"rgba({int(AEI['yellow'][1:3], 16)}, {int(AEI['yellow'][3:5], 16)}, {int(AEI['yellow'][5:7], 16)}, 0.1)"},
                    {"range": [100, 125], "color": f"rgba({int(AEI['green'][1:3], 16)}, {int(AEI['green'][3:5], 16)}, {int(AEI['green'][5:7], 16)}, 0.1)"},
                    {"range": [125, 150], "color": f"rgba({int(AEI['orange'][1:3], 16)}, {int(AEI['orange'][3:5], 16)}, {int(AEI['orange'][5:7], 16)}, 0.1)"},
                    {"range": [150, 200], "color": f"rgba({int(AEI['dark_green'][1:3], 16)}, {int(AEI['dark_green'][3:5], 16)}, {int(AEI['dark_green'][5:7], 16)}, 0.1)"},
                ],
                "threshold": {
                    "line": {"color": AEI["navy"], "width": 2},
                    "thickness": 0.75,
                    "value": 100,
                },
            },
        )
    )
    fig_gauge.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ============================================================================
# Historical IV Time Series
# ============================================================================

st.markdown("### Historical Demand Strength")

fig_iv = go.Figure()

# Add IV line
fig_iv.add_trace(
    go.Scatter(
        x=results["year"],
        y=results["IV"],
        mode="lines+markers",
        name="IV",
        line=dict(color=AEI["green"], width=3),
        marker=dict(size=6),
        hovertemplate=(
            "<b>%{x|%Y}</b><br>"
            "IV: %{y:.1f}<br>"
            "S/U: %{customdata[0]:.2f}<br>"
            "Price (real): $%{customdata[1]:.2f}/bu<extra></extra>"
        ),
        customdata=results[["su_ratio", "price_real"]].values,
    )
)

# Add 2009 baseline reference line
fig_iv.add_hline(
    y=100,
    line_dash="dash",
    line_color=AEI["gray"],
    annotation_text="2009 Baseline (IV=100)",
    annotation_position="right",
)

# Add demand strength zones as shaded regions
fig_iv.add_hrect(
    y0=0, y1=75,
    fillcolor=AEI["red"], opacity=0.05,
    layer="below", line_width=0,
)
fig_iv.add_hrect(
    y0=75, y1=100,
    fillcolor=AEI["yellow"], opacity=0.05,
    layer="below", line_width=0,
)
fig_iv.add_hrect(
    y0=100, y1=125,
    fillcolor=AEI["green"], opacity=0.05,
    layer="below", line_width=0,
)
fig_iv.add_hrect(
    y0=125, y1=150,
    fillcolor=AEI["orange"], opacity=0.05,
    layer="below", line_width=0,
)
fig_iv.add_hrect(
    y0=150, y1=200,
    fillcolor=AEI["dark_green"], opacity=0.05,
    layer="below", line_width=0,
)

fig_iv.update_layout(
    title=f"IV Time Series: {crop_label} (2000–{current_year})",
    xaxis_title="Marketing Year",
    yaxis_title="IV (Index Value, 2009 = 100)",
    hovermode="x unified",
    height=450,
    template="plotly_white",
    font=dict(family="Arial, sans-serif", size=12),
)

st.plotly_chart(fig_iv, use_container_width=True)

# ============================================================================
# Key Insights
# ============================================================================

st.markdown("### Key Insights")

latest_su = latest["su_ratio"]
latest_price_real = latest["price_real"]

# Calculate 5-year average for comparison
recent = results.tail(5)
avg_iv_5yr = recent["IV"].mean()
iv_change = current_iv - 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Current S/U Ratio",
        value=f"{latest_su:.3f}",
        delta=f"Supply/Usage balance",
        delta_color="off",
    )

with col2:
    st.metric(
        label="5-Year Avg IV",
        value=f"{avg_iv_5yr:.1f}",
        delta=f"{iv_change:+.1f} from average",
    )

with col3:
    st.metric(
        label="Real Price",
        value=f"${latest_price_real:.2f}/bu",
        delta=f"{current_year} (2025 dollars)",
        delta_color="off",
    )

# ============================================================================
# Data Table
# ============================================================================

st.markdown("### Full Historical Data")

display_cols = ["year", "supply", "usage", "su_ratio", "G", "price_real", "IV"]
display_data = results[display_cols].copy()
display_data = display_data.rename(columns={
    "year": "Year",
    "supply": "Supply (Mbu)",
    "usage": "Usage (Mbu)",
    "su_ratio": "S/U",
    "G": "G Index",
    "price_real": "Real Price ($/bu)",
    "IV": "IV",
})
display_data["Year"] = display_data["Year"].astype(int)

st.dataframe(
    display_data.sort_values("Year", ascending=False),
    use_container_width=True,
    height=400,
)
