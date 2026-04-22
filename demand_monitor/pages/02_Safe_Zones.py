"""
Safe Zones page — historical safe zone visualization with supply/usage ellipse chart.

Extracts the scenario_ellipse_chart visualization from the original app.py.
Accesses crop choice and scenario parameters via session_state (set in app.py sidebar).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app import (
    # Data & results
    corn_df, corn_res, soy_df, soy_res,
    # Parameters
    CORN_ELASTICITY, SOYBEAN_ELASTICITY,
    BASE_YEAR,
    # Helper functions
    scenario_ellipse_chart,
    # Color palette
    AEI,
)

st.markdown("### 🎯 Safe Zones — Market Balance Visualization")
st.caption(
    "Visualize your scenario (★) within historical market balance zones. "
    "Green = normal range; yellow = unusual; red = no historical precedent. "
    "Shaded ovals show 1σ and 2σ of the historical supply/usage relationship. "
    "Dotted lines are iso-price contours (same model price along each line). "
    "**Click any historical point (year label) to update your scenario.**"
)

# Get current crop from session state
scen_crop = st.session_state.get("scen_crop", "Corn")

# Select data based on crop
if scen_crop == "Corn":
    results = corn_res
    elasticity = CORN_ELASTICITY
    supply_key = "supply_Corn"
    usage_key = "usage_Corn"
    crop_label = "Corn"
else:
    results = soy_res
    elasticity = SOYBEAN_ELASTICITY
    supply_key = "supply_Soybeans"
    usage_key = "usage_Soybeans"
    crop_label = "Soybeans"

# Retrieve scenario values from session state
scen_supply = st.session_state.get(supply_key, float(results.iloc[-1]["supply"]))
scen_usage = st.session_state.get(usage_key, float(results.iloc[-1]["usage"]))
ols_pred = st.session_state.get("_cached_ols_pred", None)

# Use a default price if not available
if ols_pred is None:
    ols_pred = float(results.iloc[-1]["pred_price"])

# Render the scenario ellipse chart
st.plotly_chart(
    scenario_ellipse_chart(
        results, scen_supply, scen_usage, ols_pred, elasticity
    ),
    use_container_width=True,
    on_select="rerun",
    key=f"scatter_{crop_label}",
)

# Handle chart interaction: user clicked a historical point
if st.session_state.get(f"scatter_{crop_label}_selection", None):
    scatter_event = st.session_state[f"scatter_{crop_label}_selection"]
    if scatter_event and scatter_event.get("points"):
        pt = scatter_event["points"][0]
        clicked_s = pt.get("x")
        clicked_u = pt.get("y")
        if clicked_s is not None and clicked_u is not None:
            st.session_state[supply_key] = float(clicked_s)
            st.session_state[usage_key] = float(clicked_u)
            st.rerun()

st.divider()
st.markdown("### 📊 Historical Market Balance Reference (S/U Ratio Guide)")
st.caption(
    "The supply-to-usage (S/U) ratio is the single most important number in this model. "
    "It measures how much total supply there is relative to total usage. "
    "Over the past 25 years of markets, this ratio has stayed in a narrow band — "
    "use the table below to put any scenario in context."
)

# Import the table rendering function
from app import render_guardrail_table

render_guardrail_table(crop_label)
