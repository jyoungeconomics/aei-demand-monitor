"""
Price Prediction page — historical actual vs model prices with Shapley decomposition.

Extracts lines ~1450-1600 from the original app.py, adapted for multipage structure.
Accesses crop choice via st.session_state.scen_crop (set in app.py).
Imports shared helper functions and data from the main app module.
"""

import sys
import os

# Add parent directory to path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import math
import numpy as np
from app import (
    # Data & results
    corn_df, corn_res, soy_df, soy_res,
    # Parameters
    CORN_ELASTICITY, SOYBEAN_ELASTICITY,
    CORN_OLS_INTERCEPT, CORN_OLS_SLOPE,
    SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE,
    BASE_YEAR,
    # Helper functions
    render_tab, _synced_input,
    # Color palette
    AEI,
)

st.markdown("### 📈 Price Prediction & Decomposition")
st.caption(
    "Year-by-year breakdown of how supply and usage shifts drove price changes. "
    "The Shapley decomposition shows the marginal impact of each driver. "
    "Adjust supply and usage below to explore different scenarios — changes update the sidebar and all pages."
)

# Get current crop from session state (set in app.py)
scen_crop = st.session_state.get("scen_crop", "Corn")

# Get scenario parameters from session state (set in sidebar of app.py)
if scen_crop == "Corn":
    results = corn_res
    df_full = corn_df
    elasticity = CORN_ELASTICITY
    b0, b1 = CORN_OLS_INTERCEPT, CORN_OLS_SLOPE
    supply_key = "supply_Corn"
    usage_key = "usage_Corn"
    spot_key = "spot_price_Corn"
    scen_row = st.session_state.get("_corn_scen_row", None)
    _all_s = corn_df["supply"].dropna()
    _all_u = corn_df["usage"].dropna()
else:
    results = soy_res
    df_full = soy_df
    elasticity = SOYBEAN_ELASTICITY
    b0, b1 = SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE
    supply_key = "supply_Soybeans"
    usage_key = "usage_Soybeans"
    spot_key = "spot_price_Soybeans"
    scen_row = st.session_state.get("_soy_scen_row", None)
    _all_s = soy_df["supply"].dropna()
    _all_u = soy_df["usage"].dropna()

# Calculate slider ranges based on historical data
s_lo = math.floor(_all_s.min() * 0.80 / 100) * 100
s_hi = math.ceil(_all_s.max() * 1.20 / 100) * 100
u_lo = math.floor(_all_u.min() * 0.80 / 100) * 100
u_hi = math.ceil(_all_u.max() * 1.20 / 100) * 100

ref = results.iloc[-1]
default_s = float(round(ref["supply"]))
default_u = float(round(ref["usage"]))

# ============================================================================
# S/U Adjustment Panel (synced to sidebar)
# ============================================================================
st.markdown("### Adjust Your Scenario")

col1, col2 = st.columns(2)

with col1:
    scen_supply = _synced_input(
        "Total Supply (mil. bu)",
        supply_key, s_lo, s_hi, 50, default_s,
    )

with col2:
    scen_usage = _synced_input(
        "Total Usage (mil. bu)",
        usage_key, u_lo, u_hi, 50, default_u,
    )

ols_pred = st.session_state.get("_cached_ols_pred", None)
spot_price = st.session_state.get(spot_key, float(results.iloc[-1]["price_real"]))

st.divider()

# Render the tab WITHOUT price chart history (moved to Safe Zones page)
render_tab(
    scen_crop,
    results,
    elasticity,
    scenario_supply=scen_supply,
    scenario_usage=scen_usage,
    scenario_price=ols_pred,
    spot_price_=spot_price,
    scenario_row=scen_row,
    show_price_chart=False,
)
