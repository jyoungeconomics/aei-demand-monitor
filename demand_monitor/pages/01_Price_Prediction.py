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
from app import (
    # Data & results
    corn_df, corn_res, soy_df, soy_res,
    # Parameters
    CORN_ELASTICITY, SOYBEAN_ELASTICITY,
    CORN_OLS_INTERCEPT, CORN_OLS_SLOPE,
    SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE,
    BASE_YEAR,
    # Helper functions
    render_tab,
    # Color palette
    AEI,
)

st.markdown("### 📈 Price Prediction & Decomposition")
st.caption(
    "Year-by-year breakdown of how supply and usage shifts drove price changes. "
    "The Shapley decomposition shows the marginal impact of each driver. "
    "Your scenario (if entered) appears as the last row."
)

# Get current crop from session state (set in app.py)
scen_crop = st.session_state.get("scen_crop", "Corn")

# Get scenario parameters from session state (set in sidebar of app.py)
if scen_crop == "Corn":
    results = corn_res
    elasticity = CORN_ELASTICITY
    b0, b1 = CORN_OLS_INTERCEPT, CORN_OLS_SLOPE
    supply_key = "supply_Corn"
    usage_key = "usage_Corn"
    spot_key = "spot_price_Corn"
    scen_row = st.session_state.get("_corn_scen_row", None)
else:
    results = soy_res
    elasticity = SOYBEAN_ELASTICITY
    b0, b1 = SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE
    supply_key = "supply_Soybeans"
    usage_key = "usage_Soybeans"
    spot_key = "spot_price_Soybeans"
    scen_row = st.session_state.get("_soy_scen_row", None)

# Retrieve scenario values from session state
scen_supply = st.session_state.get(supply_key, float(results.iloc[-1]["supply"]))
scen_usage = st.session_state.get(usage_key, float(results.iloc[-1]["usage"]))
ols_pred = st.session_state.get("_cached_ols_pred", None)
spot_price = st.session_state.get(spot_key, float(results.iloc[-1]["price_real"]))

# Render the tab (which contains all charts and tables)
render_tab(
    scen_crop,
    results,
    elasticity,
    scenario_supply=scen_supply,
    scenario_usage=scen_usage,
    scenario_price=ols_pred,
    spot_price_=spot_price,
    scenario_row=scen_row,
)
