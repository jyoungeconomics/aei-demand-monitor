"""
Implied Balance page — scenarios table showing S/U ratio, supply, and usage combinations.

Extracted from original app.py lines ~1611-1637 (the inversion feature section).
Adapted to use session_state and display on its own dedicated page.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from app import (
    # Data & results
    corn_df, corn_res, soy_df, soy_res,
    # Parameters
    CORN_ELASTICITY, SOYBEAN_ELASTICITY,
    BASE_YEAR,
    # Helper functions
    _safe_compute_g,
    AEI,
)

st.markdown("### 📋 Implied Balance Scenarios")
st.caption(
    "Based on your spot price input (from the sidebar), this table shows what supply and usage "
    "combinations would be needed to maintain a specific S/U ratio and thus support your entered price. "
    "This is the inverse of the model: instead of 'given S and U, what price?', "
    "it answers 'given a price, what S/U ratio must be true?'"
)

# Get current crop from session state
scen_crop = st.session_state.get("scen_crop", "Corn")

# Select data and parameters based on crop
if scen_crop == "Corn":
    results = corn_res
    elasticity = CORN_ELASTICITY
    b0, b1 = corn_res.iloc[-1].get("ols_intercept"), corn_res.iloc[-1].get("ols_slope")
    # Fallback to module constants if not in results
    from app import CORN_OLS_INTERCEPT, CORN_OLS_SLOPE
    b0, b1 = CORN_OLS_INTERCEPT, CORN_OLS_SLOPE
    spot_key = "spot_price_Corn"
    crop_name = "Corn"
else:
    results = soy_res
    elasticity = SOYBEAN_ELASTICITY
    from app import SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE
    b0, b1 = SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE
    spot_key = "spot_price_Soybeans"
    crop_name = "Soybeans"

# Get spot price from session state (set in sidebar)
spot_price = st.session_state.get(spot_key, float(results.iloc[-1]["price_real"]))

# Get base year reference values
base_row = results[results["year"] == BASE_YEAR]
if base_row.empty:
    base_row = results.iloc[0]
else:
    base_row = base_row.iloc[0]

G_base_val = _safe_compute_g(float(base_row["supply"]), float(base_row["usage"]), elasticity)
P_base_val = float(base_row["price_real"])

# Compute the S/U ratio implied by the spot price
if G_base_val is not None and G_base_val > 0:
    try:
        G_spot_implied = G_base_val * (spot_price / P_base_val)
        su_spot_implied = G_spot_implied ** elasticity

        st.markdown(f"**Your entered price:** ${spot_price:.2f}/bu")
        st.markdown(
            f"**Implied S/U ratio:** {su_spot_implied:.3f} "
            f"(historical range: 1.07–1.19)"
        )

        # Get the most recent actual values for building scenarios
        last_row = results.iloc[-1]
        last_supply = float(last_row["supply"])
        last_usage = float(last_row["usage"])

        # Define likelihood function
        def _likelihood(u_delta):
            if u_delta == 0:
                return "Most likely"
            elif abs(u_delta) <= 100:
                return "Possible"
            else:
                return "Less likely"

        # Build scenario table
        scenarios = []
        usage_changes = [-200, -100, 0, 100, 200]
        for u_delta in usage_changes:
            u_new = last_usage + u_delta
            if u_new > 0:
                s_new = u_new * su_spot_implied
                scenarios.append({
                    "Usage Change": f"{u_delta:+.0f} mil. bu" if u_delta != 0 else "No change",
                    "Probability": _likelihood(u_delta),
                    "Implied Supply": f"{s_new:,.0f}",
                    "Implied Usage": f"{u_new:,.0f}",
                    "S/U Ratio": f"{su_spot_implied:.3f}",
                })

        if scenarios:
            st.markdown(
                f"##### Example (Supply, Usage) scenarios to achieve this price"
            )
            st.caption(
                f"If usage changes relative to last year ({int(last_usage):,} mil. bu), "
                f"here's what supply would need to be to maintain a {su_spot_implied:.3f} ratio "
                f"(and thus a ${spot_price:.2f}/bu price):"
            )
            scenario_df = pd.DataFrame(scenarios)
            st.dataframe(scenario_df, use_container_width=True, hide_index=True)
        else:
            st.info("No valid scenarios could be generated.")

    except Exception as e:
        st.warning(f"Could not compute implied balance scenarios: {str(e)}")
else:
    st.warning(
        "Cannot compute implied balance — base year values are invalid. "
        "Check supply and usage data."
    )

st.divider()
st.markdown("### How This Works")
st.markdown("""
The **constant-elasticity demand model** says that price is a function of the supply-to-usage ratio:

**G = (S/U)^(1/ε)**

Where:
- **G** is the price ratio (predicted price ÷ base-year price)
- **S** is total supply
- **U** is total usage
- **ε** is the demand elasticity (negative; here ~−0.17 for corn)

**Inverting the model:** Given a price and elasticity, you can solve for the S/U ratio that must be true:

**S/U = G^ε = (Price_base / Price_target)^ε**

Once you know the required S/U ratio, you can ask: "If usage changes by X, how much must supply change
to maintain this ratio?" That's what this table shows. It helps you understand what supply and demand
conditions are consistent with your price view.
""")
