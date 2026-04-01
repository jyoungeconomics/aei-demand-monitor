"""
Corn & Soybean Demand Monitor — Streamlit dashboard.

Run with:
    python -m streamlit run demand_monitor/app.py
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

try:
    import requests
except ImportError:
    requests = None

try:
    import yfinance as yf
except ImportError:
    yf = None

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
    "green":      "#609D42",
    "navy":       "#001425",
    "orange":     "#F28C28",
    "gray":       "#A7A9AC",
    "teal":       "#3B8B8B",
    "red":        "#C0504D",
    "dark_green": "#508428",
    "yellow":     "#E8A020",
}

# ---------------------------------------------------------------------------
# Guardrail thresholds (S/U ratio bands, from technical paper section 4.2)
# ---------------------------------------------------------------------------
# These thresholds apply to the supply-to-usage ratio.
# A ratio between 1.07 and 1.19 covers the full 25-year historical record.
# Outside that range, predictions become extrapolations.

_GUARDRAIL_BANDS = [
    # (su_lo, su_hi, status_key, display_label)
    (None,  1.02,  "danger",   "Danger — no historical precedent"),
    (1.02,  1.07,  "warning",  "Warning — tighter than 2012 drought"),
    (1.07,  1.19,  "plausible","Plausible — within historical range"),
    (1.19,  1.26,  "warning",  "Warning — looser than any year on record"),
    (1.26,  None,  "danger",   "Danger — no historical precedent"),
]

_STATUS_COLOR = {
    "plausible": AEI["green"],
    "warning":   AEI["yellow"],
    "danger":    AEI["red"],
}

_STATUS_FILL = {
    "plausible": "rgba(96,157,66,0.13)",
    "warning":   "rgba(232,160,32,0.13)",
    "danger":    "rgba(192,80,77,0.13)",
}

_STATUS_MESSAGE = {
    "plausible": (
        "Your supply and usage inputs are within the range this model has seen "
        "over the past 25 years. The model-implied price reflects a plausible outcome."
    ),
    "warning": (
        "The supply/usage balance you've entered is tighter or looser than anything "
        "observed in modern markets. Treat the implied price as an extrapolation — "
        "check whether your supply and usage figures are internally consistent."
    ),
    "danger": (
        "Your inputs imply a supply/usage balance with no historical precedent. "
        "The implied price may not be reliable. Total supply is typically 7–20% "
        "larger than total usage across all years on record."
    ),
}

# Reference tables for corn and soybeans (computed from section 4.2)
_GUARDRAIL_REF_CORN = pd.DataFrame([
    {"S/U Ratio": "> 1.26",    "Market Condition": "Extremely loose",  "Status": "Danger",
     "Price Signal": "Below $2.50/bu",   "What It Means": "More supply than any year in modern records"},
    {"S/U Ratio": "1.19–1.26", "Market Condition": "Loose",            "Status": "Warning",
     "Price Signal": "$2.50–$3.50/bu",   "What It Means": "Looser than all 25 years of history"},
    {"S/U Ratio": "1.07–1.19", "Market Condition": "Normal",           "Status": "Plausible",
     "Price Signal": "$3.50–$9.50/bu",   "What It Means": "Within the full 2000–2025 historical band"},
    {"S/U Ratio": "1.02–1.07", "Market Condition": "Tight",            "Status": "Warning",
     "Price Signal": "$9.50–$11.00/bu",  "What It Means": "Only the 2012 drought came close to this"},
    {"S/U Ratio": "< 1.02",    "Market Condition": "Extremely tight",  "Status": "Danger",
     "Price Signal": "Above $11.00/bu",  "What It Means": "No historical precedent — supply nearly equal to usage"},
])

def _compute_soybean_guardrail_table():
    """Compute soybean price ranges for each S/U ratio band."""
    hist = soy_res.dropna(subset=["su_ratio", "price_real"])
    su_vals = hist["su_ratio"].values
    price_vals = hist["price_real"].values

    def price_range_for_su(su_lo, su_hi):
        """Find price range when S/U is between su_lo and su_hi."""
        if su_lo is None:
            mask = su_vals <= su_hi
        elif su_hi is None:
            mask = su_vals >= su_lo
        else:
            mask = (su_vals >= su_lo - 0.01) & (su_vals <= su_hi + 0.01)

        if mask.any():
            prices = price_vals[mask]
            return f"${prices.min():.1f}–${prices.max():.1f}/bu"
        return "–"

    return pd.DataFrame([
        {"S/U Ratio": "> 1.26",    "Market Condition": "Extremely loose",  "Status": "Danger",
         "Price Signal": price_range_for_su(1.26, None),   "What It Means": "More supply than any year in modern records"},
        {"S/U Ratio": "1.19–1.26", "Market Condition": "Loose",            "Status": "Warning",
         "Price Signal": price_range_for_su(1.19, 1.26),   "What It Means": "Looser than all 25 years of history"},
        {"S/U Ratio": "1.07–1.19", "Market Condition": "Normal",           "Status": "Plausible",
         "Price Signal": price_range_for_su(1.07, 1.19),   "What It Means": "Within the full 2000–2025 historical band"},
        {"S/U Ratio": "1.02–1.07", "Market Condition": "Tight",            "Status": "Warning",
         "Price Signal": price_range_for_su(1.02, 1.07),   "What It Means": "Only the 2012 drought came close to this"},
        {"S/U Ratio": "< 1.02",    "Market Condition": "Extremely tight",  "Status": "Danger",
         "Price Signal": price_range_for_su(None, 1.02),   "What It Means": "No historical precedent — supply nearly equal to usage"},
    ])


def get_guardrail_status(su_ratio: float) -> tuple[str, str, str, str]:
    """
    Return (status_key, hex_color, display_label, message) for a given S/U ratio.
    status_key is 'plausible', 'warning', or 'danger'.
    """
    for lo, hi, key, label in _GUARDRAIL_BANDS:
        lo_ok = (lo is None) or (su_ratio >= lo)
        hi_ok = (hi is None) or (su_ratio <  hi)
        if lo_ok and hi_ok:
            return key, _STATUS_COLOR[key], label, _STATUS_MESSAGE[key]
    # fallback
    return "danger", AEI["red"], "Danger — outside all historical bounds", _STATUS_MESSAGE["danger"]


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
      [data-testid="stAppViewContainer"] > .main {{ padding-top: 0.5rem; }}
      header[data-testid="stHeader"] {{
          background: {AEI["navy"]};
          height: auto;
          min-height: 2rem;
      }}
      /* Ensure sidebar is accessible - restore collapse button and normal behavior */
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
      .guardrail-box {{
          border-radius: 4px;
          padding: 0.6rem 0.85rem;
          margin: 0.4rem 0;
          font-size: 0.83rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with logo and title
col1, col2 = st.columns([1, 6])
with col1:
    # Try to find and display AEI logo from multiple possible paths
    logo_found = False
    for logo_candidate in [
        os.path.join(os.path.dirname(__file__), "aei_logo.png"),
        os.path.join(os.getcwd(), "demand_monitor", "aei_logo.png"),
        os.path.join(os.getcwd(), "aei_logo.png"),
        os.path.join(os.path.dirname(__file__), "aei_watermark.png"),
        os.path.join(os.getcwd(), "demand_monitor", "aei_watermark.png"),
    ]:
        if os.path.exists(logo_candidate):
            try:
                st.image(logo_candidate, width=150)
                logo_found = True
                break
            except Exception:
                pass
    if not logo_found:
        st.markdown("# AEI", help="Ag Economic Insights")

with col2:
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
    Both widgets are always clamped to [min_val, max_val].
    """
    # Initialize and clamp main value
    if ss_key not in st.session_state:
        st.session_state[ss_key] = float(init_val)

    try:
        st.session_state[ss_key] = float(
            np.clip(st.session_state[ss_key], min_val, max_val)
        )
    except (ValueError, TypeError):
        st.session_state[ss_key] = float(init_val)

    _sldr = f"_sldr_{ss_key}"
    _num  = f"_num_{ss_key}"

    # Ensure slider and input state are initialized
    if _sldr not in st.session_state:
        st.session_state[_sldr] = int(st.session_state[ss_key])
    if _num not in st.session_state:
        st.session_state[_num] = st.session_state[ss_key]

    def _from_slider():
        try:
            val = float(st.session_state[_sldr])
            val = float(np.clip(val, min_val, max_val))
            st.session_state[ss_key] = val
            # Explicitly sync the number input to avoid desync on slider move
            st.session_state[_num] = val
        except Exception:
            pass

    def _from_input():
        try:
            val = float(st.session_state[_num])
            val = float(np.clip(val, min_val, max_val))
            st.session_state[ss_key] = val
            st.session_state[_sldr] = int(val)
        except Exception:
            pass

    st.markdown(f"**{label}**")

    # Slider: use clamped value
    slider_val = int(np.clip(st.session_state[ss_key], min_val, max_val))
    st.slider(
        "", min_value=int(min_val), max_value=int(max_val),
        value=slider_val, step=int(step),
        key=_sldr, on_change=_from_slider, label_visibility="collapsed",
    )

    # Number input: use clamped value
    input_val = float(np.clip(st.session_state[ss_key], min_val, max_val))
    st.number_input(
        "", min_value=float(min_val), max_value=float(max_val),
        value=input_val, step=float(step), format=fmt,
        key=_num, on_change=_from_input, label_visibility="collapsed",
    )
    return float(st.session_state[ss_key])


# ---------------------------------------------------------------------------
# Safe wrapper for compute_g (prevents crashes on extreme inputs)
# ---------------------------------------------------------------------------

def _safe_compute_g(supply: float, usage: float, elasticity: float) -> float | None:
    """Return compute_g result, or None if supply/usage are invalid."""
    if usage <= 0 or supply <= 0:
        return None
    try:
        result = compute_g(supply, usage, elasticity)
        if not math.isfinite(result):
            return None
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_futures_price_from_barchart(contract_symbol: str) -> float | None:
    """
    Fetch current futures price from Yahoo Finance via yfinance.
    contract_symbol: e.g., "ZCZ26" for Dec corn or "ZSX26" for Nov soybeans.
    Returns the last price in $/bu, or None if fetch fails.

    yfinance returns CBOT prices in cents/bu, so we divide by 100 to get $/bu.
    """
    if not yf:
        return None
    try:
        # Map to Yahoo Finance tickers
        # ZCZ26 (Dec corn) -> ZC=F (corn futures generic)
        # ZSX26 (Nov soybeans) -> ZS=F (soybeans futures generic)
        if contract_symbol.startswith("ZC"):
            ticker = "ZC=F"  # Corn futures
        else:
            ticker = "ZS=F"  # Soybeans futures

        # Fetch data with appropriate timeout for cloud deployment
        data = yf.Ticker(ticker)
        hist = data.history(period="1d", timeout=15)

        if hist is not None and not hist.empty and "Close" in hist.columns:
            # yfinance returns prices in cents/bu for CBOT contracts
            price_cents = float(hist["Close"].iloc[-1])
            if price_cents > 0:
                price_dollars = price_cents / 100.0
                return price_dollars
    except Exception as e:
        # Fail gracefully — will use fallback price
        pass
    return None


# ---------------------------------------------------------------------------
# Sidebar — Scenario / "What-If" Tool
# ---------------------------------------------------------------------------

# Crop selection (will be displayed in main area, accessed here via session_state)
if "scen_crop" not in st.session_state:
    st.session_state.scen_crop = "Corn"
scen_crop = st.session_state.scen_crop

with st.sidebar:
    st.markdown("### 🔧 What-If Scenario")
    st.caption(
        "Enter your own supply, usage, and spot price to see "
        "what the model implies. Select crop in the main area above."
    )

    if scen_crop == "Corn":
        ref       = corn_res.iloc[-1]
        e         = CORN_ELASTICITY
        b0, b1    = CORN_OLS_INTERCEPT, CORN_OLS_SLOPE
        df_full   = corn_df
        res_full  = corn_res
        _all_s    = corn_df["supply"].dropna()
        _all_u    = corn_df["usage"].dropna()
    else:
        ref       = soy_res.iloc[-1]
        e         = SOYBEAN_ELASTICITY
        b0, b1    = SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE
        df_full   = soy_df
        res_full  = soy_res
        _all_s    = soy_df["supply"].dropna()
        _all_u    = soy_df["usage"].dropna()

    s_lo = math.floor(_all_s.min() * 0.80 / 100) * 100
    s_hi = math.ceil( _all_s.max() * 1.20 / 100) * 100
    u_lo = math.floor(_all_u.min() * 0.80 / 100) * 100
    u_hi = math.ceil( _all_u.max() * 1.20 / 100) * 100

    default_s = float(round(ref["supply"]))
    default_u = float(round(ref["usage"]))

    scen_supply = _synced_input(
        "Total Supply (mil. bu)",
        f"supply_{scen_crop}", s_lo, s_hi, 50, default_s,
    )
    scen_usage = _synced_input(
        "Total Usage (mil. bu)",
        f"usage_{scen_crop}", u_lo, u_hi, 50, default_u,
    )

    # Spot price input — e.g., December futures or current cash price
    st.markdown("**Spot / Futures Price ($/bu)**")
    last_actual = float(ref["price_real"])
    spot_key    = f"spot_price_{scen_crop}"

    # Try to fetch current futures price from Yahoo Finance
    contract_symbol = "ZCZ26" if scen_crop == "Corn" else "ZSX26"
    current_price = _get_futures_price_from_barchart(contract_symbol)

    # Use fetched price if available, otherwise fall back to last actual
    if current_price is not None:
        # Clamp to valid range (0.50–50.00) in case of bad data
        default_spot = round(np.clip(current_price, 0.50, 50.00), 2)
        st.caption(f"✅ Current {scen_crop} futures ({contract_symbol}) from Yahoo Finance")
        # Always update session state with newly fetched price
        st.session_state[spot_key] = default_spot
    elif spot_key not in st.session_state:
        default_spot = round(last_actual, 2)
        st.session_state[spot_key] = default_spot
        st.caption(f"📊 Using {int(ref['year'])} actual price (live futures unavailable)")
    else:
        default_spot = st.session_state[spot_key]
        st.caption(f"📊 Using session price (live futures unavailable)")

    spot_price = st.number_input(
        "",
        min_value=0.50,
        max_value=50.00,
        value=float(st.session_state[spot_key]),
        step=0.05,
        format="%.2f",
        key=f"_spot_num_{scen_crop}",
        label_visibility="collapsed",
        help="Enter the current cash price, Dec/Nov futures, or your own price expectation. "
             "The model compares this market signal to its supply/usage prediction.",
    )
    st.session_state[spot_key] = spot_price

    # Model-implied price (ratio method)
    G_scen    = _safe_compute_g(scen_supply, scen_usage, e)
    base_row  = df_full[df_full["year"] == BASE_YEAR].iloc[0]
    G_base_val = _safe_compute_g(float(base_row["supply"]), float(base_row["usage"]), e)
    P_base_val = float(df_full[df_full["year"] == BASE_YEAR]["price_real"].values[0])

    if G_scen is not None and G_base_val is not None and G_base_val > 0:
        _rp = P_base_val * (G_scen / G_base_val)
        # Sanity-clamp: prices outside $0.25–$100/bu are not meaningful
        ratio_pred = _rp if (math.isfinite(_rp) and 0.25 <= _rp <= 100.0) else None
    else:
        ratio_pred = None

    # S/U ratio and guardrail status
    su_scen = scen_supply / max(scen_usage, 1.0)
    grd_status, grd_color, grd_label, grd_msg = get_guardrail_status(su_scen)

    st.markdown("---")

    # Model price display
    if ratio_pred is not None:
        st.markdown("**Model-implied price**")
        st.markdown(
            f'<div class="scenario-box">'
            f'<span style="font-size:1.4rem; font-weight:700; color:{AEI["navy"]}">'
            f"${ratio_pred:.2f}<small>/bu</small></span><br>"
            f'<span style="color:{AEI["gray"]}; font-size:0.78rem;">'
            f"G index: {G_scen:.4f} &nbsp;|&nbsp; S/U ratio: {su_scen:.3f}"
            f"</span></div>",
            unsafe_allow_html=True,
        )
        # Spot vs model comparison
        delta_model   = ratio_pred - spot_price
        delta_actual  = ratio_pred - last_actual
        sign_m  = "+" if delta_model  >= 0 else ""
        sign_a  = "+" if delta_actual >= 0 else ""
        col_m   = AEI["green"] if delta_model  >= 0 else AEI["red"]
        col_a   = AEI["green"] if delta_actual >= 0 else AEI["red"]
        st.markdown(
            f'<span style="color:{col_m}; font-weight:600; font-size:0.85rem;">'
            f"{sign_m}${delta_model:.2f} &nbsp; vs your spot price (${spot_price:.2f})"
            f"</span><br>"
            f'<span style="color:{col_a}; font-weight:600; font-size:0.85rem;">'
            f"{sign_a}${delta_actual:.2f} &nbsp; vs last actual ({int(ref['year'])})"
            f"</span>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Cannot compute model price — check supply and usage values.")

    # Guardrail badge
    st.markdown("---")
    st.markdown("**Market Balance Check**")
    st.markdown(
        f'<div class="guardrail-box" style="background:{grd_color}18; '
        f'border-left:4px solid {grd_color};">'
        f'<span style="font-weight:700; color:{grd_color}; font-size:0.88rem;">'
        f'{grd_label}</span><br>'
        f'<span style="color:{AEI["navy"]}; font-size:0.78rem;">{grd_msg}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span style="font-size:0.78rem; color:{AEI["gray"]};">'
        f"Supply is {su_scen:.1%} of usage &nbsp;(historical range: 107%–119%)"
        f"</span>",
        unsafe_allow_html=True,
    )

    # Inversion feature: show what S/U ratio the spot price implies, plus example S & U pairs
    if G_base_val is not None and G_base_val > 0:
        try:
            G_spot_implied = G_base_val * (spot_price / P_base_val)
            su_spot_implied = G_spot_implied ** e
            st.markdown("---")
            st.markdown("💡 **Market's Implied Balance**")
            st.markdown(
                f"The market is pricing {scen_crop.lower()} at ${spot_price:.2f}/bu, "
                f"which implies a supply/usage ratio of **{su_spot_implied:.3f}**. "
                f"(Historical range: 1.07–1.19)"
            )

            # Show example (S, U) pairs that would hit this ratio
            last_supply = float(ref["supply"])
            last_usage = float(ref["usage"])

            # Assign relative likelihood: closer to 0 change is more likely
            def _likelihood(u_delta):
                if u_delta == 0:
                    return "Most likely"
                elif abs(u_delta) <= 100:
                    return "Possible"
                else:
                    return "Less likely"

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
                with st.expander("📋 Example (Supply, Usage) scenarios to achieve this price"):
                    st.caption(
                        f"If usage changes relative to last year ({int(last_usage):,} mil. bu), "
                        f"here's what supply would need to be to maintain a {su_spot_implied:.3f} ratio "
                        f"(and thus a ${spot_price:.2f}/bu price):"
                    )
                    scenario_df = pd.DataFrame(scenarios)
                    st.dataframe(scenario_df, use_container_width=True, hide_index=True)
        except Exception:
            pass



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
    model_price: float,
    spot_price_: float,
    elasticity: float,
    b0_: float,
    b1_: float,
) -> dict | None:
    """
    Compute supply/usage Shapley impacts for the hypothetical scenario year.
    model_price  : ratio-method model prediction (shown in "Model Price" column)
    spot_price_  : user-entered cash/futures price (shown in "Actual Price" column)
    Returns a dict with farmer_table keys, or None if inputs are invalid.
    """
    prev   = results.iloc[-1]
    S_tm1  = float(prev["supply"])
    U_tm1  = float(prev["usage"])
    P_prev = float(prev["price_real"])
    y_tm1  = float(prev["pred_price"])

    base_rows = results[results["year"] == BASE_YEAR]
    base_r    = base_rows.iloc[0] if not base_rows.empty else results.iloc[0]
    G_base_r  = _safe_compute_g(float(base_r["supply"]), float(base_r["usage"]), elasticity)
    if G_base_r is None or G_base_r <= 0:
        return None
    K = 100.0 / (G_base_r * float(base_r["price_real"]))

    G_t  = _safe_compute_g(scenario_supply, scenario_usage, elasticity)
    GD   = _safe_compute_g(S_tm1, scenario_usage, elasticity)
    GS   = _safe_compute_g(scenario_supply, U_tm1, elasticity)
    if any(v is None for v in [G_t, GD, GS]):
        return None

    # Use spot price as the "P" anchor — this is what the market is pricing in.
    y_t  = b0_ + b1_ * K * spot_price_ * G_t
    yD   = b0_ + b1_ * K * spot_price_ * GD
    yS   = b0_ + b1_ * K * spot_price_ * GS

    dy        = y_t - y_tm1
    dPrice    = spot_price_ - P_prev
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
        "Actual Price ($/bu)":         round(spot_price_, 2),
        "Model Price ($/bu)":          round(model_price, 2),
        "Change vs Prior Year ($/bu)": round(dPrice, 2),
        "Supply Drove ($/bu)":         round(supply_impact, 2),
        "Usage Drove ($/bu)":          round(demand_impact, 2),
    }


# Pre-compute scenario rows
if ratio_pred is not None:
    if scen_crop == "Corn":
        _corn_scen_row = compute_scenario_row(
            corn_res, scen_supply, scen_usage, ratio_pred, spot_price,
            CORN_ELASTICITY, CORN_OLS_INTERCEPT, CORN_OLS_SLOPE,
        )
        _soy_scen_row = None
    else:
        _corn_scen_row = None
        _soy_scen_row = compute_scenario_row(
            soy_res, scen_supply, scen_usage, ratio_pred, spot_price,
            SOYBEAN_ELASTICITY, SOYBEAN_OLS_INTERCEPT, SOYBEAN_OLS_SLOPE,
        )
else:
    _corn_scen_row = None
    _soy_scen_row  = None


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
        line=dict(color="#000000", width=2, dash="dash"),
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
    """Bar chart: year-over-year price change decomposed into supply and usage effects."""
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
      - Guardrail region shading (green/yellow/red) by S/U ratio band.
      - Iso-price contour lines (dotted).
      - 1σ / 2σ confidence ellipses.
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

    def make_ellipse(n_sigma, scale=1.0):
        """Create an ellipse, optionally scaled to account for outliers."""
        scaled = eigenvectors @ np.diag(n_sigma * scale * np.sqrt(eigenvalues)) @ circle
        return scaled[0] + mean_s, scaled[1] + mean_u

    # Scale up ellipses slightly (1.35x) to account for data outliers and improve visibility
    x_2sig, y_2sig = make_ellipse(2, scale=1.35)
    x_1sig, y_1sig = make_ellipse(1, scale=1.35)

    # Iso-price lines: U = S / r_target
    base_rows = hist[hist["year"] == BASE_YEAR]
    base_r    = base_rows.iloc[0] if not base_rows.empty else hist.iloc[0]
    r_base    = float(base_r["supply"]) / float(base_r["usage"])
    P_base    = float(base_r["price_real"])

    # Axis bounds
    all_s = np.append(supply_vals, scenario_supply)
    all_u = np.append(usage_vals,  scenario_usage)
    s_pad = (supply_vals.max() - supply_vals.min()) * 0.04
    u_pad = (usage_vals.max()  - usage_vals.min())  * 0.04
    x_min = all_s.min() - s_pad
    x_max = all_s.max() + s_pad
    y_min = all_u.min() - u_pad
    y_max = all_u.max() + u_pad

    # Extend slightly for fill coverage, but clip data traces to actual bounds
    x_ext = x_max * 1.08
    y_ext = y_max * 1.08

    fig = go.Figure()

    # ----------------------------------------------------------------
    # Guardrail region fills (background layer, added first)
    # Regions defined by S/U ratio bands: U = S/r defines the boundary.
    # A point has S/U = r when it sits exactly on the line U = S/r.
    # S/U < 1.02 → above the line U = S/1.02 → tight market, red
    # S/U > 1.26 → below the line U = S/1.26 → loose market, red
    # ----------------------------------------------------------------

    def _fill_band(r_lo, r_hi, fillcolor, legend_name, show_legend):
        """
        Fill the wedge between S/U=r_lo (upper) and S/U=r_hi (lower).
        r_lo=None means extend to top; r_hi=None means extend to bottom.
        """
        top_l = (x_min / r_lo) if r_lo is not None else y_ext
        top_r = (x_ext / r_lo) if r_lo is not None else y_ext
        bot_l = (x_min / r_hi) if r_hi is not None else y_min * 0.95
        bot_r = (x_ext / r_hi) if r_hi is not None else y_min * 0.95
        fig.add_scatter(
            x=[x_min, x_ext, x_ext, x_min],
            y=[top_l, top_r, bot_r, bot_l],
            fill="toself",
            fillcolor=fillcolor,
            line=dict(width=0),
            name=legend_name,
            showlegend=show_legend,
            hoverinfo="skip",
            legendgroup="guardrail",
            legendgrouptitle_text="Market Balance" if show_legend else None,
        )

    # Danger (tight): S/U < 1.02 — fill above U = S/1.02
    _fill_band(None, 1.02, _STATUS_FILL["danger"],   "Danger zone",    True)
    # Warning (tight): S/U 1.02–1.07
    _fill_band(1.02, 1.07, _STATUS_FILL["warning"],  "Warning zone",   True)
    # Plausible: S/U 1.07–1.19
    _fill_band(1.07, 1.19, _STATUS_FILL["plausible"],"Plausible zone", True)
    # Warning (loose): S/U 1.19–1.26 — share legend entry
    _fill_band(1.19, 1.26, _STATUS_FILL["warning"],  "Warning zone",   False)
    # Danger (loose): S/U > 1.26 — fill below U = S/1.26
    _fill_band(1.26, None, _STATUS_FILL["danger"],   "Danger zone",    False)

    # ----------------------------------------------------------------
    # Iso-price contour lines
    # ----------------------------------------------------------------
    prices_hist = hist["price_real"].values
    p_lo = np.floor(prices_hist.min() * 2) / 2
    p_hi = np.ceil( prices_hist.max() * 2) / 2
    price_levels = np.linspace(p_lo, p_hi, 6)
    s_line = np.array([x_min, x_max])

    for P_target in price_levels:
        if P_base <= 0:
            continue
        try:
            r_target = r_base * (P_target / P_base) ** elasticity
        except Exception:
            continue
        if r_target < 1e-6:
            continue
        u_line = s_line / r_target
        # Clip to axis range — prevents runaway lines from crashing the chart
        u_line = np.clip(u_line, y_min * 0.9, y_max * 1.1)
        if u_line.min() > y_max * 1.05 or u_line.max() < y_min * 0.95:
            continue
        fig.add_scatter(
            x=s_line, y=u_line,
            mode="lines",
            line=dict(color=AEI["gray"], width=1, dash="dot"),
            marker=dict(size=0),
            name=f"${P_target:.1f}/bu",
            legendgroup="iso",
            showlegend=True,
            hovertemplate=f"Iso-price: ${P_target:.1f}/bu<extra></extra>",
        )

    # ----------------------------------------------------------------
    # Confidence ellipses
    # ----------------------------------------------------------------
    fig.add_scatter(
        x=x_2sig, y=y_2sig,
        mode="lines",
        line=dict(color="rgba(96,157,66,0.45)", width=1.5, dash="dot"),
        fill="toself",
        fillcolor="rgba(96,157,66,0.08)",
        name="2σ region (95% historical)",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=x_1sig, y=y_1sig,
        mode="lines",
        line=dict(color="rgba(96,157,66,0.80)", width=1.5),
        fill="toself",
        fillcolor="rgba(96,157,66,0.15)",
        name="1σ region (68% historical)",
        hoverinfo="skip",
    )

    # ----------------------------------------------------------------
    # Historical data points
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Scenario star
    # ----------------------------------------------------------------
    scenario_label = _ellipse_region_label(
        scenario_supply, scenario_usage, mean_s, mean_u, eigenvalues, eigenvectors
    )
    su_label_str = f"S/U = {scenario_supply/max(scenario_usage,1):.3f}"
    fig.add_scatter(
        x=[scenario_supply], y=[scenario_usage],
        mode="markers+text",
        text=[f"Your scenario"],
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
            f"{su_label_str}<br>"
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
        height=500,
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
        label   = scenario_row["year_label"]
        scen_df = pd.DataFrame(
            [{k: v for k, v in scenario_row.items() if k != "year_label"}],
            index=[label],
        )
        scen_df.index.name = "Year"
        tbl = pd.concat([tbl, scen_df])

    return tbl


# ---------------------------------------------------------------------------
# Guardrail reference table display
# ---------------------------------------------------------------------------

def render_guardrail_table(crop_name: str) -> None:
    """Show the historical S/U ratio reference table in an expander."""
    with st.expander("📊 Historical Market Balance Reference (S/U Ratio Guide)", expanded=False):
        st.caption(
            "The supply-to-usage (S/U) ratio is the single most important number "
            "in this model. It measures how much total supply there is relative to "
            "total usage. Over the past 25 years of markets, this ratio has "
            "stayed in a narrow band — use the table below to put any scenario in context."
        )

        # Show crop-specific table
        if crop_name.lower() == "soybeans":
            tbl = _compute_soybean_guardrail_table()
            crop_label = "Soybeans"
        else:
            tbl = _GUARDRAIL_REF_CORN
            crop_label = "Corn"

        def _row_style(row):
            status = row["Status"]
            color  = _STATUS_COLOR.get(status.lower(), "#000000")
            return [f"background-color: {color}18; color: {AEI['navy']}" for _ in row]

        styled = (
            tbl.style
            .apply(_row_style, axis=1)
            .set_properties(**{"text-align": "left"})
            .hide(axis="index")
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.caption(
            f"Historical price ranges are actual {crop_label.lower()} prices from USDA "
            f"(2000–2025, real 2025 dollars). Source: WASDE & AEI Demand Model."
        )


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
    spot_price_: float | None = None,
    scenario_row: dict | None = None,
) -> None:
    check = verify_decomp(results)
    if not check["pass"]:
        st.warning(f"Decomposition check failed (max error = {check['max_resid_error']:.1e})")

    # ---- Key metric strip ----
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

    # ---- Historical S/U reference table ----
    render_guardrail_table(crop_label)

    # ---- Scenario supply/usage chart ----
    if scenario_supply is not None and scenario_usage is not None and scenario_price is not None:
        st.markdown("---")
        st.markdown("**Where Does Your Scenario Fall?** — Supply & Usage in Context")
        st.caption(
            "The shaded background shows market balance zones based on 25 years of history: "
            "green = normal range, yellow = unusual, red = no historical precedent. "
            "Dotted gray lines connect every (Supply, Usage) pair that implies the same model price. "
            "The ovals show 1σ and 2σ of the historical supply/usage relationship. "
            "The ★ marks your scenario."
        )
        st.plotly_chart(
            scenario_ellipse_chart(
                results, scenario_supply, scenario_usage, scenario_price, elasticity
            ),
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

# Crop selection radio button
col1, col2, col3 = st.columns([1, 2, 2])
with col1:
    def _on_crop_change():
        st.session_state.scen_crop = st.session_state._crop_radio
    st.radio(
        "Select Crop",
        ["Corn", "Soybeans"],
        key="_crop_radio",
        on_change=_on_crop_change,
        horizontal=True,
        label_visibility="collapsed",
    )

tab_corn, tab_soy = st.tabs(["🌽 Corn", "🫘 Soybeans"])

with tab_corn:
    render_tab(
        "corn", corn_res, CORN_ELASTICITY,
        scenario_supply = scen_supply  if scen_crop == "Corn"     else None,
        scenario_usage  = scen_usage   if scen_crop == "Corn"     else None,
        scenario_price  = ratio_pred   if scen_crop == "Corn"     else None,
        spot_price_     = spot_price   if scen_crop == "Corn"     else None,
        scenario_row    = _corn_scen_row,
    )

with tab_soy:
    render_tab(
        "soybeans", soy_res, SOYBEAN_ELASTICITY,
        scenario_supply = scen_supply  if scen_crop == "Soybeans" else None,
        scenario_usage  = scen_usage   if scen_crop == "Soybeans" else None,
        scenario_price  = ratio_pred   if scen_crop == "Soybeans" else None,
        spot_price_     = spot_price   if scen_crop == "Soybeans" else None,
        scenario_row    = _soy_scen_row,
    )
