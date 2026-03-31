"""
Shapley decomposition for the Corn & Soybean Demand Monitor.

For each marketing year, this module decomposes the change in the real MYA
price into two components:

  - Supply impact  : how much of the price change came from the supply side
  - Demand impact  : how much came from the demand side (total usage)

The algorithm mirrors the 'Shapley Decomposition' sheet in the Excel workbooks.

Key formulas
------------
S/D Index (IV):
    IV_t = K * G_t * P_t
    where G_t = (Supply_t / Usage_t)^(1/ε)
          P_t = real MYA price in year t (2025 dollars)
          K   = 100 / (G_base * P_base)   -- normalises IV to 100 in base year

Predicted real price:
    y_t = b0 + b1 * IV_t

This OLS was estimated with IV (not G alone) as the regressor, which is why
G * P_actual appears in the predicted-price formula.

Shapley counterfactuals (year t vs year t-1):
    GD_t = (Supply_{t-1} / Usage_t)^(1/ε)   -- G if only usage changed
    GS_t = (Supply_t  / Usage_{t-1})^(1/ε)  -- G if only supply changed
    yD_t = b0 + b1 * K * P_t * GD_t
    yS_t = b0 + b1 * K * P_t * GS_t

Shapley attribution of the predicted price change (dy = y_t - y_{t-1}):
    demand_shapley = 0.5 * [(yD_t - y_{t-1}) + (y_t - yS_t)]
    supply_shapley = 0.5 * [(yS_t - y_{t-1}) + (y_t - yD_t)]

The residual (actual price change minus predicted change) is allocated back to
each component in proportion to its absolute Shapley share:
    demand_impact = demand_shapley + resid * |demand_shapley| / (|demand| + |supply|)
    supply_impact = supply_shapley + resid * |supply_shapley| / (|demand| + |supply|)

supply_impact + demand_impact == dPrice  (actual price change) for every year.
"""

import pandas as pd
from model import compute_g


def run_shapley(
    df: pd.DataFrame,
    elasticity: float,
    b0: float,
    b1: float,
    base_year: int = 2009,
) -> pd.DataFrame:
    """
    Run the Shapley price decomposition for one crop.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``year``, ``supply``, ``usage``, ``price_real``.
        Rows with missing ``price_real`` are dropped before computing.
    elasticity : float
        Own-price elasticity (negative). Use CORN_ELASTICITY or SOYBEAN_ELASTICITY.
    b0 : float
        OLS intercept for the IV regression.
    b1 : float
        OLS slope on IV for the IV regression.
    base_year : int, optional
        Reference marketing year (default 2009).

    Returns
    -------
    pd.DataFrame
        One row per year starting from the *second* year in the data (the
        first year is needed only as the prior-year reference).  Columns:

        - ``year``            : int
        - ``supply``          : float  (million bushels)
        - ``usage``           : float  (million bushels)
        - ``su_ratio``        : float  (Supply / Usage)
        - ``G``               : float  (supply/usage tightness index)
        - ``price_real``      : float  (actual real MYA price, 2025 $)
        - ``IV``              : float  (S/D Index, 100 = base year)
        - ``pred_price``      : float  (predicted real price from OLS)
        - ``dy``              : float  (change in predicted price from prior year)
        - ``dPrice``          : float  (change in actual price from prior year)
        - ``resid``           : float  (dPrice - dy)
        - ``GD``              : float  (counterfactual G, supply held at t-1)
        - ``GS``              : float  (counterfactual G, usage held at t-1)
        - ``yD``              : float  (predicted price under demand-only change)
        - ``yS``              : float  (predicted price under supply-only change)
        - ``demand_shapley``  : float  (Shapley share of dy attributed to demand)
        - ``supply_shapley``  : float  (Shapley share of dy attributed to supply)
        - ``demand_impact``   : float  (final demand share of dPrice, incl. resid)
        - ``supply_impact``   : float  (final supply share of dPrice, incl. resid)
    """
    # Drop years with no real price data.
    df = df.dropna(subset=["price_real"]).copy()
    df = df.sort_values("year").reset_index(drop=True)

    # Compute G for every year.
    df["G"] = df.apply(
        lambda r: compute_g(r["supply"], r["usage"], elasticity), axis=1
    )
    df["su_ratio"] = df["supply"] / df["usage"]

    # Compute K from the base year.
    base = df[df["year"] == base_year]
    if base.empty:
        raise KeyError(f"Base year {base_year} not found in data.")
    G_base = base["G"].values[0]
    P_base = base["price_real"].values[0]
    K = 100.0 / (G_base * P_base)

    # Compute IV and predicted price for every year.
    df["IV"]         = K * df["G"] * df["price_real"]
    df["pred_price"] = b0 + b1 * df["IV"]

    rows = []
    for i in range(1, len(df)):
        cur  = df.iloc[i]
        prev = df.iloc[i - 1]

        S_t,   U_t,   P_t   = cur["supply"],  cur["usage"],  cur["price_real"]
        S_tm1, U_tm1          = prev["supply"], prev["usage"]
        G_t,   y_t            = cur["G"],       cur["pred_price"]
        y_tm1                 = prev["pred_price"]

        # Counterfactual G values.
        GD = compute_g(S_tm1, U_t,   elasticity)   # only usage changed
        GS = compute_g(S_t,   U_tm1, elasticity)   # only supply changed

        # Counterfactual predicted prices (use current year's P and K).
        yD = b0 + b1 * K * P_t * GD
        yS = b0 + b1 * K * P_t * GS

        dy     = y_t - y_tm1
        dPrice = P_t - prev["price_real"]
        resid  = dPrice - dy

        # Shapley attributions of dy.
        demand_shapley = 0.5 * ((yD - y_tm1) + (y_t - yS))
        supply_shapley = 0.5 * ((yS - y_tm1) + (y_t - yD))

        # Residual allocated proportionally by absolute Shapley shares.
        abs_total = abs(demand_shapley) + abs(supply_shapley)
        if abs_total > 1e-12:
            demand_impact = demand_shapley + resid * abs(demand_shapley) / abs_total
            supply_impact = supply_shapley + resid * abs(supply_shapley) / abs_total
        else:
            demand_impact = resid / 2.0
            supply_impact = resid / 2.0

        rows.append({
            "year":           int(cur["year"]),
            "supply":         S_t,
            "usage":          U_t,
            "su_ratio":       cur["su_ratio"],
            "G":              G_t,
            "price_real":     P_t,
            "IV":             cur["IV"],
            "pred_price":     y_t,
            "dy":             dy,
            "dPrice":         dPrice,
            "resid":          resid,
            "GD":             GD,
            "GS":             GS,
            "yD":             yD,
            "yS":             yS,
            "demand_shapley": demand_shapley,
            "supply_shapley": supply_shapley,
            "demand_impact":  demand_impact,
            "supply_impact":  supply_impact,
        })

    return pd.DataFrame(rows)


def verify_decomp(results: pd.DataFrame, tol: float = 1e-8) -> dict:
    """
    Sanity-check the decomposition results.

    Returns a dict with:
      - ``max_resid_error``: max |demand_impact + supply_impact - dPrice|
      - ``max_shapley_error``: max |demand_shapley + supply_shapley - dy|
    """
    sum_impacts    = results["demand_impact"] + results["supply_impact"]
    sum_shapley    = results["demand_shapley"] + results["supply_shapley"]
    resid_error    = (sum_impacts - results["dPrice"]).abs().max()
    shapley_error  = (sum_shapley - results["dy"]).abs().max()

    return {
        "max_resid_error":   resid_error,
        "max_shapley_error": shapley_error,
        "pass": resid_error < tol and shapley_error < tol,
    }
