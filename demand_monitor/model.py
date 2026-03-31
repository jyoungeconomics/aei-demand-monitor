"""
Core price prediction model for the Corn & Soybean Demand Monitor.

The model uses a constant elasticity of demand framework:

    G = (Supply / Usage) ^ (1 / elasticity)

G is an index of market tightness.  A smaller S/U ratio (tighter market)
produces a higher G and thus a higher predicted price.

Two methods are provided for converting G to an absolute price level:

  1. OLS regression:   Price = b0 + b1 * G
     This is what the "old method" Excel sheets use (labeled "not used" in
     the workbooks — the current model relies on Shapley decomposition).

  2. Base-year ratio:  Price = base_price * (G / G_base)
     This is the formulation described in CLAUDE.md and is the method the
     current Excel model actually uses for price prediction.

Corn and soybeans have separate elasticities and OLS coefficients.
"""

# ---------------------------------------------------------------------------
# Model parameters — CORN
# ---------------------------------------------------------------------------

# Own-price elasticity of demand for corn, estimated via 2SLS (log-log).
# Source: Demand model.docx, Table 1.
CORN_ELASTICITY = -0.1651652067

# OLS regression coefficients for corn (from 'old method --- not used' sheet).
# These regress the real MYA price on the G index.  Not the active method.
# Source: CORN/decomp of S&D impact on price.xlsx, 'old method --- not used'.
CORN_OLS_INTERCEPT = 1.9155    # b0
CORN_OLS_SLOPE     = 0.0324    # b1

# ---------------------------------------------------------------------------
# Model parameters — SOYBEANS
# ---------------------------------------------------------------------------

# Own-price elasticity of demand for soybeans, estimated via 2SLS (log-log).
# Source: Demand model.docx, Table 1; Output.MVP2since2009 workbook cell O1.
SOYBEAN_ELASTICITY = -0.1656585432

# OLS regression coefficients for soybeans (from 'old method --- not used' sheet).
# Source: SOYBEANS/decomp of S&D impact on price.xlsx, 'old method --- not used'.
SOYBEAN_OLS_INTERCEPT = 5.73894554255869    # b0
SOYBEAN_OLS_SLOPE     = 0.0964584910872625  # b1

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

# Marketing year used as the reference / base year throughout.
BASE_YEAR = 2009


# ---------------------------------------------------------------------------
# Core formula functions
# ---------------------------------------------------------------------------

def compute_su_ratio(supply: float, usage: float) -> float:
    """
    Calculate the supply-to-usage ratio for a single marketing year.

    Parameters
    ----------
    supply : float
        Total supply in million bushels
        (Beginning Stocks + Production + Imports).
    usage : float
        Total usage (disappearance) in million bushels
        (Feed + FSI/Crush + Exports for corn;
         Crush + Feed/Waste + Exports for soybeans).

    Returns
    -------
    float
        The S/U ratio (dimensionless).
    """
    return supply / usage


def compute_g(supply: float, usage: float, elasticity: float) -> float:
    """
    Compute the G index: G = (Supply / Usage) ^ (1 / elasticity).

    G captures market tightness in a way that is directly proportional to
    price.  Because elasticity is negative, 1/elasticity is also negative,
    so a tighter market (lower S/U) produces a *larger* G, predicting a
    higher price.

    Parameters
    ----------
    supply : float
        Total supply in million bushels.
    usage : float
        Total usage in million bushels.
    elasticity : float
        Own-price elasticity of demand (negative number).
        Use CORN_ELASTICITY or SOYBEAN_ELASTICITY as appropriate.

    Returns
    -------
    float
        The G index value.
    """
    su_ratio = compute_su_ratio(supply, usage)
    return su_ratio ** (1.0 / elasticity)


def predict_price_ols(
    supply: float,
    usage: float,
    elasticity: float,
    b0: float,
    b1: float,
) -> float:
    """
    Predict the marketing-year average (MYA) price using OLS regression.

        Predicted Price = b0 + b1 * G

    Note: this method is labeled 'old method --- not used' in the Excel
    workbooks.  The coefficients were estimated against *real* (2025-dollar)
    prices, so predictions should be compared to real prices, not nominal.

    Parameters
    ----------
    supply : float
        Total supply in million bushels.
    usage : float
        Total usage in million bushels.
    elasticity : float
        Own-price elasticity (negative). Use CORN_ELASTICITY or SOYBEAN_ELASTICITY.
    b0 : float
        OLS intercept. Use CORN_OLS_INTERCEPT or SOYBEAN_OLS_INTERCEPT.
    b1 : float
        OLS slope on G. Use CORN_OLS_SLOPE or SOYBEAN_OLS_SLOPE.

    Returns
    -------
    float
        Predicted real MYA price in $/bushel (2025 dollars).
    """
    g = compute_g(supply, usage, elasticity)
    return b0 + b1 * g


def predict_price_ratio(
    supply: float,
    usage: float,
    base_supply: float,
    base_usage: float,
    base_price: float,
    elasticity: float,
) -> float:
    """
    Predict price as a ratio relative to the base year.

        Predicted Price = base_price * (G / G_base)

    This is the active method used in the current Excel workbooks.  It anchors
    predictions to the known base-year price rather than an OLS intercept.

    Parameters
    ----------
    supply : float
        Total supply in million bushels (current year).
    usage : float
        Total usage in million bushels (current year).
    base_supply : float
        Total supply in million bushels (base year, 2009).
    base_usage : float
        Total usage in million bushels (base year, 2009).
    base_price : float
        Actual MYA price in $/bushel for the base year.
    elasticity : float
        Own-price elasticity (negative). Use CORN_ELASTICITY or SOYBEAN_ELASTICITY.

    Returns
    -------
    float
        Predicted MYA price in $/bushel (same units as base_price).
    """
    g_current = compute_g(supply, usage, elasticity)
    g_base    = compute_g(base_supply, base_usage, elasticity)
    return base_price * (g_current / g_base)
