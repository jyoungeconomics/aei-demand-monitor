"""
Run the Corn & Soybean Demand Monitor model and print a verification table.

For each marketing year 2000–2026 this script computes:
  - The G index (market tightness)
  - Predicted price via OLS regression  (P = b0 + b1 * G)
  - Predicted price via base-year ratio (P = P_base * G / G_base)
  - Actual nominal MYA price from USDA data
  - Difference between each prediction method and actual price

It also loads the 'source of change' Excel files and compares our G values
against theirs as a cross-check.

Notes on the source-of-change comparison
-----------------------------------------
The corn source-of-change file uses elasticity = −0.17 (rounded), whereas
this model uses the full −0.1656585432.  Small G differences for corn are
expected and are flagged in the output.  The soybean file uses the full
elasticity so its G values should match ours exactly.

The source-of-change files also use *real* (inflation-adjusted) MYA prices,
so the predicted price columns there are not directly comparable to the
nominal prices from MVPData.  G-value agreement is the primary check.
"""

import os
import pandas as pd

from data  import load_crop_data, get_base_year_row
from model import (
    CORN_ELASTICITY,
    CORN_OLS_INTERCEPT,
    CORN_OLS_SLOPE,
    SOYBEAN_ELASTICITY,
    SOYBEAN_OLS_INTERCEPT,
    SOYBEAN_OLS_SLOPE,
    BASE_YEAR,
    compute_g,
    predict_price_ols,
    predict_price_ratio,
)

# Convenience lookup so build_results can pull the right params by crop name.
_CROP_PARAMS = {
    "corn": {
        "elasticity": CORN_ELASTICITY,
        "b0": CORN_OLS_INTERCEPT,
        "b1": CORN_OLS_SLOPE,
    },
    "soybeans": {
        "elasticity": SOYBEAN_ELASTICITY,
        "b0": SOYBEAN_OLS_INTERCEPT,
        "b1": SOYBEAN_OLS_SLOPE,
    },
}

# ---------------------------------------------------------------------------
# Paths to verification files
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")

_VERIFY_CORN = os.path.join(_ROOT, "source of change_corn.xlsx")
_VERIFY_SB   = os.path.join(_ROOT, "source of change_sb.xlsx")

# Column index for G in the 'data' sheet of the source-of-change workbooks.
_SOC_COL_YEAR = 0
_SOC_COL_G    = 16   # 'G' column (0-indexed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_verification_g(path: str) -> dict:
    """Return {year: G} from a source-of-change 'data' sheet."""
    raw = pd.read_excel(path, sheet_name="data", header=0, engine="openpyxl")
    year_col = raw.iloc[:, _SOC_COL_YEAR]
    g_col    = raw.iloc[:, _SOC_COL_G]
    result = {}
    for year, g in zip(year_col, g_col):
        if isinstance(year, (int, float)) and not pd.isna(year) and isinstance(g, float):
            result[int(year)] = g
    return result


def build_results(crop: str, verify_path: str) -> pd.DataFrame:
    """
    Load data and compute model outputs for one crop.

    Returns a DataFrame with columns:
      year, supply, usage, su_ratio, G_model, G_verify, G_diff,
      pred_ols, pred_ratio, actual_price, err_ols, err_ratio
    """
    params   = _CROP_PARAMS[crop.lower()]
    elasticity = params["elasticity"]
    b0, b1     = params["b0"], params["b1"]

    df   = load_crop_data(crop)
    base = get_base_year_row(df, BASE_YEAR)
    g_verify = load_verification_g(verify_path)

    rows = []
    for _, r in df.iterrows():
        year   = int(r["year"])
        supply = r["supply"]
        usage  = r["usage"]
        price  = r["price"]

        su     = supply / usage
        g_mod  = compute_g(supply, usage, elasticity)

        p_ols  = predict_price_ols(supply, usage, elasticity, b0, b1)
        p_rat  = predict_price_ratio(
            supply, usage,
            base["supply"], base["usage"], base["price"],
            elasticity,
        )

        g_ver  = g_verify.get(year, float("nan"))
        g_diff = g_mod - g_ver if not pd.isna(g_ver) else float("nan")

        rows.append({
            "year":        year,
            "supply":      supply,
            "usage":       usage,
            "su_ratio":    su,
            "G_model":     g_mod,
            "G_verify":    g_ver,
            "G_diff":      g_diff,
            "pred_ols":    p_ols,
            "pred_ratio":  p_rat,
            "actual_price": price,
            "err_ols":     p_ols - price,
            "err_ratio":   p_rat - price,
        })

    return pd.DataFrame(rows)


def print_results(crop: str, results: pd.DataFrame) -> None:
    """Print a formatted comparison table for one crop."""
    params = _CROP_PARAMS[crop.lower()]
    label = crop.upper()
    print(f"\n{'=' * 95}")
    print(f"  {label}  |  elasticity = {params['elasticity']}  |  base year = {BASE_YEAR}")
    print(f"{'=' * 95}")

    header = (
        f"{'Year':>4}  {'Supply':>8}  {'Usage':>8}  {'S/U':>6}  "
        f"{'G(model)':>10}  {'G(verify)':>10}  {'G diff':>8}  "
        f"{'Pred OLS':>8}  {'Pred Rat':>8}  {'Actual':>6}  "
        f"{'Err OLS':>7}  {'Err Rat':>7}"
    )
    print(header)
    print("-" * 95)

    for _, row in results.iterrows():
        g_diff_str  = f"{row['G_diff']:+8.5f}" if not pd.isna(row["G_diff"])  else "    n/a "
        g_ver_str   = f"{row['G_verify']:10.6f}" if not pd.isna(row["G_verify"]) else "       n/a"
        print(
            f"{int(row['year']):>4}  "
            f"{row['supply']:>8,.0f}  "
            f"{row['usage']:>8,.0f}  "
            f"{row['su_ratio']:>6.4f}  "
            f"{row['G_model']:>10.6f}  "
            f"{g_ver_str}  "
            f"{g_diff_str}  "
            f"{row['pred_ols']:>8.4f}  "
            f"{row['pred_ratio']:>8.4f}  "
            f"{row['actual_price']:>6.2f}  "
            f"{row['err_ols']:>+7.4f}  "
            f"{row['err_ratio']:>+7.4f}"
        )

    print()
    # Summary stats
    valid = results.dropna(subset=["G_diff"])
    max_g_diff = valid["G_diff"].abs().max()
    print(f"  Max |G diff vs verify|  : {max_g_diff:.6f}")
    print(f"  Mean abs err (OLS)       : ${results['err_ols'].abs().mean():.4f}/bu")
    print(f"  Mean abs err (ratio)     : ${results['err_ratio'].abs().mean():.4f}/bu")

    # Flag the 2025 soybean value if near zero change
    if crop.lower() == "soybeans":
        row_2025 = results[results["year"] == 2025]
        if not row_2025.empty:
            err = row_2025["err_ols"].values[0]
            print(f"\n  Note (2025 soybeans): OLS prediction error = ${err:+.4f}/bu")
            print("  If 2025 actual price looks like a placeholder, treat with caution.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\nCorn & Soybean Demand Monitor — Priority 1 Verification")
    print(f"Model: G = (S/U)^(1/e)  |  Ratio method: P = P_base * (G / G_base)")
    print(f"Corn elasticity: {CORN_ELASTICITY}  |  Soybean elasticity: {SOYBEAN_ELASTICITY}")
    print(f"\nNote: 'G verify' column reads from the source-of-change Excel files.")
    print("Corn source-of-change uses e = -0.17 (old rounded value); expect small G diffs for corn.")

    corn_results = build_results("corn",     _VERIFY_CORN)
    sb_results   = build_results("soybeans", _VERIFY_SB)

    print_results("Corn",     corn_results)
    print_results("Soybeans", sb_results)


if __name__ == "__main__":
    main()
