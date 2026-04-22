"""
Data loading for the Corn & Soybean Demand Monitor.

Reads supply, usage, and price data from the master Excel workbook
CornSBDemand.MVPData.xlsx (located in the SOYBEANS/ folder — that copy
is more recent and contains both crops).

Sheet layout (Corn - bu and Soybeans - bu):
  Col 0  : Year
  Col 6  : Total Supply (million bushels)
  Col 20 : Total Usage*** (million bushels)
  Col 12 : US MYA Price, nominal ($/bushel)

Units note
----------
Both "Corn - bu" and "Soybeans - bu" sheets store quantities already converted
to million bushels. This module always uses million-bushel units.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Path to the master data workbook and verification files
# ---------------------------------------------------------------------------

# Build an absolute path relative to this file so the script works from any
# working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "..")           # Demand Stuff/
MVPDATA_PATH = os.path.join(_DATA_DIR, "SOYBEANS", "CornSBDemand.MVPData.xlsx")

# Source-of-change workbooks contain real (2025-dollar) MYA prices.
_SOC_PATHS = {
    "corn":     os.path.join(_DATA_DIR, "source of change_corn.xlsx"),
    "soybeans": os.path.join(_DATA_DIR, "source of change_sb.xlsx"),
}

# Column positions in the source-of-change 'data' sheet (0-indexed)
_SOC_COL_YEAR       = 0
_SOC_COL_PRICE_REAL = 2   # 'Real MYA Price' (2025 dollars)

# Which Excel sheet to read for each crop
# Use the "-bu" (bushel) converted sheets for both crops to ensure million-bushel units
_SHEET = {
    "corn":     "Corn - bu",
    "soybeans": "Soybeans - bu",
}

# Column positions (0-indexed) within each sheet
_COL_YEAR   = 0
_COL_SUPPLY = 6
_COL_USAGE  = 20
_COL_PRICE  = 12

# ---------------------------------------------------------------------------
# WASDE Monthly MYA Price Forecast
# ---------------------------------------------------------------------------
# Update these values monthly from WASDE publications
# These are the MYA (marketing year average) price forecasts in $/bushel
WASDE_MYA_PRICE = {
    "corn":     3.85,
    "soybeans": 10.25,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_crop_data(
    crop: str,
    year_start: int = 2000,
    year_end: int = 2026,
    workbook_path: str = MVPDATA_PATH,
) -> pd.DataFrame:
    """
    Load annual supply/usage/price data for one crop from the master workbook.

    Parameters
    ----------
    crop : str
        Either ``"corn"`` or ``"soybeans"`` (case-insensitive).
    year_start : int, optional
        First marketing year to include (default 2000).
    year_end : int, optional
        Last marketing year to include (default 2026).
    workbook_path : str, optional
        Path to CornSBDemand.MVPData.xlsx.  Defaults to the SOYBEANS/ copy.

    Returns
    -------
    pd.DataFrame
        One row per marketing year with columns:
        - ``year``    : int
        - ``supply``  : float  (million bushels, Total Supply)
        - ``usage``   : float  (million bushels, Total Usage)
        - ``price``   : float  (nominal $/bushel, US MYA Price)
    """
    crop_key = crop.lower()
    if crop_key not in _SHEET:
        raise ValueError(f"crop must be 'corn' or 'soybeans', got '{crop}'")

    sheet_name = _SHEET[crop_key]

    # Read the full sheet; first row is the header.
    raw = pd.read_excel(
        workbook_path,
        sheet_name=sheet_name,
        header=0,          # row 0 is the column header
        engine="openpyxl",
    )

    # Pull only the columns we need and give them clean names.
    df = pd.DataFrame({
        "year":   raw.iloc[:, _COL_YEAR],
        "supply": raw.iloc[:, _COL_SUPPLY],
        "usage":  raw.iloc[:, _COL_USAGE],
        "price":  raw.iloc[:, _COL_PRICE],
    })

    # Drop rows where Year is not a valid integer (e.g., trailing blank rows).
    df = df[df["year"].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    df["year"] = df["year"].astype(int)

    # Filter to the requested date range.
    df = df[(df["year"] >= year_start) & (df["year"] <= year_end)].copy()
    df = df.reset_index(drop=True)

    return df


def load_real_prices(
    crop: str,
    soc_path: str | None = None,
) -> dict:
    """
    Load real (2025-dollar) MYA prices from the source-of-change workbook.

    Parameters
    ----------
    crop : str
        Either ``"corn"`` or ``"soybeans"`` (case-insensitive).
    soc_path : str, optional
        Path to the source-of-change Excel file.  Defaults to the copy in the
        main project folder.

    Returns
    -------
    dict
        ``{year: real_price}`` mapping.  Only years with numeric price values
        are included.
    """
    crop_key = crop.lower()
    if crop_key not in _SOC_PATHS:
        raise ValueError(f"crop must be 'corn' or 'soybeans', got '{crop}'")

    path = soc_path or _SOC_PATHS[crop_key]
    raw  = pd.read_excel(path, sheet_name="data", header=0, engine="openpyxl")

    result = {}
    for _, row in raw.iterrows():
        year  = row.iloc[_SOC_COL_YEAR]
        price = row.iloc[_SOC_COL_PRICE_REAL]
        if (
            isinstance(year, (int, float))
            and not pd.isna(year)
            and isinstance(price, (int, float))
            and not pd.isna(price)
        ):
            result[int(year)] = float(price)
    return result


def get_base_year_row(df: pd.DataFrame, base_year: int = 2009) -> pd.Series:
    """
    Return the single row for the base year from a DataFrame returned by
    :func:`load_crop_data`.

    Raises
    ------
    KeyError
        If the base year is not present in the DataFrame.
    """
    row = df[df["year"] == base_year]
    if row.empty:
        raise KeyError(f"Base year {base_year} not found in data.")
    return row.iloc[0]
