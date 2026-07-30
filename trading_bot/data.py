"""Load and clean US Treasury constant-maturity yield data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "us_treasury_yields_daily.csv"

YIELD_COLS = [
    "US1M",
    "US3M",
    "US6M",
    "US1Y",
    "US2Y",
    "US3Y",
    "US5Y",
    "US7Y",
    "US10Y",
    "US20Y",
    "US30Y",
]

# Maturities in years for duration math / curve features.
MATURITY_YEARS = {
    "US1M": 1 / 12,
    "US3M": 0.25,
    "US6M": 0.5,
    "US1Y": 1.0,
    "US2Y": 2.0,
    "US3Y": 3.0,
    "US5Y": 5.0,
    "US7Y": 7.0,
    "US10Y": 10.0,
    "US20Y": 20.0,
    "US30Y": 30.0,
}


def load_yields(path: Path | str | None = None) -> pd.DataFrame:
    """Load daily yields (percent), indexed by date, with numeric columns."""
    csv_path = Path(path) if path else DEFAULT_CSV
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    for col in YIELD_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Require the 10Y — our traded risk factor.
    df = df.dropna(subset=["US10Y"])
    # Zero yields (rare data quirks) break duration; treat as missing.
    df.loc[df["US10Y"] <= 0, "US10Y"] = np.nan
    df = df.dropna(subset=["US10Y"])
    return df
