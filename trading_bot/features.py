"""Causal yield-curve features for treasury strategies.

All features use only information available at close of day t. Positions are
applied on day t+1 in the backtester (``.shift(1)``), so there is no same-day
lookahead into the return being predicted.
"""

from __future__ import annotations

import pandas as pd


FEATURE_COLS = [
    "y10",
    "dy10_5d",
    "dy10_21d",
    "mom_63d",
    "vol_21d",
    "spread_2s10s",
    "spread_3m10y",
    "spread_10s30s",
    "curve_level",
    "curve_slope",
    "z_2s10s_63d",
    "rsi_14",
]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, pd.NA)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a small, interpretable feature set from the yield curve."""
    out = df.copy()
    out["y10"] = out["US10Y"]
    out["dy10_5d"] = out["US10Y"].diff(5)
    out["dy10_21d"] = out["US10Y"].diff(21)
    out["mom_63d"] = out["US10Y"].diff(63)
    out["vol_21d"] = out["US10Y"].diff().rolling(21).std()

    # Spreads (percentage points). Missing legs → NaN (dropped later).
    out["spread_2s10s"] = out["US10Y"] - out["US2Y"]
    out["spread_3m10y"] = out["US10Y"] - out["US3M"]
    out["spread_10s30s"] = out["US30Y"] - out["US10Y"]

    # Crude level / slope from available points.
    level_cols = [c for c in ("US2Y", "US5Y", "US10Y") if c in out.columns]
    out["curve_level"] = out[level_cols].mean(axis=1)
    out["curve_slope"] = out["spread_2s10s"]

    mu = out["spread_2s10s"].rolling(63).mean()
    sd = out["spread_2s10s"].rolling(63).std()
    out["z_2s10s_63d"] = (out["spread_2s10s"] - mu) / sd.replace(0.0, pd.NA)

    out["rsi_14"] = _rsi(out["US10Y"], 14)
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only model feature columns (may contain NaNs)."""
    return df[FEATURE_COLS].copy()
