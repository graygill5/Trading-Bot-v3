"""Map yield changes into approximate bond portfolio returns.

Trading raw ``US10Y.pct_change()`` is wrong: yield levels are not prices, and
near-zero yields make percent changes explode. A constant-maturity 10Y note
is approximated with modified duration + carry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def modified_duration(yield_pct: pd.Series, maturity_years: float = 10.0) -> pd.Series:
    """Annual-pay par-bond modified duration from yield in percent."""
    y = (yield_pct / 100.0).clip(lower=1e-4)
    # Macaulay for a par bond ≈ (1+y)/y * (1 - (1+y)^(-T)); ModD = Mac / (1+y).
    mac = ((1.0 + y) / y) * (1.0 - (1.0 + y) ** (-maturity_years))
    return mac / (1.0 + y)


def duration_returns(
    yield_pct: pd.Series,
    maturity_years: float = 10.0,
    clip_abs: float = 0.05,
) -> pd.Series:
    """Approximate daily total return of a constant-maturity bond.

    ``r_t ≈ -ModD_{t-1} * Δy_t + y_{t-1}/252``

    where ``Δy`` is the change in yield in decimal terms. Daily moves are
    clipped so a single bad print cannot dominate multi-decade compounding.
    """
    y = yield_pct / 100.0
    dy = y.diff()
    dur = modified_duration(yield_pct, maturity_years).shift(1)
    carry = y.shift(1) / 252.0
    ret = (-dur * dy + carry).replace([np.inf, -np.inf], np.nan)
    if clip_abs is not None:
        ret = ret.clip(lower=-clip_abs, upper=clip_abs)
    return ret.rename("bond_return")


def add_bond_returns(df: pd.DataFrame, col: str = "US10Y") -> pd.DataFrame:
    """Attach duration, yield change, and bond return columns."""
    out = df.copy()
    out["mod_duration"] = modified_duration(out[col], 10.0)
    out["yield_change_bp"] = out[col].diff() * 100.0  # percent points → bp
    out["bond_return"] = duration_returns(out[col], maturity_years=10.0)
    return out
