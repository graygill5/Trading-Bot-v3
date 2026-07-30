"""Quick sanity checks for duration returns and no yield-pct PnL."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.returns import duration_returns, modified_duration


def test_duration_positive():
    y = pd.Series([4.0, 4.1, 3.9])
    d = modified_duration(y, 10.0)
    assert (d > 5).all() and (d < 12).all()


def test_yield_up_hurts_bond():
    # Parallel +100bp should produce a negative return around -ModD * 0.01
    y = pd.Series([4.0, 5.0])
    r = duration_returns(y, maturity_years=10.0, clip_abs=None)
    assert r.iloc[1] < -0.05


def test_no_explosion_near_low_yields():
    y = pd.Series([0.5, 0.51, 0.49, 0.6])
    r = duration_returns(y, clip_abs=0.05)
    assert r.dropna().abs().max() <= 0.05 + 1e-12


def test_pct_change_is_not_used_as_return():
    # Guard: yield pct_change at low levels is huge; our returns must not match it.
    y = pd.Series([0.5, 0.6])
    bad = y.pct_change().iloc[1]
    good = duration_returns(y, clip_abs=None).iloc[1]
    assert bad > 0.15
    assert abs(good) < abs(bad)


if __name__ == "__main__":
    test_duration_positive()
    test_yield_up_hurts_bond()
    test_no_explosion_near_low_yields()
    test_pct_change_is_not_used_as_return()
    print("all sanity checks passed")
