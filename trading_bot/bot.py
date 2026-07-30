"""Emit the latest treasury strategy signals (research / paper-trading helper).

This is not broker-connected. It prints what the locked rule blend would hold
given yesterday's close features, for use as a discretionary or paper signal.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_bot.data import DEFAULT_CSV, load_yields
from trading_bot.features import build_features
from trading_bot.returns import add_bond_returns
from trading_bot.strategies import (
    buy_and_hold,
    combine_signals,
    curve_mean_reversion,
    vol_scaled_trend,
    yield_trend,
)

REPO = Path(__file__).resolve().parents[1]


def latest_signals(df: pd.DataFrame) -> pd.DataFrame:
    sigs = {
        "buy_and_hold": buy_and_hold(df),
        "yield_trend_63d": yield_trend(df, lookback=63),
        "curve_mean_reversion": curve_mean_reversion(df, z_entry=1.0),
        "vol_scaled_trend": vol_scaled_trend(df, lookback=63, target_vol=0.06),
    }
    sigs["rule_blend"] = combine_signals(
        [sigs["yield_trend_63d"], sigs["curve_mean_reversion"]],
        weights=[0.6, 0.4],
    )
    out = pd.DataFrame(sigs)
    out["US10Y"] = df["US10Y"]
    out["bond_return"] = df["bond_return"]
    return out


def main() -> None:
    raw = load_yields(DEFAULT_CSV)
    df = build_features(add_bond_returns(raw)).dropna(subset=["bond_return"])
    hist = latest_signals(df)
    last = hist.dropna(how="any").iloc[-1]
    asof = hist.dropna(how="any").index[-1]

    print(f"Signals as of {asof.date()} (apply next session; not live brokerage)")
    print(f"  US10Y level: {last['US10Y']:.2f}%")
    for name in (
        "buy_and_hold",
        "yield_trend_63d",
        "curve_mean_reversion",
        "vol_scaled_trend",
        "rule_blend",
    ):
        pos = float(last[name])
        side = "LONG" if pos > 0.05 else "SHORT" if pos < -0.05 else "FLAT"
        print(f"  {name:24s}  pos={pos:+.2f}  → {side}")

    # Persist a small trailing history for inspection.
    out = REPO / "outputs" / "latest_signals.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    hist.tail(60).to_csv(out)
    print(f"Wrote trailing 60 days → {out}")


if __name__ == "__main__":
    main()
