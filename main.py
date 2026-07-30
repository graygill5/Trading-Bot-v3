"""Treasury bot research runner — realistic duration PnL + walk-forward tests.

Replaces the old sprint chain that compounded yield % changes in-sample
(which produced fantasy multi-hundred× equity curves).
"""

from __future__ import annotations

from pathlib import Path

from trading_bot.backtest import run_backtest, time_splits
from trading_bot.data import load_yields
from trading_bot.features import build_features
from trading_bot.report import print_metrics, save_outputs
from trading_bot.returns import add_bond_returns
from trading_bot.strategies import (
    buy_and_hold,
    combine_signals,
    curve_mean_reversion,
    sparse_logistic_walkforward,
    vol_scaled_trend,
    yield_trend,
)

REPO = Path(__file__).resolve().parent
OUT = REPO / "outputs"
COST_BPS = 1.0  # ~1 bp per unit of position turnover (ETF-like)
OOS_START = "2019-01-01"  # locked test window; strategies must not peek


def prepare() -> object:
    raw = load_yields(REPO / "us_treasury_yields_daily.csv")
    df = add_bond_returns(raw)
    df = build_features(df)
    df = df.dropna(subset=["bond_return"])
    return df


def run_all(df) -> list:
    # Full-sample rule signals are OK economically (no fitted params), but we
    # still only *evaluate* on the OOS window so we do not cherry-pick eras.
    sig_bh = buy_and_hold(df)
    sig_trend = yield_trend(df, lookback=63)
    sig_curve = curve_mean_reversion(df, z_entry=1.0)
    sig_vol = vol_scaled_trend(df, lookback=63, target_vol=0.06)

    print("Fitting walk-forward L1 logistic (this can take a bit)...")
    sig_ml = sparse_logistic_walkforward(df)

    # Blend only the two simple rules — no fitting.
    sig_blend = combine_signals([sig_trend, sig_curve], weights=[0.6, 0.4])

    specs = [
        ("buy_and_hold", sig_bh),
        ("yield_trend_63d", sig_trend),
        ("curve_mean_reversion", sig_curve),
        ("vol_scaled_trend", sig_vol),
        ("rule_blend", sig_blend),
        ("l1_logistic_walkforward", sig_ml),
    ]

    results = []
    for name, sig in specs:
        results.append(
            run_backtest(
                df,
                sig,
                name=name,
                cost_bps=COST_BPS,
                allow_short=True,
                start=OOS_START,
            )
        )
    return results


def sanity_checks(df) -> None:
    """Fail loudly if someone reintroduces yield-pct PnL nonsense."""
    r = df["bond_return"].dropna()
    assert r.abs().max() <= 0.05 + 1e-9, "bond returns should be clipped"
    # Multi-decade buy-and-hold of duration returns should not be hundreds of ×
    # on a short OOS window either — checked after backtest.
    print(
        f"Data: {df.index.min().date()} → {df.index.max().date()} "
        f"({len(df)} days). Mean daily bond return={r.mean():.5f}, "
        f"ann vol≈{r.std() * (252 ** 0.5):.2%}."
    )


def main() -> None:
    df = prepare()
    sanity_checks(df)

    train, val, test = time_splits(df)
    print(
        f"Splits — train: {len(train)} days (≤{train.index.max().date()}), "
        f"val: {len(val)}, test/OOS from {OOS_START}: {len(df.loc[OOS_START:])}."
    )

    results = run_all(df)
    table = print_metrics(results)
    save_outputs(results, OUT, prefix="treasury_oos")

    # Soft guardrail: if any strategy claims >50× on ~5y OOS, something is wrong.
    for r in results:
        if r.metrics.get("final_equity", 1) > 50:
            print(
                f"WARNING: {r.name} final equity={r.metrics['final_equity']} "
                "looks unrealistic for this window — check PnL definition."
            )

    print(f"Wrote metrics/equity/positions under {OUT}/treasury_oos_*")

    # Latest paper-trading signal from the rule blend.
    from trading_bot.bot import latest_signals

    hist = latest_signals(df)
    last = hist.dropna(how="any").iloc[-1]
    asof = hist.dropna(how="any").index[-1]
    blend = float(last["rule_blend"])
    side = "LONG" if blend > 0.05 else "SHORT" if blend < -0.05 else "FLAT"
    print(f"\nLatest rule_blend signal @ {asof.date()}: {blend:+.2f} → {side}")
    print("For signals only: python3 -m trading_bot.bot")


if __name__ == "__main__":
    main()
