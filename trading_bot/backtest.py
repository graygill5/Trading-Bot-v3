"""Walk-forward backtest engine with transaction costs.

Design goals:
- Decide signal at close t, earn return on day t+1 (position shift).
- Never fit or tune on the evaluation window being reported.
- Charge costs on position changes (ETF/futures-like friction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from trading_bot.metrics import summarize_returns


SignalFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    positions: pd.Series
    equity: pd.Series
    metrics: dict
    oos_start: pd.Timestamp | None = None


def apply_costs(positions: pd.Series, cost_bps: float) -> pd.Series:
    """Daily cost as fraction of NAV from absolute position change."""
    turnover = positions.diff().abs().fillna(positions.abs())
    return turnover * (cost_bps / 1e4)


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    *,
    name: str = "strategy",
    return_col: str = "bond_return",
    cost_bps: float = 1.0,
    allow_short: bool = True,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> BacktestResult:
    """Vectorized backtest: position_t = signal_{t-1}."""
    data = df[[return_col]].copy()
    pos = signal.reindex(data.index).fillna(0.0).astype(float)
    if not allow_short:
        pos = pos.clip(lower=0.0)
    # Trade next day on yesterday's signal.
    pos = pos.shift(1).fillna(0.0)

    if start is not None:
        data = data.loc[pd.Timestamp(start) :]
        pos = pos.loc[data.index]
    if end is not None:
        data = data.loc[: pd.Timestamp(end)]
        pos = pos.loc[data.index]

    raw = pos * data[return_col]
    costs = apply_costs(pos, cost_bps)
    net = (raw - costs).dropna()
    pos = pos.reindex(net.index)
    equity = (1.0 + net).cumprod()
    metrics = summarize_returns(net, pos, name=name)
    return BacktestResult(
        name=name,
        returns=net,
        positions=pos,
        equity=equity,
        metrics=metrics,
        oos_start=net.index.min() if len(net) else None,
    )


def walk_forward_signals(
    df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], object],
    predict_fn: Callable[[object, pd.DataFrame], pd.Series],
    *,
    train_years: int = 8,
    test_years: int = 1,
    min_train_days: int = 1000,
    embargos_days: int = 5,
) -> pd.Series:
    """Expanding/rolling walk-forward: fit on past, predict next block only.

    ``fit_fn(train_df) -> model``
    ``predict_fn(model, test_df) -> signal Series aligned to test_df.index``

    An embargo gap after the train end reduces label leakage from overlapping
    multi-day features into the first test days.
    """
    idx = df.index
    start = idx.min()
    end = idx.max()
    cursor = start + pd.DateOffset(years=train_years)
    signals = pd.Series(0.0, index=idx, dtype=float)

    while cursor < end:
        train_end = cursor - pd.Timedelta(days=embargos_days)
        test_end = min(cursor + pd.DateOffset(years=test_years), end)
        train = df.loc[:train_end]
        test = df.loc[cursor:test_end]
        if len(train) < min_train_days or test.empty:
            cursor = test_end + pd.Timedelta(days=1)
            continue
        model = fit_fn(train)
        pred = predict_fn(model, test)
        signals.loc[pred.index] = pred.astype(float)
        cursor = test_end + pd.Timedelta(days=1)

    return signals


def time_splits(
    df: pd.DataFrame,
    train_end: str = "2012-12-31",
    val_end: str = "2018-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fixed chronological train / validation / test slices."""
    train = df.loc[: pd.Timestamp(train_end)]
    val = df.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1) : pd.Timestamp(val_end)]
    test = df.loc[pd.Timestamp(val_end) + pd.Timedelta(days=1) :]
    return train, val, test
