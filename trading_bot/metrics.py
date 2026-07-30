"""Performance metrics for treasury backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def summarize_returns(
    returns: pd.Series,
    positions: pd.Series | None = None,
    name: str = "strategy",
) -> dict:
    """Annualized metrics from daily strategy returns."""
    r = returns.dropna()
    if r.empty:
        return {"name": name, "n_days": 0}

    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    years = len(r) / 252.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(r.std() * np.sqrt(252.0))
    # Standard annualized Sharpe: (mean/std) * √252
    sharpe = float((r.mean() / r.std()) * np.sqrt(252.0)) if r.std() > 0 else 0.0
    hit = float((r > 0).mean())

    turnover = 0.0
    if positions is not None:
        pos = positions.reindex(r.index).fillna(0.0)
        turnover = float(pos.diff().abs().sum() / max(len(pos), 1))

    return {
        "name": name,
        "n_days": int(len(r)),
        "years": round(years, 2),
        "total_return": round(total, 4),
        "cagr": round(cagr, 4),
        "ann_vol": round(vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_drawdown(equity), 4),
        "hit_rate": round(hit, 4),
        "avg_daily_turnover": round(turnover, 4),
        "final_equity": round(float(equity.iloc[-1]), 4),
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("name")
