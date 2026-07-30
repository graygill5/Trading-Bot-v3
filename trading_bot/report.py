"""Run research report: save metrics CSV and equity-curve plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from trading_bot.backtest import BacktestResult
from trading_bot.metrics import metrics_table


def print_metrics(results: list[BacktestResult]) -> pd.DataFrame:
    table = metrics_table([r.metrics for r in results])
    print("\n=== Backtest summary (duration-based 10Y returns, with costs) ===\n")
    print(table.to_string())
    print(
        "\nNotes: total_return/CAGR are for the evaluation window only. "
        "Sharpe uses daily net returns × √252. "
        "Values near buy-and-hold are expected; 100×+ curves are a red flag.\n"
    )
    return table


def save_outputs(
    results: list[BacktestResult],
    out_dir: Path,
    prefix: str = "backtest",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = metrics_table([r.metrics for r in results])
    table.to_csv(out_dir / f"{prefix}_metrics.csv")

    equity = pd.concat({r.name: r.equity for r in results}, axis=1)
    equity.to_csv(out_dir / f"{prefix}_equity.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    for r in results:
        ax.plot(r.equity.index, r.equity.values, label=r.name, linewidth=1.4)
    ax.set_title("Growth of $1 — synthetic constant-maturity 10Y")
    ax.set_ylabel("Equity ($1 start)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_equity.png", dpi=140)
    plt.close(fig)

    positions = pd.concat({r.name: r.positions for r in results}, axis=1)
    positions.to_csv(out_dir / f"{prefix}_positions.csv")
