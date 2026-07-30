# Treasury Trading Bot v3

Research bot for US Treasury **constant-maturity yields** with **realistic bond PnL** and **walk-forward** evaluation. The old sprint scripts compounded `US10Y.pct_change()` in-sample and reported fantasy returns (hundreds–billions ×). That approach is retired under `progs/` (legacy).

## What changed

| Before | After |
|--------|--------|
| Traded yield % changes as “returns” | Duration + carry approx. for a 10Y note |
| Fit ML / tuned thresholds on full history | Walk-forward L1 logistic; OOS report from 2019+ |
| Only US10Y levels | Curve spreads, slope, z-scores as features |
| No / inconsistent costs | 1 bp per unit turnover |

Approximate daily bond return:

```text
r_t ≈ -ModD_{t-1} * Δy_t + y_{t-1}/252
```

(with a ±5% daily clip so bad prints cannot dominate compounding).

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run

From the repo root:

```bash
python3 main.py                 # OOS backtests + plot
python3 -m trading_bot.bot      # latest rule signals (paper / discretionary)
```

Artifacts land in `outputs/`:

- `treasury_oos_metrics.csv` — CAGR, Sharpe, max DD, turnover
- `treasury_oos_equity.csv` / `.png` — growth of $1 from the OOS start
- `treasury_oos_positions.csv` — daily positions
- `latest_signals.csv` — trailing signal history from the bot helper

## Strategies

1. **buy_and_hold** — always long the synthetic 10Y  
2. **yield_trend_63d** — long when yields fell over 63d, short when they rose  
3. **curve_mean_reversion** — fade extreme 2s10s z-scores  
4. **vol_scaled_trend** — trend with vol targeting  
5. **rule_blend** — 60/40 trend + curve (no fitting)  
6. **l1_logistic_walkforward** — sparse logistic, refit yearly on past data only  

Evaluation window defaults to **2019-01-01 → sample end**. Rule signals use fixed economics; the ML model never trains on the day it predicts.

## Honest expectations

Sharpes around **0–1** and equity in the same ballpark as buy-and-hold are normal. If you see 100×+ curves, the return definition or leakage is wrong again — do not “fix” that by adding more model complexity.

## Legacy

`progs/sprint1.py` … `sprint13.py` are the original day-by-day learning path. Do not use them for live decisions or reported performance.
