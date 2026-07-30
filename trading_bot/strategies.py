"""Treasury strategies designed to stay simple and avoid in-sample abuse.

Rule strategies use fixed economics (duration trend, curve mean-reversion).
The ML strategy is L1 logistic regression with few features, fit only inside
walk-forward folds — never on the full sample used for the equity curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_bot.backtest import walk_forward_signals
from trading_bot.features import FEATURE_COLS


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Always long the synthetic 10Y bond portfolio."""
    return pd.Series(1.0, index=df.index, name="signal")


def yield_trend(df: pd.DataFrame, lookback: int = 63, thresh: float = 0.0) -> pd.Series:
    """Long bonds when yields have been falling (negative yield momentum).

    Falling yields → rising bond prices. Flat when momentum is mixed.
    """
    mom = df["US10Y"].diff(lookback)
    sig = pd.Series(0.0, index=df.index)
    sig = sig.mask(mom < -thresh, 1.0)
    sig = sig.mask(mom > thresh, -1.0)
    return sig.rename("signal")


def curve_mean_reversion(
    df: pd.DataFrame,
    z_entry: float = 1.0,
    z_exit: float = 0.25,
) -> pd.Series:
    """Fade extreme 2s10s z-scores: steep → long duration, inverted → short.

    Uses a simple state machine so we do not flip every day around the threshold.
    """
    z = df["z_2s10s_63d"]
    pos = 0.0
    out = []
    for val in z:
        if pd.isna(val):
            out.append(pos)
            continue
        if pos == 0.0:
            if val > z_entry:
                pos = 1.0
            elif val < -z_entry:
                pos = -1.0
        elif pos > 0 and val < z_exit:
            pos = 0.0
        elif pos < 0 and val > -z_exit:
            pos = 0.0
        out.append(pos)
    return pd.Series(out, index=df.index, name="signal")


def vol_scaled_trend(df: pd.DataFrame, lookback: int = 63, target_vol: float = 0.06) -> pd.Series:
    """Trend signal with position size capped by trailing bond-return vol."""
    direction = yield_trend(df, lookback=lookback)
    vol = df["bond_return"].rolling(63).std() * np.sqrt(252.0)
    scale = (target_vol / vol.replace(0.0, np.nan)).clip(upper=1.5).fillna(0.0)
    return (direction * scale).rename("signal")


def _ml_fit(train: pd.DataFrame):
    cols = [c for c in FEATURE_COLS if c in train.columns]
    X = train[cols]
    # Predict next-day bond return sign using today's features.
    y = (train["bond_return"].shift(-1) > 0).astype(int)
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    # Drop last row if label is NaN from shift(-1) at train end — already masked.
    if len(X) < 500 or y.nunique() < 2:
        return None
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    l1_ratio=1.0,
                    C=0.05,  # strong sparsity — resists curve-fitting noise
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return {"model": pipe, "cols": cols}


def _ml_predict(model, test: pd.DataFrame) -> pd.Series:
    if model is None:
        return pd.Series(0.0, index=test.index)
    cols = model["cols"]
    X = test[cols]
    sig = pd.Series(0.0, index=test.index)
    valid = X.notna().all(axis=1)
    if valid.any():
        proba = model["model"].predict_proba(X.loc[valid])[:, 1]
        # Conservative bands: only trade when the model is reasonably sure.
        raw = np.where(proba > 0.55, 1.0, np.where(proba < 0.45, -1.0, 0.0))
        sig.loc[valid] = raw
    return sig


def sparse_logistic_walkforward(df: pd.DataFrame) -> pd.Series:
    """L1 logistic classifier, refit yearly on prior history only."""
    return walk_forward_signals(
        df,
        fit_fn=_ml_fit,
        predict_fn=_ml_predict,
        train_years=10,
        test_years=1,
        min_train_days=1500,
        embargos_days=5,
    ).rename("signal")


def combine_signals(
    signals: list[pd.Series],
    weights: list[float] | None = None,
) -> pd.Series:
    """Equal-weight (or weighted) average of signals, clipped to [-1, 1]."""
    if not signals:
        raise ValueError("need at least one signal")
    w = weights or [1.0 / len(signals)] * len(signals)
    aligned = pd.concat(signals, axis=1).fillna(0.0)
    combo = sum(aligned.iloc[:, i] * w[i] for i in range(len(w)))
    return combo.clip(-1.0, 1.0).rename("signal")
