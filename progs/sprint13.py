# day13.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("outputs/day12_output.csv", parse_dates=['date'], index_col='date')

base_feats = ['momentum_3d','roc_5d','rolling_sharpe_5','z_score_10','volatility_band','SMA_diff','lag_1','lag_2','volatility_3d','Signal','momentum_z','roc_volatility']
new_feats = ['MACD','BB_width','RSI_trend']
all_feats = base_feats + new_feats

# Drop rows with NaNs in features or target
df_feats = df.dropna(subset=all_feats + ['Target']).copy()

# scale
X_raw = df_feats[all_feats]
scaler = StandardScaler()
X_scaled = pd.DataFrame( scaler.fit_transform(X_raw),index=X_raw.index,columns=all_feats)

pca = PCA(n_components=3, random_state=42)
pcs = pca.fit_transform(X_scaled)
pc_cols = ['PC1','PC2','PC3']
X_pca = pd.DataFrame(pcs, index=X_scaled.index, columns=pc_cols)

# Merge scaled features and PCs back
for c in all_feats:
    df_feats[c] = X_scaled[c]
for c in pc_cols:
    df_feats[c] = X_pca[c]

# Final feature matrix and target
feature_cols = all_feats + pc_cols
X = df_feats[feature_cols]
y = df_feats['Target']

# build the ensemble model
tscv = TimeSeriesSplit(n_splits=5)

ensemble = VotingClassifier(
    estimators=[('rf', RandomForestClassifier(n_estimators=100, random_state=42)),('xgb', XGBClassifier(eval_metric='logloss', random_state=42)),('lr', LogisticRegression(max_iter=1000, random_state=42))],
    voting='soft'
)

clf = CalibratedClassifierCV(estimator=ensemble,cv=tscv, method='sigmoid')
clf.fit(X, y)

df_feats['prob'] = clf.predict_proba(X)[:, 1]

# set a max drawdown
results = []
drawdown_limit = -0.10  # no worse than -10%

for entry in np.arange(0.50, 0.61, 0.01):
    for exit in np.arange(0.49, 0.40, -0.01):
        tmp = df_feats.copy()
        tmp['Position'] = 0
        tmp.loc[tmp['prob'] > entry, 'Position'] = 1
        tmp.loc[tmp['prob'] < exit,  'Position'] = -1
        tmp['Position'] = tmp['Position'].shift(1)
        tmp.dropna(subset=['Position'], inplace=True)

        # daily & cumulative returns
        tmp['Daily_Return'] = tmp['Position'] * tmp['Yield_Return']
        cum = (1 + tmp['Daily_Return']).cumprod()

        # metrics
        total_ret = cum.iloc[-1] - 1
        max_dd    = (cum / cum.cummax() - 1).min()

        # per-trade win rate
        tmp['chg'] = tmp['Position'].diff().fillna(0)
        tid, in_trade = 0, False
        trade_ids = []
        for c, pos in zip(tmp['chg'], tmp['Position']):
            if not in_trade and c != 0:
                tid += 1
                in_trade = True
            elif in_trade and pos == 0:
                in_trade = False
            trade_ids.append(tid if in_trade else 0)
        tmp['trade_id'] = trade_ids

        trade_returns = (tmp[tmp['trade_id'] != 0].groupby('trade_id')['Daily_Return'].apply(lambda r: (1 + r).prod() - 1))
        win_rate = trade_returns.gt(0).mean()
        n_trades = len(trade_returns)

        results.append({'entry': entry,'exit': exit,'n_trades': n_trades, 'win_rate': win_rate,'total_return': total_ret,'max_drawdown': max_dd})

grid = pd.DataFrame(results)

# filter by max_drawdown constraint
filtered = grid[grid['max_drawdown'] >= drawdown_limit]
best   = filtered.loc[filtered['total_return'].idxmax()]
entry_best, exit_best = best['entry'], best['exit']

print("Best thresholds (drawdown ≥ {:.0%}):".format(-drawdown_limit))
print(best.to_frame().T)

# create the final backtest for the model
out = df_feats.copy()
out['Position'] = 0
out.loc[out['prob'] > entry_best, 'Position'] = 1
out.loc[out['prob'] < exit_best,  'Position'] = -1
out['Position'] = out['Position'].shift(1)
out.dropna(subset=['Position'], inplace=True)

out['Daily_Return'] = out['Position'] * out['Yield_Return']
out['Cumulative_Return'] = (1 + out['Daily_Return']).cumprod()
out['Benchmark_CumReturn'] = (1 + out['Yield_Return']).cumprod()

# final metrics
n_trades       = int((out['Position'].diff().abs() > 0).sum())
wins           = ((out[out['Position'] != 0]['Daily_Return']) > 0).sum()
win_rate_final = wins / n_trades * 100
total_ret_final= out['Cumulative_Return'].iloc[-1] - 1
sharpe_final   = out['Daily_Return'].mean() / out['Daily_Return'].std() * np.sqrt(252)
max_dd_final   = (out['Cumulative_Return'] / out['Cumulative_Return'].cummax() - 1).min()

print("\nDay 13 Performance (best thresholds):")
print(f"Entry/Exit : {entry_best:.2f} / {exit_best:.2f}")
print(f"Total Return   : {total_ret_final*100:.2f}%")
print(f"Sharpe Ratio   : {sharpe_final:.2f}")
print(f"Win Rate/Trade : {win_rate_final:.2f}%")
print(f"Max Drawdown   : {max_dd_final*100:.2f}%")
print(f"Total Trades   : {n_trades}")

#plot and save the final results
plt.figure(figsize=(10,5))
plt.plot(out.index, out['Cumulative_Return'], label='Strategy', linewidth=2)
plt.plot(out.index, out['Benchmark_CumReturn'], '--', label='Buy & Hold')
plt.title('Day 13: Ensemble + Calibrated Probs\nDrawdown ≤ {:.0%}'.format(-drawdown_limit))
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

out.to_csv("outputs/day13_output.csv")
grid.to_csv("outputs/day13_threshold_grid.csv")