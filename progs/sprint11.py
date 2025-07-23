import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import RFE

data = pd.read_csv("outputs/day10_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

features = ['momentum_3d', 'roc_5d', 'rolling_sharpe_5', 'z_score_10', 'volatility_band','SMA_diff', 'lag_1', 'lag_2', 'volatility_3d', 'Signal','momentum_z', 'roc_volatility']
X = data[features].copy()
y = data['Target'].copy()

# found this equation online to remove outliers based on the 3 sigma rule
for col in features:
    μ, σ = X[col].mean(), X[col].std()
    mask = (X[col] >= μ - 3*σ) & (X[col] <= μ + 3*σ)
    X = X.loc[mask]
    y = y.loc[mask]

# scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X),index=X.index,columns=features )

# drop the weekends
X_scaled = X_scaled[X_scaled.index.dayofweek < 5]
y = y.loc[X_scaled.index]

rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42),n_features_to_select=7)
rfe.fit(X_scaled, y)
selected = list(X_scaled.columns[rfe.support_])

# tune parameters again
params = {'n_estimators': [100, 200],'max_depth': [3, 5, 10],'min_samples_split': [2, 5],}
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    params,
    cv=tscv,
    scoring='f1'
)
grid.fit(X_scaled[selected], y)
model = grid.best_estimator_

# get signals 
data = data.loc[X_scaled.index]  # align original DataFrame
data[selected] = X_scaled[selected]
data['prob'] = model.predict_proba(X_scaled[selected])[:, 1]

# tell when to enter and exit a trade
data['Position'] = 0
data.loc[data['prob'] > 0.55, 'Position'] = 1
data.loc[data['prob'] < 0.45, 'Position'] = -1
data['Position'] = data['Position'].shift(1)
data.dropna(subset=['Position'], inplace=True)

# get returns
data['Daily_Return'] = data['Position'] * data['Yield_Return']
data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()
data['Benchmark_CumReturn'] = (1 + data['Yield_Return']).cumprod()

total_ret = data['Cumulative_Return'].iloc[-1] - 1
sharpe = data['Daily_Return'].mean() / data['Daily_Return'].std() * np.sqrt(252)
win_rate = (data['Daily_Return'] > 0).mean()
drawdown = data['Cumulative_Return'] / data['Cumulative_Return'].cummax() - 1
max_dd = drawdown.min()
trades = int((data['Position'].diff().abs() > 0).sum())

print("Day 11 Performance")
print(f"Total Return:  {total_ret*100:.2f}%")
print(f"Sharpe Ratio:  {sharpe:.2f}")
print(f"Win Rate:      {win_rate*100:.2f}%")
print(f"Max Drawdown:  {max_dd*100:.2f}%")
print(f"Total Trades:  {trades}")

plt.figure(figsize=(10,5))
plt.plot(data.index, data['Cumulative_Return'], label='Strategy', linewidth=2)
plt.plot(data.index, data['Benchmark_CumReturn'], '--', label='Buy & Hold')
plt.title('Day 11: Filtered & Normalized Strategy vs Benchmark')
plt.xlabel('Date'); plt.ylabel('Cumulative Return')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

data.to_csv("outputs/day11_output.csv")