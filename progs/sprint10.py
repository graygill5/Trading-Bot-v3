import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import RFE

data = pd.read_csv("outputs/day9_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

features = ['momentum_3d', 'roc_5d', 'rolling_sharpe_5', 'z_score_10', 'volatility_band','SMA_diff', 'lag_1', 'lag_2', 'volatility_3d', 'Signal', 'momentum_z', 'roc_volatility']
X = data[features]
y = data['Target']

rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=7)
rfe.fit(X, y)
selected_features = list(X.columns[rfe.support_])

# tuneing the random forest model for the best features using grid search
params = {'n_estimators': [100, 200],'max_depth': [3, 5, 10],'min_samples_split': [2, 5],}
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), params, cv=tscv, scoring='f1')
grid.fit(X[selected_features], y)
model = grid.best_estimator_

# use model predictions with the signals from early days
data['prob'] = model.predict_proba(X[selected_features])[:, 1]
data['Position'] = 0
data.loc[data['prob'] > 0.55, 'Position'] = 1
data.loc[data['prob'] < 0.45, 'Position'] = -1
data['Position'] = data['Position'].shift(1)
data.dropna(inplace=True)

# get retuns
data['Daily_Return'] = data['Position'] * data['Yield_Return']
data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()

# show normal market returns 
data['Benchmark_CumReturn'] = (1 + data['Yield_Return']).cumprod()
final_strategy_value = data['Cumulative_Return'].iloc[-1]
final_benchmark_value = data['Benchmark_CumReturn'].iloc[-1]

print("Final Portfolio Values:")
print(f"Strategy: {final_strategy_value:.2f}x")
print(f"Benchmark: {final_benchmark_value:.2f}x")

# track some of the usual finance stats
total_return = data['Cumulative_Return'].iloc[-1] - 1
sharpe = np.mean(data['Daily_Return']) / np.std(data['Daily_Return']) * np.sqrt(252)
win_rate = (data['Daily_Return'] > 0).mean()
drawdown = (data['Cumulative_Return'] / data['Cumulative_Return'].cummax()) - 1
max_drawdown = drawdown.min()
trade_count = (data['Position'].diff().abs() > 0).sum()

print("Day 10 Strategy Performance")
print(f"Total Return: {total_return * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Win Rate: {win_rate * 100:.2f}%")
print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
print(f"Total Trades: {int(trade_count)}")

plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Cumulative_Return'], label="Strategy", linewidth=2)
plt.plot(data.index, data['Benchmark_CumReturn'], label="Buy & Hold", linestyle='--')
plt.title("Strategy vs Benchmark – Cumulative Return")
plt.xlabel("Date")
plt.ylabel("Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

data.to_csv("outputs/day10_output.csv")