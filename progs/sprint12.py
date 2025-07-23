import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import RFE

data = pd.read_csv("outputs/day11_output.csv", parse_dates=['date'], index_col='date')

price = (1 + data['Yield_Return']).cumprod()

# movign average divergence 
ema12 = price.ewm(span=12, adjust=False).mean()
ema26 = price.ewm(span=26, adjust=False).mean()
data['MACD'] = ema12 - ema26

# get bandwidhts
sma20 = price.rolling(window=20).mean()
std20 = price.rolling(window=20).std()
upper = sma20 + 2 * std20
lower = sma20 - 2 * std20
data['BB_width'] = (upper - lower) / sma20

# get rsi trends for 14 and 28
delta = price.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain_14 = gain.rolling(14).mean()
avg_loss_14 = loss.rolling(14).mean()
rs14 = avg_gain_14 / avg_loss_14
rsi14 = 100 - 100 / (1 + rs14)

avg_gain_28 = gain.rolling(28).mean()
avg_loss_28 = loss.rolling(28).mean()
rs28 = avg_gain_28 / avg_loss_28
rsi28 = 100 - 100 / (1 + rs28)

data['RSI_trend'] = rsi14 - rsi28

# features
base_feats = ['momentum_3d', 'roc_5d', 'rolling_sharpe_5', 'z_score_10', 'volatility_band','SMA_diff', 'lag_1', 'lag_2', 'volatility_3d', 'Signal','momentum_z', 'roc_volatility']
new_feats = ['MACD', 'BB_width', 'RSI_trend']
all_feats = base_feats + new_feats

# Drop rows with any NaNs from indicator construction
df = data.dropna(subset=all_feats + ['Target']).copy()
X = df[all_feats]
y = df['Target']

# scale data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X),index=X.index,columns=all_feats)

# get PCAs
pca = PCA(n_components=3, random_state=42)
pcs = pca.fit_transform(X_scaled)
pc_cols = [f'PC{i+1}' for i in range(pcs.shape[1])]
X_pca = pd.DataFrame(pcs, index=X_scaled.index, columns=pc_cols)

X_final = pd.concat([X_scaled, X_pca], axis=1)

# selects the best 7 featurs and combines the RFE with the grid seach
rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=7)
rfe.fit(X_final, y)
selected = list(X_final.columns[rfe.support_])

params = { 'n_estimators': [100, 200],'max_depth': [3, 5, 10], 'min_samples_split': [2, 5], }
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),params,cv=tscv,scoring='f1')
grid.fit(X_final[selected], y)
model = grid.best_estimator_

# generate signals again
df['prob'] = model.predict_proba(X_final[selected])[:, 1]
df['Position'] = 0
df.loc[df['prob'] > 0.55, 'Position'] = 1
df.loc[df['prob'] < 0.45, 'Position'] = -1
df['Position'] = df['Position'].shift(1)
df.dropna(subset=['Position'], inplace=True)

df['Daily_Return'] = df['Position'] * df['Yield_Return']
df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
df['Benchmark_CumReturn'] = (1 + df['Yield_Return']).cumprod()

# same as before
total_ret = df['Cumulative_Return'].iloc[-1] - 1
sharpe = df['Daily_Return'].mean() / df['Daily_Return'].std() * np.sqrt(252)
win_rate = (df['Daily_Return'] > 0).mean()
drawdown = df['Cumulative_Return'] / df['Cumulative_Return'].cummax() - 1
max_dd = drawdown.min()
trades = int((df['Position'].diff().abs() > 0).sum())

print("Day 12 Performance")
print(f"Total Return:  {total_ret*100:.2f}%")
print(f"Sharpe Ratio:  {sharpe:.2f}")
print(f"Win Rate:      {win_rate*100:.2f}%")
print(f"Max Drawdown:  {max_dd*100:.2f}%")
print(f"Total Trades:  {trades}")

plt.figure(figsize=(10,5))
plt.plot(df.index, df['Cumulative_Return'], label='Strategy', linewidth=2)
plt.plot(df.index, df['Benchmark_CumReturn'], '--', label='Buy & Hold')
plt.title('Day 12: Expanded Features + PCA Strategy vs Benchmark')
plt.xlabel('Date'); plt.ylabel('Cumulative Return')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

df.to_csv("outputs/day12_output.csv")

# how many days you actually take a position?
n_trades = ((df['Position'] != 0).sum())
pct_trades = n_trades / len(df) * 100
print(f"Trades taken: {n_trades} days ({pct_trades:.2f}% of total days)")

print(df['prob'].describe())
df['prob'].hist(bins=50)

# trades taken (long or short):
trades = df['Position'].diff().abs().fillna(0).astype(bool).sum()
# wins on days you had a position:
wins   = ((df['Daily_Return'] > 0) & (df['Position'] != 0)).sum()

print(f"Trades taken: {trades} ({trades/len(df)*100:.2f}% of days)")
print(f"Win rate (per trade): {wins/trades*100:.2f}%")

results = []
for entry in np.arange(0.50, 0.61, 0.01):
    for exit in np.arange(0.49, 0.40, -0.01):
        df2 = df.copy()
        df2['Position'] = 0
        df2.loc[df2['prob'] > entry, 'Position'] = 1
        df2.loc[df2['prob'] < exit,  'Position'] = -1
        df2['Position'] = df2['Position'].shift(1)
        df2.dropna(subset=['Position'], inplace=True)
        
        trades = df2['Position'].diff().abs().sum()
        win_rate = (df2.loc[df2['Position'] != 0, 'Daily_Return'] > 0).mean()
        cumret   = (1 + df2['Daily_Return']).cumprod().iloc[-1] - 1
        
        results.append({
            'entry': entry, 'exit': exit,
            'trades': int(trades),
            'win_rate': win_rate,
            'total_return': cumret
        })

# display the top configs by win_rate or total_return
grid = pd.DataFrame(results)
best = grid.sort_values('win_rate', ascending=False).head(5)
print(best)