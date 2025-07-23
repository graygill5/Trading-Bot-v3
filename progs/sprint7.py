import pandas as pd

# load data 
data = pd.read_csv("outputs/day6_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

# adding new features 
data['momentum_3d'] = data['US10Y'].pct_change(3)

# rate of change over 5 days
data['roc_5d'] = data['US10Y'].pct_change(5)

# rolling Sharpe ratio (5-day)
roll_mean = data['Yield_Return'].rolling(5).mean()
roll_std = data['Yield_Return'].rolling(5).std()
data['rolling_sharpe_5'] = roll_mean / roll_std

# Z-score (how far from 10-day mean)
data['z_score_10'] = (data['US10Y'] - data['US10Y'].rolling(10).mean()) / data['US10Y'].rolling(10).std()

# 5-day average volume-like proxy using volatility
data['volatility_band'] = data['Yield_Return'].rolling(5).std()

# drops NAs 
data.dropna(inplace=True)

# save for version controll
data.to_csv("outputs/day7_output.csv")