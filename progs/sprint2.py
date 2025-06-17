import pandas as pd
import matplotlib.pyplot as plt

# load data from day 1
data = pd.read_csv("outputs/day1_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

# create a new row called signal
data['Signal'] = 0

# checks if there was crossovers in yesterday and today
prev_crossover = data['SMA_5'].shift(1) < data['SMA_20'].shift(1)
curr_crossover = data['SMA_5'] > data['SMA_20']
strong_uptrend = prev_crossover & curr_crossover

# catches early breakouts by catching sharp changes in the yield and confirming its not overbought
early_breakout = (data['US10Y'].pct_change() > 0.01) & (data['RSI_14'] < 70)

# our buy condition will make sure that the stock is not being overbought aswell that there is a strong upward trend in the stock 
buy_condition = strong_uptrend | early_breakout

# our sell condtion is that the SMA10 be smaller meaning that price is decreasing or if the stock is beign overbought 
sell_condition = (data['SMA_5'] < data['SMA_20']) | (data['RSI_14'] > 70)

# applies buy and sell conditons to the entire signal column rest are left as 0's
data.loc[buy_condition, 'Signal'] = 1
data.loc[sell_condition, 'Signal'] = -1

# create a new row called postion
data['Position'] = 0 

# shift signals forward 1 day for realism
data['Position'] = data['Signal'].shift(1).fillna(0)
data['Position'] = data['Position'].replace({-1: 0})

# calculate the daily cahnge of the yield
data['Yield_Return'] = data['US10Y'].pct_change()

# calculates return only on days where the stock was held (anything times 0 = 0)
data['Strategy_Return'] = data['Yield_Return'] * data['Position']

# cumalative yield and strat for 100 dollars
data['Cumulative_Yield'] = (1 + data['Yield_Return']).cumprod() * 100
data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod() * 100

# plot the figures
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Cumulative_Yield'], label='Holding 100', color='gray')
plt.plot(data.index, data['Cumulative_Strategy'], label='AlphaBot', color='gold')
plt.title("DAY 2: Growth of $100: Strategy vs Holding 100")
plt.ylabel("Growth of $100")
plt.xlabel("Date")
plt.legend()
plt.grid()
plt.show()

# Export the data for version control
data.to_csv("outputs/day2_output.csv")