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

# applies this to the entire datable column
data.loc[buy_condition, 'Signal'] = 1
data.loc[sell_condition, 'Signal'] = -1

# create a new row called postion
data['Position'] = 0 

#loop through the data set skip the first day due to the fact that there is no yesterday
for i in range(1, len(data)):
    #if signal at i is 1 enter postion to buy
    if data['Signal'].iloc[i] == 1:
        data.iloc[i, data.columns.get_loc('Position')] = 1
    # if signal is -1 we are exiting a trade
    elif data['Signal'].iloc[i] == -1:
        data.iloc[i, data.columns.get_loc('Position')] = 0 
    # other wise just do the same thing as yesterday and hold the stock
    else:
        data.iloc[i, data.columns.get_loc('Position')] = data.iloc[i - 1, data.columns.get_loc('Position')] 

# print(data[['Signal', 'Position']].tail(15)) // testing

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
plt.title("Growth of $100: Strategy vs Holding 100")
plt.ylabel("Growth of $100")
plt.xlabel("Date")
plt.legend()
plt.grid()
plt.show()

# track stats
start_value = 100
end_value = data['Cumulative_Strategy'].iloc[-1]
total_return = end_value - start_value
percent_return = (total_return / start_value) * 100

print("---------------------------------------------------------------------")
print(f"AlphaBot Start Value: ${start_value:.2f}")
print(f"AlphaBot Final Value: ${end_value:.2f}")
print(f"Total Profit: ${total_return:.2f}")
print(f"Total Return: {percent_return:.2f}%")
print("---------------------------------------------------------------------")

data.to_csv("outputs/day2_output.csv")