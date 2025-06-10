import pandas as pd

data = pd.read_csv("outputs/day2_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

# start with 100 bucks
start_value = 100

# get the end value and calculate the total return and percent return ration
end_value = data['Cumulative_Strategy'].iloc[-1]
total_return = end_value - start_value
percent_return = (total_return / start_value) * 100

# count how many times you're buying and selling
buy_signals = (data['Signal'] == 1).sum()
sell_signals = (data['Signal'] == -1).sum()

# filter for only buys/trades and then create a new column that tracks the return for the next day
trades = data[data['Signal'] == 1].copy()
trades['Next_Return'] = data['Strategy_Return'].shift(-1).loc[trades.index]

# Count trade wins
wins = (trades['Next_Return'] > 0).sum()
total_trades = len(trades)
win_rate = wins / total_trades * 100

# calculate the max drawdown by taking the highest value seen at the time and subtracting from teh current value, then find the lowest using min
roll_max = data['Cumulative_Strategy'].cummax()
drawdown = (data['Cumulative_Strategy'] - roll_max) / roll_max
max_drawdown = drawdown.min() * 100

# calc avg return
avg_return = trades['Next_Return'].mean() * 100

# calc the sharpe ratio using the 260 days in a year you can trade minus 8 days for holidays (252 for standard)
strategy_returns = data['Strategy_Return'].dropna()
sharpe = (strategy_returns.mean() / strategy_returns.std()) * (252 ** 0.5)


# prints
print("DAY 3: Results")
print("---------------------------------------------------------------------")
print(f"AlphaBot Start Value: ${start_value:.2f}")
print(f"AlphaBot Final Value: ${end_value:.2f}")
print(f"Total Profit: ${total_return:.2f}")
print(f"Total Return: {percent_return:.2f}%")
print(f"Buy Signals: {buy_signals} | Sell Signals: {sell_signals}")
print(f"Win Rate: {win_rate:.2f}% ({wins} wins / {total_trades} trades)")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print(f"Avg Return per Trade: {avg_return:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print("---------------------------------------------------------------------")

# Export the data for version control
data.to_csv("outputs/day3_output.csv")

#Summary: for days 4 and 5 work on tuning the bot, max drawdown is too high and possibly making too many sell trades