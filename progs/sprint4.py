import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("outputs/day3_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

# define parameters like starting cash cooldown and stop loss
stop_loss_threshold = -0.03  
cooldown = 3                  
cost_per_trade = 0.0005      
start_val = 100

# test with data after 2020
test = data.loc['2020-01-01':].copy()

# define function so it can use on test data also
def run_backtest(data):
    # resets the position column
    data['Position'] = 0
    
    #sets paramters for entry price and cooldowns
    in_position = False
    entry_price = None
    cooldown_counter = 0

    # starts at 1 because theres no previous data for day 0
    for i in range(1, len(data)):
        # get the signal for each day
        sig = data['Signal'].iloc[i]

        # if the bot is not in postion to buy, the cool down is over, and its signal is telling you to buy
        if not in_position and cooldown_counter == 0 and sig == 1:
            in_position = True # set to in postion
            entry_price = data['US10Y'].iloc[i] # set entry price
            data.iat[i, data.columns.get_loc('Position')] = 1 # sets the positon to 1 at the index

        # if you're already bought
        elif in_position:
            current_price = data['US10Y'].iloc[i]  # get the current yield
            drawdown = (current_price - entry_price) / entry_price # calc the draw down

            # if your draw down is less than or equal to -.03 or signal is a sell
            if drawdown <= stop_loss_threshold or sig == -1:
                in_position = False # reset position 
                entry_price = None # reset price
                cooldown_counter = cooldown # reset cool down
                data.iat[i, data.columns.get_loc('Position')] = 0 # out of postion
            else:
                data.iat[i, data.columns.get_loc('Position')] = 1 # set position to hold

        # if the cooldown is more that 0 decrement it
        elif cooldown_counter > 0:
            cooldown_counter -= 1

    # calculate the returns
    data['Yield_Return'] = data['US10Y'].pct_change()
    trade_events = data['Position'].diff().abs()
    data['Strategy_Return'] = data['Yield_Return'] * data['Position'] - trade_events * cost_per_trade
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod() * start_val
    data['Cumulative_Yield'] = (1 + data['Yield_Return']).cumprod() * start_val

    return data

# run on both sets of data
data = run_backtest(data)
test = run_backtest(test)

# Figure 1 for all of the data
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Cumulative_Strategy'], label='AlphaBot (Full)', color='gold')
plt.plot(data.index, data['Cumulative_Yield'], label='Buy & Hold (Full)', color='gray')
plt.title("AlphaBot vs Buy & Hold: Full Period")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid()
plt.legend()
plt.show()

# Figure 2 for 2020+ data
plt.figure(figsize=(14, 6))
plt.plot(test.index, test['Cumulative_Strategy'], label='AlphaBot (Test)', color='blue')
plt.plot(test.index, test['Cumulative_Yield'], label='Buy & Hold (Test)', color='black')
plt.title("AlphaBot vs Buy & Hold: Out-of-Sample (2020+)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid()
plt.legend()
plt.show()

# summarizes data
print("DAY 4: Results for full period")
print("---------------------------------------------------------------------")
print(f"Final AlphaBot Value: ${data['Cumulative_Strategy'].iloc[-1]:.2f}")
print(f"Final Buy & Hold:     ${data['Cumulative_Yield'].iloc[-1]:.2f}")
print("---------------------------------------------------------------------\n")

print("DAY 4: Results for testing data")
print("---------------------------------------------------------------------")
print(f"Test AlphaBot Value:  ${test['Cumulative_Strategy'].iloc[-1]:.2f}")
print(f"Test Buy & Hold:      ${test['Cumulative_Yield'].iloc[-1]:.2f}")
print("---------------------------------------------------------------------\n")

# Export the data for version control
data.to_csv("outputs/day4_output.csv")

# bot is over preforming and over fitting pretty bad due to the skewed data in 2020, will fix in day4