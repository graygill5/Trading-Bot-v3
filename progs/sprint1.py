#include all the nessecary imports
import pandas as pd
import matplotlib.pyplot as plt
import ta

# Load the data set
file_path = "us_treasury_yields_daily.csv"
data = pd.read_csv(file_path) 

# print(data.columns.tolist()) // finds the column names

# turns into real calendar dates and then sorts them by data and sets index
data['date'] = pd.to_datetime(data['date'])   
data = data.sort_values('date')                  
data.set_index('date', inplace=True)

# Selects 10 year as the column to analyze, also removes values without a yield to clean up
yield_col = 'US10Y'                             
data[yield_col] = pd.to_numeric(data[yield_col], errors='coerce') 
data = data.dropna(subset=[yield_col])

# will help smooth out the data from noise with short term and long term aswell as the momentum for yield
data['SMA_5'] = ta.trend.sma_indicator(data[yield_col], window=10)
data['SMA_20'] = ta.trend.sma_indicator(data[yield_col], window=30)
data['RSI_14'] = ta.momentum.rsi(data[yield_col], window=14)

# create a plot figure for SMA or Simple Moving Averages
plt.figure(figsize=(14, 6))
plt.plot(data.index, data[yield_col], label='Yield', linewidth=1.5)
plt.plot(data.index, data['SMA_5'], label='SMA 5', linestyle='--', color = 'blue')
plt.plot(data.index, data['SMA_20'], label='SMA 20', linestyle='--', color = 'orange')
plt.title(f" DAY 1: {yield_col} Yield with SMAs")
plt.xlabel("Date")
plt.ylabel("Yield (%)")
plt.legend()
plt.grid()
plt.show()

# create a plot figure for RSI or Relative Strength Index (high RSI = overbought) (low RSI = underbought)
plt.figure(figsize=(14, 4))
plt.plot(data.index, data['RSI_14'], color='purple', label='RSI 14')
plt.axhline(70, linestyle='--', color='red')    # Overbought threshold
plt.axhline(30, linestyle='--', color='green')  # Oversold threshold
plt.title(f"DAY 1: {yield_col} RSI (14-day/2-week)")
plt.xlabel("Date")
plt.ylabel("RSI Value")
plt.legend()
plt.grid()
plt.show()

# Export the data for version control
data.to_csv("outputs/day1_output.csv")