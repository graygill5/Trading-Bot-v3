import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("outputs/day4_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

def results(data, counter):
    # Calculate trade outcomes
    trades = data[(data['Signal'] == 1) & (data['Position'] == 1)].copy()
    trades['Next_Return'] = data['Strategy_Return'].shift(-1).loc[trades.index]

    # Total return
    start_val = 100
    end_val = data['Cumulative_Strategy'].iloc[-1]
    total_return = end_val - start_val

    # Win rate
    wins = trades['Next_Return'] > 0
    win_rate = (wins.sum() / len(trades)) * 100

    # Average return per trade
    avg_return = trades['Next_Return'].mean() * 100

    # Sharpe ratio (annualized, assuming 252 trading days)
    daily_mean = data['Strategy_Return'].mean()
    daily_std = data['Strategy_Return'].std()
    sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252)

    # Profit Factor
    gross_profit = trades['Next_Return'][wins].sum()
    gross_loss = abs(trades['Next_Return'][~wins].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

    # Max Drawdown (already done, but recalculated here for clarity)
    roll_max = data['Cumulative_Strategy'].cummax()
    drawdown = (data['Cumulative_Strategy'] - roll_max) / roll_max
    max_drawdown = drawdown.min() * 100
    
    # Print Results
    if counter == 0:
        print("DAY 5: Results for full period")
    else:
         print("DAY 5: Results for test period")
    
    print("---------------------------------------------------------------------")
    print(f"End Value: {end_val:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Return per Trade: {avg_return:.4f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print("---------------------------------------------------------------------\n")

def plot_drawdown(data, title="Strategy Drawdown"):
    roll_max = data['Cumulative_Strategy'].cummax()
    drawdown = (data['Cumulative_Strategy'] - roll_max) / roll_max

    plt.figure(figsize=(12, 4))
    drawdown.plot(color='red', label='Drawdown')
    plt.title(title)
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_monthly_returns(data):
    monthly = data['Strategy_Return'].resample('ME').sum() * 100
    monthly.plot(kind='bar', figsize=(20, 5))
    plt.title("Monthly Strategy Returns")
    plt.ylabel("Return (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_trade_histogram(data):
    trades = data[(data['Signal'] == 1) & (data['Position'] == 1)].copy()
    trades['Next_Return'] = data['Strategy_Return'].shift(-1).loc[trades.index]
    
    plt.figure(figsize=(10, 4))
    trades['Next_Return'].hist(bins=50, edgecolor='black')
    plt.title("Distribution of Next-Day Trade Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(data, title="Confusion Matrix: Signal vs Actual Return"):
    # Define ground truth and predictions
    y_true = (data['Yield_Return'].shift(-1) > 0).astype(int)  # 1 if next-day return is positive
    y_pred = (data['Signal'] == 1).astype(int)                # 1 if signal = buy

    # Drop any rows with NaNs due to shift
    mask = ~y_true.isna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Up", "Down"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
test = data.loc['2020-01-01':].copy()
test['Cumulative_Strategy'] = (test['Strategy_Return'] + 1).cumprod() * 100

results(data, 0)
results(test, 1) 

plot_drawdown(data, title="Full Period Drawdown")
plot_drawdown(test, title="Test Period Drawdown")

plot_monthly_returns(data)
plot_monthly_returns(test)

plot_trade_histogram(data)
plot_trade_histogram(test)

plot_confusion_matrix(data, title="Confusion Matrix (Full Period)")
plot_confusion_matrix(test, title="Confusion Matrix (Test Period)")

# Export the data for version control
data.to_csv("outputs/day5_output.csv")