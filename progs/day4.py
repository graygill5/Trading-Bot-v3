import pandas as pd

data = pd.read_csv("outputs/day3_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)