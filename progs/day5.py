import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("outputs/day4_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)