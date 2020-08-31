#!/usr/bin/env python
# coding: utf-8

# ## Calculating the Sharpe Ratio simple example

# Import the required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib styling
plt.style.use('fivethirtyeight')

# Read in the data
stock_data = pd.read_csv('datasets/stock_data.csv', parse_dates=['Date'],
                         index_col='Date').dropna()

benchmark_data = pd.read_csv('datasets/benchmark_data.csv',
                             parse_dates=['Date'], index_col='Date').dropna()


# ## Check the number of observations and variables in the data

# Display summary for stock_data
print(stock_data.info())
print(stock_data.head())


# Display summary for benchmark_data
print(benchmark_data.info())
print(benchmark_data.head())


# ## Plot & summarize daily prices for Amazon and Facebook

# Visualize the stock_data
stock_data.plot(kind='line', subplots=True, title='Stock Data');

# Summarize the stock_data statistically
print(stock_data.describe())


# ## Visualize & summarize daily values for the S&P 500

# Plot the benchmark_data
benchmark_data.plot(kind='line', title='S&P 500')

# Summarize the benchmark_data statistically
print(benchmark_data.describe())


# ## The inputs for the Sharpe Ratio: Starting with Daily Stock Returns

# Calculate daily stock_data returns
stock_returns = stock_data.pct_change()

# Plot the daily returns
stock_returns.plot()

# Summarize the daily returns statistically
print(stock_returns.describe())


# ## Daily S&P 500 returns

# Calculate daily benchmark_data returns

sp_returns = benchmark_data['S&P 500'].pct_change()

# Plot the daily returns
sp_returns.plot()


# Summarize the daily returns statistically
print(sp_returns.describe())


# ## Calculate Excess Returns for Amazon and Facebook vs. S&P 500

# Calculate the difference in daily returns
excess_returns = stock_returns.sub(sp_returns, axis=0)

# Plot the excess_returns
excess_returns.plot()


# Summarize the excess_returns statistically
print(excess_returns.describe())


# ## The Average Difference in Daily Returns Stocks vs S&P 500

# Calculate the mean of excess_returns
avg_excess_return = excess_returns.mean()

# Plot avg_excess_returns
avg_excess_return.plot.bar(title='Mean of the Return Difference')


# ## Standard Deviation of the Return Difference

# Calculate the standard deviations
sd_excess_return = excess_returns.std()

# Plot the standard deviations
sd_excess_return.plot.bar(title='Standard Deviation of the Return Difference')


# ## Compute the Sharpe Ratio by dividing mean excess returns by std

# Calculate the daily Sharpe Ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

# Annualize the Sharpe Ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

# Plot the annualized Sharpe Ratio
annual_sharpe_ratio.plot.bar(title='Annualized Sharpe Ratio: Stocks vs S&P 500')


# ## Conclusion
# Amazon is the better buy choice.
