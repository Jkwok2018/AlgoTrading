# allows for reading in data from sources
import pandas_datareader as pdr
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
# from pandas.core import datetools

# a dataframe (2D labeled data structure with columns of potentially different types)
aapl = pdr.get_data_yahoo('AAPL',
                           start=datetime.datetime(2006, 10, 1),
                           end=datetime.datetime(2012, 1, 1))

# get first row of data
aapl.head()

# save data to a csv file
aapl.to_csv('aapl_ohlc.csv')
# read data from csv
df = pd.read_csv('aapl_ohlc.csv', header=0, index_col='Date', parse_dates=True)

# get the index, in this case, it will be an array of dates
aapl.index
# get the column attributes (['High','Low',...])
aapl.columns
# get the subset from the last 10 data points of the close column
ts = aapl['Close'][-10:]
# ts will be type Series (1D labeled array)
type(ts)

# loc() is label-baed indexing
# Inspect the first rows of November-December 2006
aapl.loc[pd.Timestamp('2006-11-01'):pd.Timestamp('2006-12-31')].head()

# iloc() is positional indexing
# Inspect  November 2006
aapl.iloc[22:43]

# resample() data into monthly level instead of daily
# the option 'M' makes it monthly
monthly_aapl = aapl.resample('M').mean()

#########   Visualize Time Series Data   #########  
# Plot the closing prices for `aapl`
# aapl['Close'].plot(grid=True)

#########   Returns   #########  
# Assign `Adj Close` to `daily_close`
daily_close = aapl[['Adj Close']]
# Daily returns
daily_pct_change = daily_close.pct_change()
# This is the daily return, after replacing NA values with 0
daily_pct_change.fillna(0, inplace=True)
# Daily log returns
daily_log_returns = np.log(daily_close.pct_change()+1)

# Resample `aapl` to business months, take last observation as value 
monthly = aapl.resample('BM').apply(lambda x: x[-1])
# Calculate the monthly percentage change
monthly.pct_change()

# Resample `aapl` to quarters, take the mean as value per quarter
quarter = aapl.resample("4M").mean()
# Calculate the quarterly percentage change
quarter.pct_change()

# Plot the distribution of `daily_pct_change`
# daily_pct_change.hist(bins=50)

# Calculate the cumulative daily returns
# cumprod is the cumalative product of the array 
# (multiply all the previous daily percent change)
cum_daily_return = (1 + daily_pct_change).cumprod()
# Plot the cumulative daily returns
# cum_daily_return.plot(figsize=(12,8))

#########   Getting more data   #########  
# ticker is the symbol of the stock
def get(tickers, startdate, enddate):
    def data(ticker):
        return pdr.get_data_yahoo(ticker, start=startdate, end=enddate)
    # map the data with the right tickers and return a dataframe that concates them
    datas = map(data, tickers)
    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']
all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2012, 1, 1))

#########   Histogram for multiple stocks   #########  
# Isolate the `Adj Close` values and transform the DataFrame
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')
# Calculate the daily percentage change for `daily_close_px`
daily_pct_change = daily_close_px.pct_change()
# Plot the distributions
# daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))

#########   Scatter matrix   #########  
# Plot a scatter matrix with the `daily_pct_change` data 
# KDE is a kernel density estimate: estimates the 
# probability density function of a random variable
# alpha is the transparency
# pd.plotting.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1,figsize=(12,12))

#########   Moving Windows   ######### 
# Isolate the adjusted closing prices 
adj_close_px = aapl['Adj Close']

# Calculate the moving average
# add a column in aapl called 40 that takes the mean of a 40 point window
aapl['40'] = adj_close_px.rolling(window=40).mean()
# add a column with window size 252
aapl['252'] = adj_close_px.rolling(window=252).mean()
# Plot the adjusted closing price, the short and long windows of rolling means
# aapl[['Adj Close', '40', '252']].plot()

#########   Volatility Calculation   #########  
# voltality: a measurement of the change in variance in the return over a specific period
# voltality == risk in the stock
# Define the minumum of periods to consider 
min_periods = 75 
# Calculate the volatility
# the moving historical standard deviation of the log returns
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods) 
# Plot the volatility
# figure size matters - if window wider and min_periods larger, then result will be
# less representative. the other way, result will be closer to standard deviation
# vol.plot(figsize=(10, 8))

#########   Ordinary Least-Square Regression (OLS)   #########  
# OLS is a traditional regression analysis
# statsmodels library allow for different statistical models
# Isolate the adjusted closing price
all_adj_close = all_data[['Adj Close']]
# Calculate the returns 
all_returns = np.log(all_adj_close / all_adj_close.shift(1))
# Isolate the AAPL returns
# get_level_values() is a pandas function that returns the list whose name is the input
# in this case, it returns the list of stock names
aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']
# droplevel returns the dataframe with the requested index removed
aapl_returns.index = aapl_returns.index.droplevel('Ticker')

# Isolate the MSFT returns
msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']
msft_returns.index = msft_returns.index.droplevel('Ticker')

# Build up a new DataFrame with AAPL and MSFT returns
# concat along the column axis, without the index
# added [1:] so that NaN values won't interfere with model
return_data = pd.concat([aapl_returns, msft_returns], axis=1)[1:]
return_data.columns = ['AAPL', 'MSFT']

# Add a a column of ones to an array
X = sm.add_constant(return_data['AAPL'])
# Construct the model
model = sm.OLS(return_data['MSFT'],X).fit()
# Print the summary
# print(model.summary())

# Plot returns of AAPL and MSFT
# plt.plot(return_data['AAPL'], return_data['MSFT'], 'r.')
# # Add an axis to the plot
# ax = plt.axis()
# # Initialize `x` (x axis)
# x = np.linspace(ax[0], ax[1] + 0.01)
# # # Plot the regression line
# plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)

# # # Customize the plot
# plt.grid(True)
# plt.axis('tight')
# plt.xlabel('Apple Returns')
# plt.ylabel('Microsoft returns')

# Plot the rolling correlation
# return_data['MSFT'].rolling(window=252).corr(return_data['AAPL']).plot()

#########   Simple Trading Strategy   #########  
'''
Two common trading strategies:
1. momentum/divergence/trend: believe the movement will continue in its 
current direction e.g.
    -moving average crossover: when priace moves one side of an average 
    to another, represents a change in momentum --> decide to enter or exit
    -dual moving average crossover, turtle trading
2. reversion/convergence/cycle: movement will eventually reverse e.g.
    -mean reversion: stock return to their mean
    -pair trading mean-reverison: if two stocks have high correlation, change 
    in one can signal change in another
'''
# the moving average crossover strategy
# create 2 Simple Moving Average (SMA) - one short and one long
short_window = 40
long_window = 100
# make empty dataframe then copy the index of aapl(dates) to calculate daily buy/sell signal
signals = pd.DataFrame(index=aapl.index)
signals['signal'] = 0.0
# create the set of short and long SMA
signals['short_mavg']= aapl['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
signals['long_mavg']= aapl['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
# if short > long, signal becomes 1 (enter); long > short, signal is 0 (exit)
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] >
                                     signals['long_mavg'][short_window:], 1.0, 0.0)
# Generate trading orders by calculating the difference of the signals
signals['positions'] = signals['signal'].diff()

# # Initialize the plot figure
# fig = plt.figure()
# # Add a subplot and label for y-axis
# ax1 = fig.add_subplot(111,  ylabel='Price in $')
# # Plot the closing price
# aapl['Close'].plot(ax=ax1, color='r', lw=2.)
# # Plot the short and long moving averages
# signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
# # Plot the buy signals
# ax1.plot(signals.loc[signals.positions == 1.0].index, 
#          signals.short_mavg[signals.positions == 1.0],
#          '^', markersize=10, color='m')
# # Plot the sell signals
# ax1.plot(signals.loc[signals.positions == -1.0].index, 
#          signals.short_mavg[signals.positions == -1.0],
#          'v', markersize=10, color='k')

#########   Simple Backtesting for the Strategy   #########
'''
    Backtesting usually consists of 4 components:
    1. data handle: interface to a set of data
    2. strategy: generates a signal to go long/short
    3. portfolio: generates orders and manages Profit and Loss(PnL)
    4. execution handler: sends orer to broker and recieves fills(signals)
    that confirm execution
'''

# creating a portfolio
# set initial variable and new dataframe with index
initial_capital = float(100000.0)
positions = pd.DataFrame(index=signals.index)
# create column so that when signal is 1, buy a 100 share, else do nothing
positions['AAPL'] =100*signals['signal']
# new dataframe to store market value of an open position
portfolio = positions.multiply(aapl['Adj Close'], axis=0)
# new dataframe to store difference in position (# of stocks)
pos_diff = positions.diff()
# create new column to portfolio that store the value of shares you bought*Adj close
portfolio['holdings'] = (positions.multiply(aapl['Adj Close'], axis=0)).sum(axis=1)
# new column that store capital you still have left to spend
# initial_captial - current holdings (price paid for buying stocks)
portfolio['cash'] = initial_capital - (pos_diff.multiply(aapl['Adj Close'], 
                    axis=0)).sum(axis=1).cumsum() 
# new column of total 
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
# Add `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()

# Print the first lines of `portfolio`
# print(portfolio.head())
# AAPL  holdings      cash     total  returns
# Date                                                   
# 2006-10-02   0.0       0.0  100000.0  100000.0      NaN
# 2006-10-03   0.0       0.0  100000.0  100000.0      0.0
# 2006-10-04   0.0       0.0  100000.0  100000.0      0.0

# Create a figure
# fig = plt.figure()
# ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
# # Plot the equity curve in dollars
# portfolio['total'].plot(ax=ax1, lw=2.)
# ax1.plot(portfolio.loc[signals.positions == 1.0].index, 
#          portfolio.total[signals.positions == 1.0],
#          '^', markersize=10, color='m')
# ax1.plot(portfolio.loc[signals.positions == -1.0].index, 
#          portfolio.total[signals.positions == -1.0],
#          'v', markersize=10, color='k')

#########   Evaluate Moving Average Crossover Strategy   #########  
# Sharpe ratio: ratio between returns and risk; greater the ratio, better the strategy
# Isolate the returns of your strategy
returns = portfolio['returns']
# annualized Sharpe ratio
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())

# calculate Maximum Drawdown: measure the largest single drop from peak to bottom
# Define a trailing 252 trading day window
window = 252
# Calculate the max drawdown in the past window days for each day 
rolling_max = aapl['Adj Close'].rolling(window, min_periods=1).max()
daily_drawdown = aapl['Adj Close']/rolling_max - 1.0
# Calculate the minimum (negative) daily drawdown
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

# Plot the results
daily_drawdown.plot()
max_daily_drawdown.plot()

# calculate Compound Annual Growth Rate: constant rate of return over the period
# tells you what you really have at the  end of investment period
# Get the number of days in `aapl`
days = (aapl.index[-1] - aapl.index[0]).days
# Calculate the CAGR 
cagr = ((((aapl['Adj Close'][-1]) / aapl['Adj Close'][1])) ** (365.0/days)) - 1
# Print the CAGR
print(cagr)
plt.show()