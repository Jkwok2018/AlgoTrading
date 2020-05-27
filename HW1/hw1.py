# allows for reading in data from sources
import pandas_datareader as pdr
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
aapl['Close'].plot(grid=True)

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
daily_pct_change.hist(bins=50)

# Calculate the cumulative daily returns
# cumprod is the cumalative product of the array 
# (multiply all the previous daily percent change)
cum_daily_return = (1 + daily_pct_change).cumprod()
# Plot the cumulative daily returns
cum_daily_return.plot(figsize=(12,8))

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
daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))

#########   Scatter matrix   #########  
# Plot a scatter matrix with the `daily_pct_change` data 
# KDE is a kernel density estimate: estimates the 
# probability density function of a random variable
# alpha is the transparency
pd.plotting.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1,figsize=(12,12))

#https://www.datacamp.com/community/tutorials/finance-python-trading
#########   Moving Windows   #########  
#########   Volatility Calculation   #########  
#########   Ordinary Least-Square Regression (OLS)   #########  
#########   Simple Trading Strategy   #########  
#########   Simple Backtesting for the Strategy   #########  
#########   Backtesting with Zipline & Quantopian   #########  
#########   Evaluate Moving Average Crossover Strategy   #########  
