import numpy as np
import pandas as pd
import yfinance as yf


def cumul_from_prices(df):
    """
    Compute cumulative returns of a time series. Represented as a ratio multiplier
    for an initial principle of 1 at the start of the series.

    ex. {2016-01-04 0.97} means 0.97 times the value of the initial principle will exist
    in a buy-and-hold strategy for this time series. It is not a raw percentage.
    
    Parameters
    ----------
    df : pd.Series or pd.DataFrame
        Price time series

    Returns
    -------
    pd.Series or pd.DataFrame
        Cumulative ratio of returns 
    
    """
    return cumul_from_noncumul(noncumul_from_prices(df))

def cumul_from_noncumul(df):
    """
    Compute cumulative returns from non-cumulative returns. Assumes
    returns are non-cumulative percentages expressed as decimal values. 
    (ie. 1% return on a given time stamp is 0.01 in the dataframe)

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Noncumulative return percentage of time series

    Returns
    -------
    pd.Series
        Daily noncumulative returns of the strategy.
    """
    if isinstance(df, pd.DataFrame):
        return df.apply(lambda column : (1 + column).cumprod(),axis = 0)
    else:
        return (1 + df).cumprod()

def noncumul_from_prices(df):
    """
    Compute percent change for each column in a dataframe. A 1% chance corresponds
    to 0.01. 

    Parameters
    ----------
    Values : pd.DataFrame or pd.Series
        Values dataframe or series, noncumulative.

    Returns
    -------
    percent_change : pd.DataFrame or pd.Series
        Percent change of each dataframe column or given series, first entry is padded with 0
    """
    return df.pct_change().fillna(0)

def cumul_noncumul_from_tickers(tickers, start, end, interval = '1d', ohlc= "Adj Close", nafill = "ffill"):
    """
    Given a set of tickers and a date range, return 2 dataframes: one with non-cumulative percent returns,
    and one with a cumulative returns ratio. 
    
    By default, interpolates missing data using forward filling.


    Parameters
    ----------
    tickers : list
        List of valid NASDAQ tickers or currency pairs to acquire returns of
    start : str or dt.datetime
        Starting date for the data
    end : str or dt.datetime
        Ending date for the data
    interval : str
        How frequently to sample the data {"1h" : 1 hour, "1d" : 1 day, "1y" : 1 year, "1mo" : 1 month, "1m" : 1 minute}
        higher sampling frequencies are limited to how far back they can go by yfinance. For example, 1h returns are available 
        only for the last 730 days.
    ohlc : str
        which part of the candlestick to use. By default it uses the adjusted close
    nafill : str
        Method used to fill missing data for returns. By default it forward fills

    Returns
    -------
    (non_cumul, cumul) : (pd.DataFrame, pd.DataFrame)
        Tuple of non-cumulative percentage returns, and cumulative return ratios of a given asset excluding the first day
    """
    prices_df = prices_from_tickers(tickers = tickers, start = start, end = end, interval = interval, ohlc = ohlc, nafill = nafill)
    if isinstance(prices_df, pd.DataFrame):
        prices_df = prices_df.dropna(axis = 1, how = "any").dropna()
    return noncumul_from_prices(prices_df), cumul_from_prices(prices_df)

def prices_from_tickers(tickers, start, end, interval = '1d', ohlc= "Adj Close", nafill = "ffill"):
    """
    Given a set of tickers and a date range, return dataframe of their Adjusted Close prices in time.

    Parameters
    ----------
    tickers : list
        List of valid NASDAQ tickers or currency pairs to acquire returns of
    start : str or dt.datetime
        Starting date for the data
    end : str or dt.datetime
        Ending date for the data
    interval : str
        How frequently to sample the data {"1h" : 1 hour, "1d" : 1 day, "1y" : 1 year, "1mo" : 1 month, "1m" : 1 minute}
        higher sampling frequencies are limited to how far back they can go by yfinance. For example, 1h returns are available 
        only for the last 730 days.
    ohlc : str
        which part of the candlestick to use. By default it uses the adjusted close, options include 
        "Open", "High", "Low", "Close", "Adj Close", input is case-sensitive.
    nafill : str
        Method used to fill missing data for returns. By default it forward fills (ie. missing data is replaced with what value
        it was previously)

    Returns
    -------
    prices : pd.DataFrame
        Dataframe of historical prices of each ticker
    """
    prices_df = yf.download(" ".join(tickers), start, end, interval = interval)[ohlc].fillna(method = nafill)
    if prices_df.empty:
        raise ValueError("No data found for tickers")
    if isinstance(prices_df, pd.DataFrame):
        prices_df = prices_df.dropna(axis = 1, how = "all").dropna()
    if isinstance(prices_df, pd.Series):
        prices_df.name = tickers[0]
    return prices_df
    