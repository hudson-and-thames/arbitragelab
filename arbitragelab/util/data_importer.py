# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module is a user data helper wrapping various yahoo finance libraries.
"""

import pandas as pd
import yfinance as yf
import yahoo_fin.stock_info as ys

class DataImporter:
    """
    Wrapper class that imports data from yfinance and yahoo_fin.

    This class allows for fast pulling/mangling of information needed
    for the research process. These would include; ticker groups of
    various indexes, pulling of relevant pricing data and processing
    said data.
    """

    @staticmethod
    def get_sp500_tickers() -> list:
        """
        Gets all S&P 500 stock tickers.

        :return: (list) List of tickers.
        """

        tickers_sp500 = ys.tickers_sp500()

        return tickers_sp500

    @staticmethod
    def get_dow_tickers() -> list:
        """
        Gets all DOW stock tickers.

        :return: (list) List of tickers.
        """

        tickers_dow = ys.tickers_dow()

        return tickers_dow

    @staticmethod
    def remove_nuns(dataframe: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
        """
        Remove tickers with nulls in value over a threshold.

        :param dataframe: (pd.DataFrame) Asset price data.
        :param threshold: (int) The number of null values allowed.
        :return dataframe: (pd.DataFrame) Price Data without any null values.
        """

        null_sum_each_ticker = dataframe.isnull().sum()
        tickers_passing = null_sum_each_ticker[null_sum_each_ticker <= threshold]
        tickers_under_threshold = tickers_passing.index
        dataframe = dataframe[tickers_under_threshold]

        return dataframe

    @staticmethod
    def get_price_data(tickers: list, start_date: str, end_date: str,
                       interval: str = '5m') -> pd.DataFrame:
        """
        Get the price data with custom start and end date and interval.
        For daily price, only keep the closing price.

        :param tickers: (list) List of tickers to download.
        :param start_date: (str) Download start date string (YYYY-MM-DD).
        :param end_date: (str) Download end date string (YYYY-MM-DD).
        :param interval: (str) Valid intervals: [1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo].
        :return: (pd.DataFrame) The requested price_data.
        """

        price_data = yf.download(tickers, start=start_date, end=end_date,
                                 interval=interval, group_by='column')['Close']

        return price_data

    @staticmethod
    def get_returns_data(price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate return data with custom start and end date and interval.

        :param price_data: (pd.DataFrame) Asset price data.
        :return: (pd.DataFrame) Price Data converted to returns.
        """

        returns_data = price_data.pct_change()
        returns_data = returns_data.iloc[1:]

        return returns_data

    def get_ticker_sector_info(self, tickers: list, yf_call_chunk: int = 20) -> pd.DataFrame:
        """
        This method will loop through all the tickers, using the yfinance library
        do a ticker info request and retrieve back 'sector' and 'industry' information.

        This method uses the yfinance 'Tickers' object which has a limit of the amount of
        tickers supplied as a string argument. To go around this, this method uses the
        chunking approach, where the supplied ticker list is broken down into small chunks
        and supplied sequentially to the helper function.

        :param tickers: (list) List of asset symbols.
        :param yf_call_chunk: (int) Ticker values allowed per 'Tickers'
            object. This should always be less than 200.
        :return: (pd.DataFrame) DataFrame with input asset tickers and their
            respective sector and industry information.
        """

        ticker_sector_queue = []

        # For each chunk of size 'yf_call_chunk'.
        for i in range(0, len(tickers), yf_call_chunk):

            # Set end as the limit value equals to the chunk size.
            # If we hit the last chunk, set the end value as the
            # full length of the ticker list.
            end = i+yf_call_chunk if i <= len(tickers) else len(tickers)

            ticker_sector_queue.append(self._sector_info_helper(tickers[i: end]))

        return pd.concat(ticker_sector_queue, axis=0).reset_index(drop=True)

    @staticmethod
    def _sector_info_helper(tickers: list) -> pd.DataFrame:
        """
        Helper method to supply chunked sector info to the main method.

        :param tickers: (list) List of asset symbols.
        :return: (pd.DataFrame) DataFrame with input asset tickers and their respective sector
            and industry information.
        """

        tckrs = yf.Tickers(' '.join(tickers))

        tckr_info = []

        for i, tckr in enumerate(tickers):
            ticker_info = tckrs.tickers[i].info
            tckr_tuple = (tckr, ticker_info['industry'], ticker_info['sector'])
            tckr_info.append(tckr_tuple)

        return pd.DataFrame(data=tckr_info, columns=['ticker', 'industry', 'sector'])
