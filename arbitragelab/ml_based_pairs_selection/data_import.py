"""
This module is a user data helper wrapping various yahoo finance libraries
"""

import pandas as pd

import yfinance as yf
import yahoo_fin.stock_info as ys


class ImportData:
    """
    Wrapper class that imports data from yfinance and yahoo_fin.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_sp500_tickers() -> list:
        """
        Gets all S&P 500 stock tickers.

        :return tickers: (list) : list of tickers
        """
        tickers_sp500 = ys.tickers_sp500()

        return tickers_sp500

    @staticmethod
    def get_dow_tickers() -> list:
        """
        Gets all DOW stock tickers.

        :return tickers: (list) : list of tickers
        """
        tickers_dow = ys.tickers_dow()

        return tickers_dow

    @staticmethod
    def remove_nuns(dataframe: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
        """
        Remove tickers with nuns in value over a threshold.

        Parameters
        ----------
        df : pandas dataframe
            Price time series dataframe

        threshold: int, OPTIONAL
            The number of null values allowed
            Default is 100

        Returns
        -------
        df : pandas dataframe
            Updated price time series without any nuns
        """
        null_sum_each_ticker = dataframe.isnull().sum()
        tickers_under_threshold = \
            null_sum_each_ticker[null_sum_each_ticker <= threshold].index
        dataframe = dataframe[tickers_under_threshold]

        return dataframe

    @staticmethod
    def get_price_data(tickers: list,
                       start_date: str,
                       end_date: str,
                       interval: str = '5m') -> pd.DataFrame:
        """
        Get the price data with custom start and end date and interval.

        No Pre and Post market data.
        For daily price, only keep the closing price.

        Parameters
        ----------
        start_date : str
            Download start date string (YYYY-MM-DD) or _datetime.
        end_date : str
            Download end date string (YYYY-MM-DD) or _datetime.
        interval : str, OPTIONAL
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
            Default is '5m'
        tickers : str, list
            List of tickers to download

        Returns
        -------
        price_data : pandas dataframe
            The requested price_data
        """
        price_data = yf.download(tickers,
                                 start=start_date, end=end_date,
                                 interval=interval,
                                 group_by='column')['Close']

        return price_data

    @staticmethod
    def get_returns_data(price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate return data with custom start and end date and interval.

        :param price_data: (pd.DataFrame) : Asset price data
        :return returns_df: (pd.DataFrame) : Price Data converted to returns.
        """
        returns_data = price_data.pct_change()
        returns_data = returns_data.iloc[1:]

        return returns_data

    def get_ticker_sector_info(self, tickers: list, yf_call_chunk: int = 20) -> pd.DataFrame:
        """
        This method will loop through all the tickers tagged by sector and generate
        pairwise combinations of the assets for each sector.

        :param tickers: (list) : List of asset name pairs
        :return augmented_tickers: (pd.DataFrame) : DataFrame with input asset tickers and their respective sector and industry information
        """

        if len(tickers) > yf_call_chunk:
            ticker_sector_queue = []
            for i in range(0, len(tickers), yf_call_chunk):
                end = i+yf_call_chunk if i <= len(tickers) else len(tickers)
                ticker_sector_queue.append(
                    self.get_ticker_sector_info(tickers[i: end]))
            return pd.concat(ticker_sector_queue, axis=0).reset_index(drop=True)

        tckrs = yf.Tickers(' '.join(tickers))

        tckr_info = []

        for i, tckr in enumerate(tickers):
            try:
                ticker_info = tckrs.tickers[i].info
                tckr_info.append(
                    (tckr, ticker_info['industry'], ticker_info['sector']))
            except ValueError:
                pass
            except RuntimeError:
                pass

        return pd.DataFrame(data=tckr_info, columns=['ticker', 'industry', 'sector'])
