# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This module implements the Correlation Filter described in Dunis et al. (2005).
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class CorrelationFilter:
    """
    Correlation Filter implementation.
    """

    def __init__(self, buy_threshold: float = 0.4, sell_threshold: float = 0.8, lookback: int = 30):
        """
        Initialization of trade parameters. The buy/sell threshold are values in terms
        of change in correlation.

        :param buy_threshold: (float) If larger than this value, buy.
        :param sell_threshold: (float) If smaller than this value, sell.
        :param lookback: (int) Number of lookback days for rolling correlation.
        """

        self.lookback = lookback
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.corr_series = None

    def fit(self, frame: pd.DataFrame):
        """
        Sets the correlation benchmark inside of the class object.

        :param frame: (pd.DataFrame) Time series consisting of both legs of the spread.
        """

        frame = frame.copy()

        two_legged_df = frame.iloc[:, 0:2]
        corr_series = self._get_rolling_correlation(
            two_legged_df, lookback=self.lookback).diff().dropna()

        self.corr_series = corr_series

        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Marks trade signals based on the correlation benchmark generated in the fit
        method.

        :param frame: (pd.DataFrame) Spread time series.
        :return: (pd.DataFrame) Time series augmented with the trade side
            information.
        """

        working_frame = frame.copy()

        buy_signal = working_frame.index.isin(
            self.corr_series[self.corr_series > self.buy_threshold].index)
        sell_signal = working_frame.index.isin(
            self.corr_series[self.corr_series < self.sell_threshold].index)

        working_frame['side'] = 0
        working_frame.loc[buy_signal, 'side'] = 1
        working_frame.loc[sell_signal, 'side'] = -1
        working_frame['side'] = working_frame['side'].shift(1)

        return working_frame

    @staticmethod
    def _get_rolling_correlation(frame: pd.DataFrame, lookback: int) -> pd.Series:
        """
        Calculates rolling correlation between the first two columns in the frame variable.
        Assuming that the first two columns are the opposing legs of the spread.

        :param frame: (pd.DataFrame) DataFrame representing both legs of the spread.
        :param lookback: (int) The lookback range of the rolling mean.
        :param scale: (bool) If True the correlation range will be changed from
            the usual [-1, 1] to [0, 1].
        :return: (pd.Series) Rolling correlation series of the input frame.
        """

        two_legged_df = frame.iloc[:, 0:2]
        two_legged_df.index.name = '_index_'

        daily_corr = two_legged_df.rolling(
            lookback, min_periods=lookback).corr()
        daily_corr = daily_corr.iloc[:, 0].reset_index().dropna()

        final_corr = daily_corr[daily_corr['level_1']
                                == two_legged_df.columns[1]]
        final_corr.set_index('_index_', inplace=True)
        final_corr.drop(['level_1'], axis=1, inplace=True)
        final_corr.dropna(inplace=True)

        scaler = MinMaxScaler()
        scaled_corr = scaler.fit_transform(
            final_corr.iloc[:, 0].values.reshape(-1, 1))  # .diff()
        corr_series = pd.Series(data=scaled_corr.reshape(
            1, -1)[0], index=final_corr.index)
        corr_series.dropna(inplace=True)

        return corr_series
