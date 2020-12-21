# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This module Threshold Filter described in Dunis et al. (2005).
This module implements the Correlation Filter described in Dunis et al. (2005).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch.univariate import ZeroMean, EWMAVariance
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
        self.frame = None
        self.transformed = None

    def fit(self, frame: pd.DataFrame):
        """
        Sets the correlation benchmark inside of the class object.

        :param frame: (pd.DataFrame) Time series consisting of both legs of the spread.
        """

        frame = frame.copy()

        self.frame = frame

        two_legged_df = frame.iloc[:, 0:2]
        corr_series = self._get_rolling_correlation(two_legged_df,
                                                    lookback=self.lookback).diff().dropna()

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

        buy_signal = working_frame.index.isin(self.corr_series[self.corr_series > self.buy_threshold].index)
        sell_signal = working_frame.index.isin(self.corr_series[self.corr_series < self.sell_threshold].index)

        working_frame['side'] = 0
        working_frame.loc[buy_signal, 'side'] = 1
        working_frame.loc[sell_signal, 'side'] = -1
        working_frame['side'] = working_frame['side'].shift(1)

        self.transformed = working_frame

        return working_frame

    def plot(self):
        """

        :return: (Axes)
        """

        two_legged_df = self.frame.iloc[:, 0:2]
        spread = two_legged_df.iloc[:, 0] - two_legged_df.iloc[:, 1]

        corr_series = self.corr_series

        corr_events = self.transformed['side']

        plt.figure(figsize=(15, 10))
        plt.subplot(311)
        ax1 = plt.plot(corr_series.diff())
        plt.axhline(y=self.buy_threshold, color='g', linestyle='--')
        plt.axhline(y=self.sell_threshold, color='r', linestyle='--')
        plt.title("Correlation change over time")

        plt.subplot(312)
        ax2 = plt.plot(spread)
        for trade_evnt in spread[corr_events == 1].index:
            plt.axvline(trade_evnt, color="tab:green", alpha=0.2)
        plt.title("Buy Events")

        plt.subplot(313)
        ax3 = plt.plot(spread)
        for trade_evnt in spread[corr_events == -1].index:
            plt.axvline(trade_evnt, color="tab:red", alpha=0.2)
        plt.title("Sell Events")

        return ax1, ax2, ax3

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

        daily_corr = two_legged_df.rolling(lookback,
                                           min_periods=lookback).corr()
        daily_corr = daily_corr.iloc[:, 0].reset_index().dropna()

        final_corr = daily_corr[daily_corr['level_1'] == two_legged_df.columns[1]]
        final_corr.set_index('_index_', inplace=True)
        final_corr.drop(['level_1'], axis=1, inplace=True)
        final_corr.dropna(inplace=True)

        scaler = MinMaxScaler()
        scaled_corr = scaler.fit_transform(final_corr.iloc[:, 0].values.reshape(-1, 1))
        corr_series = pd.Series(data=scaled_corr.reshape(1, -1)[0],
                                index=final_corr.index)
        corr_series.dropna(inplace=True)

        return corr_series

class ThresholdFilter:
    """
    Threshold Filter implementation.
    """

    def __init__(self, buy_threshold: float, sell_threshold: float):
        """
        Initialization of trade parameters.

        :param buy_threshold: (float) If larger than this value, buy.
        :param sell_threshold: (float) If smaller than this value, sell.
        """

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.frame = None
        self.transformed = None

    def plot(self):
        """
        :return: (Axes)
        """

        plt.figure(figsize=(15, 10))

        plt.subplot(211)
        ax1 = plt.plot(self.transformed.iloc[:, 0].cumsum())
        for trade_evnt in self.transformed[self.transformed['side'] == 1].index:
            plt.axvline(trade_evnt, color="tab:green", alpha=0.2)
        plt.title("Buy Events")

        plt.subplot(212)
        ax2 = plt.plot(self.transformed.iloc[:, 0].cumsum())
        for trade_evnt in self.transformed[self.transformed['side'] == -1].index:
            plt.axvline(trade_evnt, color="tab:red", alpha=0.2)
        plt.title("Sell Events")

        return ax1, ax2

    def fit(self, frame: pd.DataFrame):
        """
        Sets the time series to be analyzed inside of the class object.

        :param frame: (pd.DataFrame) Time series to be analyzed.
        """

        self.frame = frame.copy().to_frame()

        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a side column to describe the signal that was detected at
        that point, based on the parameters given in the constructor.

        :param frame: (pd.DataFrame) Time series to be analyzed.
        :return: (pd.DataFrame) Time series augmented with the trade side
            information.
        """

        init_frame = self.frame

        buy_signal = init_frame.index.isin(frame[frame < self.buy_threshold].index)
        sell_signal = init_frame.index.isin(frame[frame > self.sell_threshold].index)

        init_frame['side'] = 0
        init_frame.loc[buy_signal, 'side'] = 1
        init_frame.loc[sell_signal, 'side'] = -1
        init_frame['side'] = init_frame['side'].shift(1)

        self.transformed = init_frame

        return init_frame

    def fit_transform(self, frame) -> pd.DataFrame:
        """
        Convenience method that executes the fit and transform method
        in sequence.

        :param frame: (pd.DataFrame) Time series to be analyzed.
        :return: (pd.DataFrame) Time series augmented with the trade side
            information.
        """

        return self.fit(frame).transform(frame)

class VolatilityFilter:
    """
    Volatility Filter implementation.
    """

    def __init__(self, lookback: int = 80):
        """
        Initialization of trade parameters.
        """

        self.lookback = lookback
        self.sigma = None
        self.frame = None
        self.mu_avg = None
        self.spread = None
        self.vol_series = None
        self.rolling_mean_vol = None
        self.vol_forcast_series = None

    def plot(self):
        """
        :return: (Axes)
        """

        plt.figure(figsize=(15, 10))

        ax1 = plt.subplot(311)
        ax1.plot(self.spread)
        plt.title("Spread Series")

        ax2 = plt.subplot(312, sharex=ax1)
        ax2.plot(self.vol_series["regime"])
        plt.title("Regime States")

        ax3 = plt.subplot(313, sharex=ax1)
        ax3.plot(self.vol_forcast_series)
        plt.title("Forecasted Volatility")

        return ax1, ax2, ax3

    def fit(self, spread: pd.DataFrame):
        """
        Sets the time series to be analyzed inside of the class object.

        :param frame: (pd.DataFrame) Time series to be analyzed.
        """

        self.spread = spread
        self.frame = spread.diff().copy().to_frame()

        # Exponentially Weighted Moving Average Variance, known as RiskMetrics
        risk_metrics_vol = EWMAVariance(0.94)

        # (ZeroMean) - useful if using residuals from a model estimated separately
        zero_mean = ZeroMean(self.frame, volatility=risk_metrics_vol)
        zero_mean.fit()

        vol_forcast_series = pd.Series(zero_mean.resids(self.frame), index=self.frame.index)

        rolling_mean_vol = vol_forcast_series.rolling(window=self.lookback).mean()

        mu_avg = rolling_mean_vol.rolling(window=self.lookback).mean().mean()
        sigma = rolling_mean_vol.rolling(window=self.lookback).std().mean()

        self.vol_forcast_series = vol_forcast_series
        self.rolling_mean_vol = rolling_mean_vol
        self.mu_avg = mu_avg
        self.sigma = sigma

        return self

    def transform(self) -> pd.DataFrame:
        """
        Adds a side column to describe the signal that was detected at
        that point, based on the parameters given in the constructor.

        :return: (pd.DataFrame) Time series augmented with the regime
            information.
        """

        vol_series = self.vol_forcast_series.copy().to_frame()
        vol_series["regime"] = np.nan
        vol_series["leverage_multiplier"] = np.nan

        avg_minus_two_std_dev = self.mu_avg - 2*self.sigma
        avg_minus_four_std_dev = self.mu_avg - 4*self.sigma

        # Extremely Low Regime
        extrm_low_mask = (self.rolling_mean_vol <= avg_minus_four_std_dev)
        vol_series.loc[extrm_low_mask, "regime"] = -3
        vol_series.loc[extrm_low_mask, "leverage_multiplier"] = 2.5

        # Medium Low Regime
        medium_low_mask = (self.rolling_mean_vol <= avg_minus_two_std_dev) & (self.rolling_mean_vol >= avg_minus_four_std_dev)
        vol_series.loc[medium_low_mask, "regime"] = -2
        vol_series.loc[medium_low_mask, "leverage_multiplier"] = 2

        # Higher Low Regime
        higher_low_mask = (self.rolling_mean_vol <= self.mu_avg) & (self.rolling_mean_vol >= avg_minus_two_std_dev)
        vol_series.loc[higher_low_mask, 'regime'] = -1
        vol_series.loc[higher_low_mask, "leverage_multiplier"] = 1.5

        avg_plus_two_std_dev = self.mu_avg + 2*self.sigma
        avg_plus_four_std_dev = self.mu_avg + 4*self.sigma

        # Lower High Regime
        low_high_mask = (self.rolling_mean_vol >= self.mu_avg) & (self.rolling_mean_vol <= avg_plus_two_std_dev)
        vol_series.loc[low_high_mask, 'regime'] = 1
        vol_series.loc[low_high_mask, "leverage_multiplier"] = 1

        # Medium High Regime
        medium_high_mask = (self.rolling_mean_vol >= avg_plus_two_std_dev) & (self.rolling_mean_vol <= avg_plus_four_std_dev)
        vol_series.loc[medium_high_mask, "regime"] = 2
        vol_series.loc[medium_high_mask, "leverage_multiplier"] = 0.5

        # Extremely High Regime
        extrm_high_mask = (self.rolling_mean_vol >= avg_plus_four_std_dev)
        vol_series.loc[extrm_high_mask, "regime"] = 3
        vol_series.loc[extrm_high_mask, "leverage_multiplier"] = 0

        self.vol_series = vol_series

        return vol_series

    def fit_transform(self, frame) -> pd.DataFrame:
        """
        Convenience method that executes the fit and transform method
        in sequence.

        :param frame: (pd.DataFrame) Time series to be analyzed.
        :return: (pd.DataFrame) Time series augmented with the trade side
            information.
        """

        return self.fit(frame).transform()
