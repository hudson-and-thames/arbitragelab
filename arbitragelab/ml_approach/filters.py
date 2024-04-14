"""
This module implements the Correlation, Threshold, and Volatility Filters described in Dunis et al. (2005).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from arch.univariate import ZeroMean, EWMAVariance

from arbitragelab.util import segment

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

        segment.track('CorrelationFilter')

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

        # Get all the events at the specified threshold range as a list. Then
        # use it to convert those individual dates into a boolean mask to be
        # used on the working_frame variable to set each side.
        buy_signal = working_frame.index.isin(self.corr_series[self.corr_series > self.buy_threshold].index)
        sell_signal = working_frame.index.isin(self.corr_series[self.corr_series < self.sell_threshold].index)

        working_frame['side'] = 0
        working_frame.loc[buy_signal, 'side'] = 1
        working_frame.loc[sell_signal, 'side'] = -1
        working_frame['side'] = working_frame['side'].shift(1)

        self.transformed = working_frame

        return working_frame

    def plot(self) -> list:
        """
        Function to plot correlation change, buy and sell events.

        :return: (list) List of Axes objects.
        """

        two_legged_df = self.frame.iloc[:, 0:2]

        # Calculate naive spread.
        spread = two_legged_df.iloc[:, 0] - two_legged_df.iloc[:, 1]

        # Calculate correlation series.
        corr_series = self.corr_series

        # Get all buy/sell/nothing events.
        corr_events = self.transformed['side']

        plt.figure(figsize=(15, 10))

        # Plot correlation change through time and set the
        # given buy/sell threshold as horizontal lines.
        plt.subplot(311)
        ax1 = plt.plot(corr_series.diff())
        plt.axhline(y=self.buy_threshold, color='g', linestyle='--')
        plt.axhline(y=self.sell_threshold, color='r', linestyle='--')
        plt.title("Correlation change over time")

        # Plot buy events triggered by the change in correlation.
        plt.subplot(312)
        ax2 = plt.plot(spread)
        for trade_evnt in spread[corr_events == 1].index:
            plt.axvline(trade_evnt, color="tab:green", alpha=0.2)
        plt.title("Buy Events")

        # Plot sell events triggered by the change in correlation.
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
        :return: (pd.Series) Rolling correlation series of the input frame.
        """

        two_legged_df = frame.iloc[:, 0:2]
        two_legged_df.index.name = '_index_'

        # Get rolling correlation vector for the legs.
        daily_corr = two_legged_df.rolling(lookback,
                                           min_periods=lookback).corr()
        # The DataFrame index is reset to make the MultiIndex more
        # workable.
        daily_corr = daily_corr.iloc[:, 0].reset_index().dropna()

        # We select a level from the previously reset correlation index, and specify a
        # column to be selected from the matrix.
        final_corr = daily_corr[daily_corr['level_1'] == two_legged_df.columns[1]]
        final_corr.set_index('_index_', inplace=True)
        # Cleanup of some of the duplicate correlation data.
        final_corr = final_corr.drop(['level_1'], axis=1)
        final_corr = final_corr.dropna()

        # Here we scale the correlation data to [0, 1]
        scaler = MinMaxScaler()
        scaled_corr = scaler.fit_transform(final_corr.iloc[:, 0].values.reshape(-1, 1))
        corr_series = pd.Series(data=scaled_corr.reshape(1, -1)[0],
                                index=final_corr.index)
        corr_series = corr_series.dropna()

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

        segment.track('ThresholdFilter')

    def plot(self) -> list:
        """
        Function to plot buy and sell events.

        :return: (list) List of Axes objects.
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

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
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

        :param lookback: (int) Lookback period to use.
        """

        self.lookback = lookback
        self.sigma = None
        self.frame = None
        self.mu_avg = None
        self.spread = None
        self.vol_series = None
        self.rolling_mean_vol = None
        self.vol_forcast_series = None

        segment.track('VolatilityFilter')

    def plot(self) -> list:
        """
        Function to plot spread series, regime states, and forecasted volatility.

        :return: (list) List of Axes objects.
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

        :param spread: (pd.DataFrame) Time series to be analyzed.
        """

        self.spread = spread
        self.frame = spread.diff().copy().to_frame().dropna()

        # Initialize an Exponentially Weighted Moving Average Variance Object,
        # also known as RiskMetrics.
        risk_metrics_vol = EWMAVariance(0.94)

        # Initialize a ZeroMean object to let us use the residuals estimated from the
        # volatility model separately.
        zero_mean = ZeroMean(self.frame, volatility=risk_metrics_vol)
        zero_mean.fit()

        # Store the residuals as a DateIndex Series.
        vol_forcast_series = pd.Series(zero_mean.resids(self.frame), index=self.frame.index)

        # Calculate rolling estimated mean of estimated volatility.
        rolling_mean_vol = vol_forcast_series.rolling(window=self.lookback).mean()

        # Calculate the mean of the rolling mean volatility estimate.
        mu_avg = rolling_mean_vol.rolling(window=self.lookback).mean().mean()
        # Calculate the mean of the rolling std dev of the volatility estimate.
        sigma = rolling_mean_vol.rolling(window=self.lookback).std().mean()

        self.vol_forcast_series = vol_forcast_series
        self.rolling_mean_vol = rolling_mean_vol
        self.mu_avg = mu_avg
        self.sigma = sigma

        return self

    def transform(self) -> pd.DataFrame:
        """
        Adds a regime column to describe the volatility level that was detected at
        that point, based on the parameters given in the constructor. And also
        a 'leverage_multiplier' column describing the leverage factor use
        in Dunis et al. (2005).

        :return: (pd.DataFrame) Time series augmented with the regime
            information.
        """

        vol_series = self.vol_forcast_series.copy().to_frame()
        vol_series["regime"] = np.nan
        vol_series["leverage_multiplier"] = np.nan

        avg_minus_two_std_dev = self.mu_avg - 2*self.sigma
        avg_minus_four_std_dev = self.mu_avg - 4*self.sigma

        # Extremely Low Regime - smaller or equal than mean average volatility - 4 std devs
        extrm_low_mask = (self.rolling_mean_vol <= avg_minus_four_std_dev)
        vol_series.loc[extrm_low_mask, "regime"] = -3
        vol_series.loc[extrm_low_mask, "leverage_multiplier"] = 2.5

        # Medium Low Regime - smaller or equal than mean average volatility - 2 std devs and greater or equal
        # than mean average volatility - 4 std devs
        medium_low_mask = (self.rolling_mean_vol <= avg_minus_two_std_dev) & (self.rolling_mean_vol >= avg_minus_four_std_dev)
        vol_series.loc[medium_low_mask, "regime"] = -2
        vol_series.loc[medium_low_mask, "leverage_multiplier"] = 2

        # Higher Low Regime - smaller or equal than mean average volatility and greater or equal
        # than mean average volatility - 2 std devs
        higher_low_mask = (self.rolling_mean_vol <= self.mu_avg) & (self.rolling_mean_vol >= avg_minus_two_std_dev)
        vol_series.loc[higher_low_mask, 'regime'] = -1
        vol_series.loc[higher_low_mask, "leverage_multiplier"] = 1.5

        avg_plus_two_std_dev = self.mu_avg + 2*self.sigma
        avg_plus_four_std_dev = self.mu_avg + 4*self.sigma

        # Lower High Regime - greater or equal than mean average volatility + 2 std devs and smaller or equal
        # than mean average volatility + 4 std devs
        low_high_mask = (self.rolling_mean_vol >= self.mu_avg) & (self.rolling_mean_vol <= avg_plus_two_std_dev)
        vol_series.loc[low_high_mask, 'regime'] = 1
        vol_series.loc[low_high_mask, "leverage_multiplier"] = 1

        # Medium High Regime - greater or equal than mean average volatility + 2 std devs and smaller or equal
        # than mean average volatility + 4 std devs
        medium_high_mask = (self.rolling_mean_vol >= avg_plus_two_std_dev) & (self.rolling_mean_vol <= avg_plus_four_std_dev)
        vol_series.loc[medium_high_mask, "regime"] = 2
        vol_series.loc[medium_high_mask, "leverage_multiplier"] = 0.5

        # Extremely High Regime - greater or equal than mean average volatility + 4 std devs
        extrm_high_mask = (self.rolling_mean_vol >= avg_plus_four_std_dev)
        vol_series.loc[extrm_high_mask, "regime"] = 3
        vol_series.loc[extrm_high_mask, "leverage_multiplier"] = 0

        self.vol_series = vol_series

        return vol_series

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method that executes the fit and transform method
        in sequence.

        :param frame: (pd.DataFrame) Time series to be analyzed.
        :return: (pd.DataFrame) Time series augmented with the trade side
            information.
        """

        return self.fit(frame).transform()
