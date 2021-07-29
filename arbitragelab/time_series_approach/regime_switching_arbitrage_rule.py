# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
The module implements a statistical arbitrage strategy based on the Markov regime-switching model.
"""

# pylint: disable=missing-module-docstring, invalid-name
from typing import Union
import warnings

import pandas as pd
import numpy  as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from arbitragelab.util import devadarsh


class RegimeSwitchingArbitrageRule:
    """
    This class implements a statistical arbitrage strategy described in the following publication:
    `Bock, M. and Mestel, R. (2009). A regime-switching relative value arbitrage rule.
    Operations Research Proceedings 2008, pages 9–14. Springer.
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.453.3576&rep=rep1&type=pdf>`_
    """

    def __init__(self, delta: float, rho: float):
        """
        Initializes the module parameters.

        :param delta: (float) The standard deviation sensitivity parameter of the trading strategy.
        :param rho: (float) The probability threshold of the trading strategy.
        """

        self.delta = delta
        self.rho = rho

        devadarsh.track('RegimeSwitchingArbitrageRule')

    def get_signal(self, data: Union[np.array, pd.Series, pd.DataFrame], switching_variance=False,
                     silence_warnings: bool = False) -> int:
        """
        The function will first fit the Markov regime-switching model to all the input data,
        then calculate the trading signal at the last timestamp based on the strategy described in the paper.

        :param data: (np.array/pd.Series/pd.DataFrame) A time series for fitting the Markov regime-switching model.
            The dimensions should be n x 1.
        :param silence_warnings: (bool) Flag to silence warnings from failing to fit
            the Markov regime-switching model to the input data properly.
        :return: (int) The trading signal at the last timestamp of the given data.
            [1 - Long trade, -1 - Short trade, 0 - No trade, NaN - Failed to fit the Markov regime-switching model].
        """

        data_npa = self._check_type(data) # Checking the type of the input data

        # Handling warning if the module fails to fit the Markov regime-switching model to the input data.
        with warnings.catch_warnings():
            if silence_warnings: # Silencing warnings if the user doesn't want them to show out.
                warnings.filterwarnings('ignore')

            # Fitting the Markov regime-switching model
            mod = sm.tsa.MarkovRegression(data_npa, k_regimes=2, switching_variance=switching_variance)
            res = mod.fit()
            params = res.params

            if np.isnan(params).sum() == len(params):
                warnings.warn("Failed to fit the Markov regime-switching model to the input data.")
                return np.nan

        # Unpacking parameters
        mu = res.params[2:4]

        if switching_variance:
            sigma = res.params[4:6]
        else:
            sigma = (res.params[4], res.params[4])

        # Determining regime index
        high_regime = np.argmax(mu)
        low_regime = np.argmin(mu)

        current_regime = np.argmax(res.smoothed_marginal_probabilities[-1])
        smoothed_prob = res.smoothed_marginal_probabilities[-1][current_regime]


        # Calculating the position based on the strategy described in the paper
        signal = None
        if current_regime == low_regime:
            if data_npa[-1] >= mu[low_regime] + self.delta * sigma[low_regime] and smoothed_prob >= self.rho:
                signal = -1
            elif data_npa[-1] <= mu[low_regime] - self.delta * sigma[low_regime]:
                signal = 1
            else:
                signal = 0

        else:
            if data_npa[-1] >= mu[high_regime] + self.delta * sigma[high_regime]:
                signal = -1
            elif data_npa[-1] <= mu[high_regime] - self.delta * sigma[high_regime] and smoothed_prob >= self.rho:
                signal = 1
            else:
                signal = 0

        return signal

    def get_signals(self, data: Union[np.array, pd.Series, pd.DataFrame], window_size: int,
                      switching_variance=False, silence_warnings: bool = False) -> np.array:
        """
        The function will fit the Markov regime-switching model with a rolling time window,
        then calculate the trading signal at the last timestamp of the window based on the strategy described in the paper.

        :param data: (np.array/pd.Series/pd.DataFrame) A time series for fitting the Markov regime-switching model.
            The dimensions should be n x 1.
        :param silence_warnings: (bool) Flag to silence warnings from failing to fit
            the Markov regime-switching model to the input data.
        :return: (np.array) Tne array contains all the trading signals at each timestamp of the given data.
            [1 - Long trade, -1 - Short trade, 0 - No trade, NaN - Failed to fit the Markov regime-switching model].
        """

        # Changing the type of the input data to pd.Series
        if isinstance(data, pd.DataFrame):
            data_ser = data.squeeze(axis=1)

        else:
            data_ser = pd.Series(data)

        signals = data_ser.rolling(window_size).apply(self.get_signal, args=(switching_variance, silence_warnings))
        signals = signals.ffill()
        signals = signals.fillna(0)

        return signals.values

    @staticmethod
    def plot_trades(data: Union[np.array, pd.Series, pd.DataFrame], signals: Union[np.array, pd.Series, pd.DataFrame],
                    marker_size: int = 12) -> plt.figure:
        """
        Plots the trades on the given data based on the positions at each timestamp.

        :param data: (np.array/pd.Series/pd.DataFrame) The time series to plot the trades on.
            The dimensions should be n x 1.
        :param signals: (np.array/pd.Series/pd.DataFrame) A time series contains all the trading signals
            at each timestamp of the given data. The dimensions should be n x 1.
        :param marker_size: Marker size of the plot.
        :return: (plt.figure) Figure that plots trades on the given data.
        """

        long_entry = np.full(len(signals), False)
        long_exit = np.full(len(signals), False)
        short_entry = np.full(len(signals), False)
        short_exit = np.full(len(signals), False)
        positions = np.zeros(len(signals))

        mapping = {
            1 : {1 : (False, False, False, False), -1 : (False, True, True, False), 0 : (False, False, False, False)},

            -1 : {1 : (True, False, False ,True), -1 : (False, False, False, False), 0 : (False, False, False, False)},

            0 : {1 : (True, False, False ,False), -1 : (False, False, True, False), 0 : (False, False, False, False)}
        }

        for i in range(1, len(signals)):
            long_entry[i], long_exit[i], short_entry[i], short_exit[i] = mapping[positions[i - 1]][signals[i]]

            if long_entry[i]:
                positions[i] = 1

            elif short_entry[i]:
                positions[i] = -1

            else:
                positions[i] = positions[i - 1]

        # Changing the type of the input data to pd.Series
        if isinstance(data, pd.DataFrame):
            data_ser = data.squeeze(axis=1)

        else:
            data_ser = pd.Series(data)

        # Plotting trades
        fig = plt.figure()
        plt.plot(data_ser)
        plt.plot(data_ser[long_entry], 'go', markersize=marker_size, label='Open Long Position')
        plt.plot(data_ser[long_exit], 'gx', markersize=marker_size, label='Clear Long Position')
        plt.plot(data_ser[short_entry], 'ro', markersize=marker_size, label='Open Short Position')
        plt.plot(data_ser[short_exit], 'rx', markersize=marker_size, label='Clear Short Position')
        plt.title("Trades")
        plt.legend()

        return fig

    @staticmethod
    def _check_type(data) -> np.array:
        """
        Checks the type of the input data and returns it in the type of np.array.

        :param data: (np.array/pd.Series/pd.DataFrame) A time series with dimensions n x 1.
        :return: (np.array) Input data in the type of np.array.
        """

        # Checking the data type
        if isinstance(data, np.ndarray):
            return data

        return data.to_numpy()
