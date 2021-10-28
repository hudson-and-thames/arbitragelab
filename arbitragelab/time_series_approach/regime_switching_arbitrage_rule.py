# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
The module implements a statistical arbitrage strategy based on the Markov regime-switching model.
"""
# pylint: disable=invalid-name

from typing import Union, Callable
import warnings

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from arbitragelab.util import segment


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

        # Default strategy as described in the paper
        self.strategy = {
            "High": {
                "Long": {"Open": lambda Xt, mu, delta, sigma, prob, rho: Xt <= mu - delta * sigma and prob >= rho,
                         "Close": lambda Xt, mu, delta, sigma, prob, rho: Xt >= mu + delta * sigma},
                "Short": {"Open": lambda Xt, mu, delta, sigma, prob, rho: Xt >= mu + delta * sigma,
                          "Close": lambda Xt, mu, delta, sigma, prob, rho: Xt <= mu - delta * sigma and prob >= rho}
            },
            "Low": {
                "Long": {"Open": lambda Xt, mu, delta, sigma, prob, rho: Xt <= mu - delta * sigma,
                         "Close": lambda Xt, mu, delta, sigma, prob, rho: Xt >= mu + delta * sigma and prob >= rho},
                "Short": {"Open": lambda Xt, mu, delta, sigma, prob, rho: Xt >= mu + delta * sigma and prob >= rho,
                          "Close": lambda Xt, mu, delta, sigma, prob, rho: Xt <= mu - delta * sigma}
            }
        }

        # Mapping the signal to the actual trade based on the position at the previous timestamp
        self.long_entry_mapping = {
            1: {True: False, False: False},

            -1: {True: True, False: False},

            0: {True: True, False: False}
        }

        self.long_exit_mapping = {
            1: {True: True, False: False},

            -1: {True: False, False: False},

            0: {True: False, False: False}
        }

        self.short_entry_mapping = {
            1: {True: True, False: False},

            -1: {True: False, False: False},

            0: {True: True, False: False}
        }

        self.short_exit_mapping = {
            1: {True: False, False: False},

            -1: {True: True, False: False},

            0: {True: False, False: False}
        }

        segment.track('RegimeSwitchingArbitrageRule')

    def change_strategy(self, regime: str, direction: str, action: str, rule: Callable):
        """
        This function is used for changing the default strategy.

        :param regime: (str) Rule's regime. It could be either "High" or "Low".
        :param direction: (str) Rule's direction. It could be either "Long" or "Short".
        :param action: (str) Rule's action. It could be either "Open" or "Close".
        :param rule: (Callable) A new rule to replace the original rule. The parameters of the rule
            should be the subset of (Xt, mu, delta, sigma, prob, rho).
        """

        params = set(rule.__code__.co_varnames)
        if params.issubset({"Xt", "mu", "delta", "sigma", "prob", "rho"}):
            self.strategy[regime][direction][action] = rule

        else:
            raise Exception("Incorrect parameters of the rule. "
                            "Please use the subset of (Xt, mu, delta, sigma, prob, rho) as the parameters.")

    def get_signal(self, data: Union[np.array, pd.Series, pd.DataFrame], switching_variance: bool = False,
                   silence_warnings: bool = False) -> np.array:
        """
        The function will first fit the Markov regime-switching model to all the input data,
        then calculate the trading signal at the last timestamp based on the strategy.

        :param data: (np.array/pd.Series/pd.DataFrame) A time series for fitting the Markov regime-switching model.
            The dimensions should be n x 1.
        :param switching_variance: (bool) Whether the Markov regime-switching model has different variances in different regimes.
        :param silence_warnings: (bool) Flag to silence warnings from failing to fit
            the Markov regime-switching model to the input data properly.
        :return: (np.array) The trading signal at the last timestamp of the given data. The returned array will contain four Boolean values,
            representing whether to open a long trade, close a long trade, open a short trade and close a short trade.
        """

        data_npa = self._check_type(data)  # Checking the type of the input data

        # Handling warning if the module fails to fit the Markov regime-switching model to the input data
        with warnings.catch_warnings():
            if silence_warnings:  # Silencing warnings if the user doesn't want them to show out
                warnings.filterwarnings('ignore')

            # Fitting the Markov regime-switching model
            mod = sm.tsa.MarkovRegression(data_npa, k_regimes=2, switching_variance=switching_variance)
            res = mod.fit()

            if np.isnan(res.params).sum() == len(res.params):
                warnings.warn("Failed to fit the Markov regime-switching model to the input data.")
                return np.full(4, False)

        # Unpacking parameters
        mu = res.params[2:4]

        if switching_variance:
            sigma = res.params[4:6]
        else:
            sigma = (res.params[4], res.params[4])

        # Determining regime index
        low_regime = np.argmin(mu)

        current_regime = np.argmax(res.smoothed_marginal_probabilities[-1])
        smoothed_prob = res.smoothed_marginal_probabilities[-1][current_regime]

        # Mapping strategy parameters
        param = {
            "Xt": data_npa[-1], "mu": mu[current_regime], "delta": self.delta,
            "sigma": sigma[current_regime], "prob": smoothed_prob, "rho": self.rho
        }

        # Calculating the signal based on the strategy
        regime = "Low" if current_regime == low_regime else "High"
        signal = np.full(4, False)
        count = 0

        for i in self.strategy[regime].items():
            for j in i[1].items():
                rule = j[1]
                param_needed = rule.__code__.co_varnames
                signal[count] = rule(*[i[1] for i in param.items() if i[0] in param_needed])
                count += 1

        return signal

    def get_signals(self, data: Union[np.array, pd.Series, pd.DataFrame], window_size: int,
                    switching_variance: bool = False, silence_warnings: bool = False) -> np.array:
        """
        The function will fit the Markov regime-switching model with a rolling time window,
        then calculate the trading signal at the last timestamp of the window based on the strategy.

        :param data: (np.array/pd.Series/pd.DataFrame) A time series for fitting the Markov regime-switching model.
            The dimensions should be n x 1.
        :param window_size: (int) Size of the rolling time window.
        :param switching_variance: (bool) Whether the Markov regime-switching model has different variances in different regimes.
        :param silence_warnings: (bool) Flag to silence warnings from failing to fit
            the Markov regime-switching model to the input data.
        :return: (np.array) The array containing the trading signals at each timestamp of the given data.
            A trading signal at any timestamp will have four Boolean values, representing whether to open a long trade,
            close a long trade, open a short trade and close a short trade. The returned array dimensions will be n x 4.
        """

        signals = np.full((len(data), 4), False)

        for i in range(window_size, len(signals) + 1):
            signals[i - 1] = self.get_signal(data[i - window_size:i], switching_variance, silence_warnings)

        return signals

    def get_trades(self, signals: np.array) -> np.array:
        """
        The function will decide the trade actions at each timestamp based on the signal at time t
        and the position at time t - 1. The position at time 0 is assumed to be 0.

        :param signals: (np.array) The array containing the trading signals at each timestamp. A trading signal
            at any timestamp should have four Boolean values, representing whether to open a long trade,
            close a long trade, open a short trade and close a short trade. The input array dimensions should be n x 4.
        :return: (np.array) The array containing the trade actions at each timestamp. A trade action at any timestamp
            will have four Boolean values, representing whether to open a long trade, close a long trade,
            open a short trade and close a short trade. The returned array dimensions will be n x 4.
        """

        long_entry = signals[:, 0].copy()
        long_exit = signals[:, 1].copy()
        short_entry = signals[:, 2].copy()
        short_exit = signals[:, 3].copy()

        positions = np.zeros(len(signals))

        for i in range(1, len(positions)):
            long_entry[i] = self.long_entry_mapping[positions[i - 1]][long_entry[i]]
            long_exit[i] = self.long_exit_mapping[positions[i - 1]][long_exit[i]]
            short_entry[i] = self.short_entry_mapping[positions[i - 1]][short_entry[i]]
            short_exit[i] = self.short_exit_mapping[positions[i - 1]][short_exit[i]]

            if positions[i - 1] == -1 and long_entry[i]:
                short_exit[i] = True

            if positions[i - 1] == 1 and short_entry[i]:
                long_exit[i] = True

            if long_entry[i]:
                positions[i] = 1

            elif short_entry[i]:
                positions[i] = -1

            else:
                if not (long_exit[i] or short_exit[i]):
                    positions[i] = positions[i - 1]
                else:
                    positions[i] = 0

        return np.column_stack((long_entry, long_exit, short_entry, short_exit))

    @staticmethod
    def plot_trades(data: Union[np.array, pd.Series, pd.DataFrame], trades: np.array,
                    marker_size: int = 12) -> plt.figure:
        """
        Plots the trades on the given data.

        :param data: (np.array/pd.Series/pd.DataFrame) The time series to plot the trades on.
            The dimensions should be n x 1.
        :param trades: (np.array) The array containing the trade actions at each timestamp of the given data.
            A trade action at any timestamp should have four Boolean values, representing whether to open a long trade,
            close a long trade, open a short trade and close a short trade. The input array dimensions will be n x 4.
        :param marker_size: (int) Marker size of the plot.
        :return: (plt.figure) Figure that plots trades on the given data.
        """

        long_entry = trades[:, 0]
        long_exit = trades[:, 1]
        short_entry = trades[:, 2]
        short_exit = trades[:, 3]

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
    def _check_type(data: Union[np.array, pd.Series, pd.DataFrame]) -> np.array:
        """
        Checks the type of the input data and returns it in the type of np.array.

        :param data: (np.array/pd.Series/pd.DataFrame) A time series with dimensions n x 1.
        :return: (np.array) Input data in the type of np.array.
        """

        # Checking the data type
        if isinstance(data, np.ndarray):
            return data

        return data.to_numpy()
