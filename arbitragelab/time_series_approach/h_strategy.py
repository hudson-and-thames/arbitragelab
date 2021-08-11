# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the strategy described in
`Bogomolov, T. (2013). Pairs trading based on statistical variability of the spread process. Quantitative Finance, 13(9): 1411–1430.
<https://www.researchgate.net/publication/263339291_Pairs_trading_based_on_statistical_variability_of_the_spread_process>`_
"""
# pylint: disable=invalid-name, unused-variable

from itertools import compress, combinations

import pandas as pd
import numpy as np
from tqdm import tqdm

from arbitragelab.util import devadarsh

class HConstruction:
    """
    This class implements a statistical arbitrage strategy described in the following publication:
    `Bogomolov, T. (2013). Pairs trading based on statistical variability of the spread process.
    Quantitative Finance, 13(9): 1411–1430.
    <https://www.researchgate.net/publication/263339291_Pairs_trading_based_on_statistical_variability_of_the_spread_process>`_
    """

    def __init__(self, series: pd.Series, threshold: float, method: str = "Kagi"):
        """
        Initializes the module parameters.

        :param series: (pd.Series) A time series for building the H-construction.
            The dimensions should be n x 1.
        :param threshold: (float) The threshold of the H-construction.
        :param method: (str) The method used to build the H-construction. The options are ["Kagi", "Renko"].
        """

        self.series = series
        self.threshold = threshold
        self.method = method

        self.tao_a_index = None
        self.tao_b_index = None
        self.tao_a_direction = None
        self.tao_b_direction = None
        self.signals_contrarian = None
        self.signals_momentum = None

        # Dictionary for storing the results of the H-construction
        self.results = {
            "h_series": [self.series[0]],
            "direction": [0],
            "index": [0],
            "tao_a": [False],
            "tao_b": [False]
        }

        # Building the H-construction
        self._construct()

        devadarsh.track('HConstruction')

    def _process(self, i: int):
        """
        Processes H-construction on the i-th element of the series.

        :param i: (int) The index of the element.
        """

        # Calculating the gap between the current price and the latest price of the H-construction.
        gap = (self.series[i] - self.results["h_series"][-1])
        direction = np.sign(gap)

        if self.method == "Kagi":
            over_thresholds = abs(gap) >= self.threshold
            same_direction = (self.results["direction"][-1] == direction)

            # Do nothing if the direction is turned, but the amount is not enough
            if not same_direction and not over_thresholds:
                return

            # reverse == True if the direction is turned with enough amount
            reverse = (not same_direction) and over_thresholds
            self._append(i, direction, reverse)

        elif self.method == "Renko":
            # Calculating the number of bricks needed
            num_bricks = abs(gap) // self.threshold

            # Appending each bricks to the results
            for brick in range(int(num_bricks)):
                reverse = (self.results["direction"][-1] != direction)
                self._append(i, direction, reverse)

        else:
            raise Exception("Incorrect method. "
                            "Please use one of the options "
                            "[\"Kagi\", \"Renko\"].")

    def _append(self, i: int, direction: int, reverse: bool):
        """
        Appends the result of the H-construction.

        :param i: (int) The index of the element.
        :param direction: (int) The direction of the element.
        :param reverse: (bool) Whether this element is a reverse point.
        """

        # Determining the price to append based on the method
        if self.method == "Kagi":
            price = self.series[i]

        else:
            price = self.results["h_series"][-1] + direction * self.threshold

        # Marking turning point if reverse == True
        # tao_a indicates whether it is a turning point, and tao_b indicates whether it is a confirmation point for turning point
        if reverse:
            self.results["tao_a"][-1] = True
            self.results["tao_a"].append(False)
            self.results["tao_b"].append(True)

        else:
            self.results["tao_a"].append(False)
            self.results["tao_b"].append(False)

        # Appending other informations
        self.results["h_series"].append(price)
        self.results["direction"].append(direction)
        self.results["index"].append(i)

    def _construct(self):
        """
        Builds the H-construction on each element of the series.
        """

        # Processing each element one by one
        for i in range(len(self.results["h_series"]), len(self.series)):
            self._process(i)

        index = self.results["index"]
        direction = self.results["direction"]
        tao_a = self.results["tao_a"]
        tao_b = self.results["tao_b"]

        # Determining the index values and the directions of turning points and confirmation points
        self.tao_a_index = list(compress(index, tao_a))
        self.tao_b_index = list(compress(index, tao_b))
        self.tao_a_direction = list(compress(direction, tao_a))
        self.tao_b_direction = list(compress(direction, tao_b))

        # Determining the signals
        self.signals_contrarian = pd.Series(0, index = self.series.index)
        self.signals_momentum = pd.Series(0, index = self.series.index)

        # The signals will be opposite to the directions of the turning confirmation points
        self.signals_contrarian[self.tao_b_index] = [-d for d in self.tao_b_direction]

        # The signals will be same to the directions of the turning confirmation points
        self.signals_momentum[self.tao_b_index] = self.tao_b_direction

    def h_inversion(self) -> int:
        """
        Calculates H-inversion statistic, which counts the number of times the series changes its direction for the selected threshold.

        :return: (int) The value of the H-inversion.
        """

        # The number of times the series changes its direction will equal to the number of the confirmation points
        return len(self.tao_b_index)

    def h_distances(self, p: int = 1) -> float:
        """
        Calculates the sum of vertical distances between local maximums and minimums to the power p.

        :param p: (int) The number of powers when calculating the distance.
        :return: (float) The sum of vertical distances between local maximums and minimums.
        """

        summation = 0
        for i in range(1, len(self.tao_a_index)):
            diff = self.series[self.tao_a_index[i]] - self.series[self.tao_a_index[i - 1]]
            summation += abs(diff) ** p

        return summation

    def h_volatility(self, p: int = 1) -> float:
        """
        Calculates H-volatility statistic of order p, which is a measure of the variability of the series for the selected threshold.

        :param p: (int) The order of H-volatility.
        :return: (float) The value of the H-volatility.
        """

        return self.h_distances(p)/self.h_inversion()

    def extend_series(self, series: pd.Series):
        """
        Extends the original series used as input during initialization and and rebuilds the H-construction on the extended series

        :param series: (pd.Series) A time series for extending the original series used as input during initialization.
            The dimensions should be n x 1.
        """

        self.series = pd.concat([self.series, series])
        self._construct()

    def get_signals(self, method: str = "contrarian") -> pd.Series:
        """
        Gets the signals at each timestamp based on the method described in the paper.

        :param method: (str) The method used to determine the signals. The options are ["contrarian", "momentum"].
        :return: (pd.Series) The time series contains the signals at each timestamp.
        """

        if method == "contrarian":
            signals =  self.signals_contrarian

        elif method == "momentum":
            signals = self.signals_momentum

        else:
            raise Exception("Incorrect method. "
                            "Please use one of the options "
                            "[\"contrarian\", \"momentum\"].")

        return signals

class HSelection:
    """
    This class implements a pairs selection strategy described in the following publication:
    `Bogomolov, T. (2013). Pairs trading based on statistical variability of the spread process.
    Quantitative Finance, 13(9): 1411–1430.
    <https://www.researchgate.net/publication/263339291_Pairs_trading_based_on_statistical_variability_of_the_spread_process>`_
    """

    def __init__(self, data: pd.DataFrame, method: str = "Kagi"):
        """
        Initializes the module parameters.

        :param data: (pd.DataFrame) Price data with columns containing asset prices.
            The dimensions should be n x m, where n denotes the length of the data and m denotes the number of assets.
        :param method: (str) The method used to build the H-construction for each possible pair of assets.
            The options are ["Kagi", "Renko"].
        """

        self.data = data
        self.method = method
        self.length = len(data)
        self.minimum_length = self.length
        self.results = None

        self.stock_pool = list(data.columns)
        self.possible_pairs = list(combinations(self.stock_pool, 2))

    def _get_h_inversion(self, pair: tuple) -> tuple:
        """
        Calculates H-inversion statistic for the spread series formed by the specified pair,
            which counts the number of times the series changes its direction for the selected threshold.

        :param pair: (tuple) The tuple contains the column names of two selected assets.
        :return: (tuple) The tuple contains the value of the H-inversion and the threshold of the H-construction.
        """

        data_needed = self.data[list(pair)].dropna(axis = 0)

        # Return (0, 0) if the data length after removing rows with NaN is less than the minimum required length
        # or if there is any negative value.
        if len(data_needed) < self.minimum_length or (data_needed.values < 0).any():
            return 0, 0

        # Forming the spread series
        # Calculated as: log(Pt) - log(Qt)
        series = np.log(data_needed[pair[0]]) - np.log(data_needed[pair[1]])

        # Use the standard deviation of the spread series as the threshold of the H-construction
        std = series.std()
        hc = HConstruction(series, std, self.method)

        return hc.h_inversion(), std

    def select(self, minimum_length:int = None):
        """
        Calculates H-inversion statistic for the spread series formed by each possible pair, and stores the results.

        :param minimum_length: (int) Minimum length of consistent index required for the selected pair to do H-construction.
        """

        if minimum_length is not None:
            self.minimum_length = minimum_length
        else:
            self.minimum_length = self.length

        results = []
        for pair in tqdm(self.possible_pairs):
            h_inversion, std = self._get_h_inversion(pair)
            if h_inversion != 0 and std != 0:
                results.append([h_inversion, std, pair])

        # Sorting the results by H-inversion statistic
        self.results = sorted(results, key=lambda i:i[0], reverse=True)

    def get_pairs(self, num: int, method: str = "highest", allow_repeat: bool = False) -> list:
        """
        Gets top N pairs with the highest/lowest H-inversion.

        :param num: (int) The number of pairs that the user wants to get.
        :param method: (str) The method used to select pairs. The options are ["highest", "lowest"].
        :param allow_repeat: (bool) Whether the user allows the same asset to appear repeatedly in different pairs.
        :return: (list) The list contains the informations of the top N pairs. Each element in the list will contains three things:
            [H-inversion statistic, Threshold of the H-construction, Tuple contains the column names of two selected assets].
        """

        if allow_repeat:
            if method == "highest":
                chose_pairs =  self.results[:num]

            elif method == "lowest":
                chose_pairs = self.results[-num:]

            else:
                raise Exception("Incorrect method. "
                                "Please use one of the options "
                                "[\"highest\", \"lowest\"].")

        else:
            chose_tickers = []
            chose_pairs = []

            if method == "highest":
                results =  self.results

            elif method == "lowest":
                results =  self.results.copy()
                results.reverse()

            else:
                raise Exception("Incorrect method. "
                                "Please use one of the options "
                                "[\"highest\", \"lowest\"].")

            for i in results:
                tickers = i[1]
                if tickers[0] not in chose_tickers and tickers[1] not in chose_tickers:
                    chose_tickers.extend(tickers)
                    chose_pairs.append(i)

                if len(chose_pairs) == num:
                    break

        return chose_pairs
