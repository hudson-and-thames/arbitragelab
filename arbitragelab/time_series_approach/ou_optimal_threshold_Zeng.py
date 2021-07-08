# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=missing-module-docstring, invalid-name
import warnings
import numpy as np
import pandas as pd
from typing import Union, Callable
from scipy import optimize, special
from mpmath import nsum, inf, gamma, digamma, fac
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from arbitragelab.time_series_approach.ou_optimal_threshold import OUModelOptimalThreshold
from arbitragelab.util import devadarsh


class OUModelOptimalThresholdZeng(OUModelOptimalThreshold):
    """
    This class implements the analytic solutions of the optimal trading thresholds for the series
    with mean-reverting properties. The methods are described in the following publication:
    Zeng, Z. and Lee, C.-G. (2014).  Pairs trading: 
    optimal thresholds and profitability.QuantitativeFinance, 14(11):1881–1893.
    Link: https://www.tandfonline.com/doi/pdf/10.1080/14697688.2014.917806

    Assumptions of the method:
    1. The series Xt = ln(Pt) follows a Ornstein-Uhlenbeck process, where Pt is a price series of a asset or a spread.
    2. A Trading strategy is defined by entering a trade when Yt = a or -a, exiting the trade at Yt = b or -b,
       where Yt is a dimensionless series transformed from the original time series Xt.
    3. A trading cycle is defined as the time needed for Yt to change from -a to a, then to b.
    4. 0 <= b <= a or -a <= b <= 0
    """

    def __init__(self):
        """
        Initializes the module parameters.
        """

        super().__init__()

        devadarsh.track('OUModelOptimalThresholdZeng')

    def expected_return(self, a: float, b: float, c: float):
        """
        Calculates equation (11) to get the expected return given trading thresholds.

        :param a: (float) The entry threshold of the trading strategy
        :param b: (float) The exit threshold of the trading strategy
        :param c: (float) The transaction costs of the trading strategy
        :return: (float) The expected return of the strategy.
        """

        return

    def variance(self, a: float, m: float, c: float):
        """
        Calculates equation (12) to get the variance given trading thresholds.

        :param a: (float) The entry threshold of the trading strategy
        :param b: (float) The exit threshold of the trading strategy
        :param c: (float) The transaction costs of the trading strategy
        :return: (float) The variance of the strategy.
        """

        return

    def get_threshold_by_maximize_expected_return_convention(self, c: float):
        """
        Solves equation (20) in the paper to get the optimal trading thresholds.

        :param c: (float) The transaction costs of the trading strategy
        :return: (tuple) The value of the optimal trading thresholds
        """

        c_trans = self._transform_to_dimensionless(c)
        args = (c_trans, np.vectorize(self._equation_term))
        initial_guess = c_trans
        root = optimize.fsolve(self._equation_20, initial_guess, args=args)[0]

        print(root)
        return self._back_transform_from_dimensionless(root), self._back_transform_from_dimensionless(0)

    def get_threshold_by_maximize_expected_return_new(self, c: float):
        """
        Solves equation (23) in the paper to get the optimal trading thresholds.

        :param c: (float) The transaction costs of the trading strategy
        :return: (tuple) The value of the optimal trading thresholds
        """

        c_trans = self._transform_to_dimensionless(c)
        args = (c_trans, np.vectorize(self._equation_term))
        initial_guess = c_trans
        root = optimize.fsolve(self._equation_23, initial_guess, args=args)[0]

        return self._back_transform_from_dimensionless(root), self._back_transform_from_dimensionless(-root)

    def _transform_to_dimensionless(self, const: float):
        """
        Transforms input value to dimensionless system.

        :param const: Value to transform
        :return: (float) Value in dimensionless system
        """

        return (const - self.theta) * np.sqrt((2 * self.mu)) / self.sigma

    def _back_transform_from_dimensionless(self, const: float):
        """
        Back transforms input value from dimensionless system.

        :param const: Value in dimensionless system.
        :return: (float) Original Value
        """

        return const / np.sqrt((2 * self.mu)) * self.sigma + self.theta

    @staticmethod
    def _equation_term(const: float, index: int):
        """
        A helper function for simplifing equation expression

        :param const: (float) The input value of the function
        :param index: (int) It could be 0 or 1.
        :return: (float) The output value of the function
        """

        middle_term = lambda k: gamma((2 * k + 1) / 2) * ((1.414 * const) ** (2 * k + index)) / fac(2 * k + index)
        term = nsum(middle_term, [0, inf])

        return float(term)

    @staticmethod
    def _equation_20(a: float, *args: tuple):
        """
        Equation (20) in the paper.

        :param a: (float) The entry threshold of the trading strategy
        :param args: (tuple) Other parameters needed for the equation
        :return: (float) The value of the equation
        """

        c, equation_term = args
        return (1 / 2) * equation_term(a, 1) - (a - c) * (1.414 / 2) * equation_term(a, 0)

    @staticmethod
    def _equation_23(a: float, *args: tuple):
        """
        Equation (23) in the paper.

        :param a: (float) The entry threshold of the trading strategy
        :param args: (tuple) Other parameters needed for the equation
        :return: (float) The value of the equation
        """

        c, equation_term = args
        return (1 / 2) * equation_term(a, 1) - (a - c / 2) * (1.414 / 2) * equation_term(a, 0)




