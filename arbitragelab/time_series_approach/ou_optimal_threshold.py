# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=missing-module-docstring, invalid-name
import warnings
import numpy as np
import pandas as pd
from typing import Union, Callable
from scipy import optimize, special
from sympy import symbols, gamma, sqrt, factorial, Sum, oo, digamma
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from arbitragelab.util import devadarsh


class OUModelOptimalThresholdBertram:
    """
    This class implements the analytic solutions of the optimal trading thresholds for the series
    with mean-reverting properties. The methods are described in the following publication:
    Bertram, W. K. (2010). Analytic solutions for optimal statistical arbitrage trading.
    Physica A: Statistical Mechanics and its Applications, 389(11):2234–2243.

    Assumptions of the method:
    1. The series Xt = ln(Pt) follows a Ornstein-Uhlenbeck process, where Pt is a price series of a asset or a spread.
    2. A Trading strategy is defined by entering a trade when Xt = a, exiting the trade at Xt = m,
       and waiting until the process returns to Xt = a, to complete the trading cycle.
    3. a < m
    """

    def __init__(self):
        """
        Initializes the module parameters.
        """

        self.theta = None # The long-term mean of the O-U process
        self.mu = None # The speed at which the values will regroup around the long-term mean
        self.sigma = None # The amplitude of randomness of the O-U process

        devadarsh.track('OUModelOptimalThresholdBertram')

    # Users can choose two different ways to initialize the parameters of the O-U process.
    def construct_ou_model_from_given_parameters(self, theta: Union[float, int], mu: Union[float, int], sigma: Union[float, int]):
        """
        Initializes the O-U process from given parameters.

        :param theta: (float/int) The long-term mean of the O-U process
        :param mu: (float/int) The speed at which the values will regroup around the long-term mean
        :param sigma: (float/int) The amplitude of randomness of the O-U process
        """

        self.theta = theta
        self.mu = mu
        self.sigma = sigma


    def fit_ou_model_to_data(self, series_1, series_2, beta):
        """
        Fits the OU model to given data.
        """

    def expected_return(self, a: Union[float, int], m: Union[float, int], c: Union[float, int]):
        """
        Calculates equation (11) to get the expected return given trading thresholds.

        :param a: (float/int) The entry threshold of the trading strategy
        :param m: (float/int) The exit threshold of the trading strategy
        :param c: (float/int) The transaction costs of the trading strategy
        :return: (float) The expected return of the strategy.
        """

        return self.mu * (m - a - c) / (np.pi * (self._erfi_scaler(m) - self._erfi_scaler(a)))

    def variance(self, a: Union[float, int], m: Union[float, int], c: Union[float, int]):
        """
        Calculates equation (12) to get the variance given trading thresholds.

        :param a: (float/int) The entry threshold of the trading strategy
        :param m: (float/int) The exit threshold of the trading strategy
        :param c: (float/int) The transaction costs of the trading strategy
        :return: (float) The variance of the strategy.
        """

        const_1 = (m - self.theta) * np.sqrt(2 * self.mu) / self.sigma
        const_2 = (a - self.theta) * np.sqrt(2 * self.mu) / self.sigma

        term_1 = self.mu * ((m - a - c) ** 2)
        term_2 = self._w1(const_1) - self._w1(const_2) - self._w2(const_1) + self._w2(const_2)
        term_3 = (np.pi * (self._erfi_scaler(m) - self._erfi_scaler(a))) ** 3

        return term_1 * term_2 / term_3

    def get_threshold_by_maximize_expected_return(self, c: Union[float, int]):
        """
        Solves equation (13) in the paper to get the optimal trading thresholds.

        :param c: (float/int) The transaction costs of the trading strategy
        :return: (tuple) The value of the optimal trading thresholds
        """

        args = (c, self.theta, self.mu, self.sigma, self._erfi_scaler)
        initial_guess = self.theta - 2*self.sigma
        root = optimize.fsolve(self._equation_13, initial_guess, args = args)[0]

        return root, 2*self.mu - root

    def get_threshold_by_maximize_sharpe_ratio(self, c: Union[float, int], rf: Union[float, int]):
        """
        Minimize -1 * equation (16) in the paper to get the optimal trading thresholds.

        :param c: (float/int) The transaction costs of the trading strategy
        :param rf: (float/int) The risk free rate
        :return: (tuple) The value of the optimal trading thresholds
        """

        args = (c, rf, self.theta, self.mu, self.sigma, self._erfi_scaler, self._w1, self._w2)
        initial_guess = self.theta - 2 * self.sigma
        sol = optimize.minimize(self._negative_equation_16, initial_guess, args = args)

        return sol, 2*self.mu - sol

    def _erfi_scaler(self, const):
        return special.erfi((const - self.theta) * np.sqrt(self.mu) / self.sigma)

    def _w1(self, const):
        z = symbols('z')
        k = symbols('k')

        common_term = gamma(k / 2) * ((sqrt(2) * z) ** k) / factorial(k)
        term_1 = ((Sum(common_term, (k, 1, oo)) / 2) ** 2)
        term_2 = ((Sum(common_term * ((-1) ** k), (k, 1, oo)) / 2) ** 2)
        w1 = (term_1 - term_2).doit()

        w1 = lambdify(z, w1, 'numpy')
        return w1(const)

    def _w2(self, const):
        z = symbols('z')
        k = symbols('k')

        k_21 = 2 * k - 1
        phi = digamma(k_21 / 2) - digamma(1)
        middle_term = gamma(k_21 / 2) * phi * ((sqrt(2) * z) ** k_21) / factorial(k_21)
        w2 = Sum(middle_term, (k, 1, oo)).doit()

        w2 = lambdify(z, w2, 'numpy')
        return w2(const)

    @staticmethod
    def _equation_13(a, *args):
        """
        Equation (13) in the paper.

        :param a: The entry threshold of the trading strategy
        :param args: Other parameters needed for the equation
        :return: The value of the equation
        """

        c, theta, mu, sigma, scaler_func = args
        return np.exp(mu * ((a - theta) ** 2) / (sigma ** 2)) * (2 * (a - theta) + c) - sigma * np.sqrt(np.pi / mu) * scaler_func(a)

    @staticmethod
    def _negative_equation_16(a, *args):
        """
        Equation (16) in the paper.

        :param a: The entry threshold of the trading strategy
        :param args: Other parameters needed for the equation
        :return: The value of the equation
        """

        c, rf, theta, mu, sigma, scaler_func, w1, w2 = args

        const_1 = (a - theta) * np.sqrt(2 * mu) / sigma

        term_1 = - (2 * (a - theta) + c + rf)
        term_2 = np.sqrt(-mu * np.pi * scaler_func(a))
        term_3 = (2 * (a - theta) + c) ** 2
        term_4 = -1 * w1(const_1) + w2(const_1)

        return -1 * term_1 * term_2 / np.sqrt(term_3 * term_4)

    def plot_threshold_vs_cost(self):
        pass


    def plot_expected_retrun_vs_cost(self):
        pass


    def plot_variance_vs_cost(self):
        pass


    def plot_sharpe_ratio_vs_cost(self):
        pass







