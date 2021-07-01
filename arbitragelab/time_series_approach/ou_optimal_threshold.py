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
        initial_guess = self.theta - self.sigma
        root = optimize.fsolve(self._equation_13, initial_guess, args = args)[0]

        return root, 2*self.theta - root

    def get_threshold_by_maximize_sharpe_ratio(self, c: Union[float, int], rf: Union[float, int]):
        """
        Minimize -1 * Sharpe ratio to get the optimal trading thresholds.

        :param c: (float/int) The transaction costs of the trading strategy
        :param rf: (float/int) The risk free rate
        :return: (tuple) The value of the optimal trading thresholds
        """

        args = (c, rf, self.theta, self.mu, self.sigma, self._erfi_scaler, np.vectorize(self._w1), np.vectorize(self._w2))
        initial_guess = self.theta - self.sigma
        sol = optimize.minimize(self._negative_sharpe_ratio, initial_guess, args = args).x[0]

        return sol, 2*self.theta - sol

    def _erfi_scaler(self, const):
        return special.erfi((const - self.theta) * np.sqrt(self.mu) / self.sigma)

    def _w1(self, const):
        common_term = lambda k: gamma(k / 2) * ((1.414 * const) ** k) / fac(k)
        term_1 = (nsum(common_term, [1, inf]) / 2) ** 2
        term_2 = (nsum(lambda k: common_term(k) * ((-1) ** k), [1, inf]) / 2) ** 2
        w1 = term_1 - term_2

        return float(w1)

    def _w2(self, const):
        middle_term = lambda k: (digamma((2 * k - 1) / 2) - digamma(1)) * gamma((2 * k - 1) / 2) * (
                    (1.414 * const) ** (2 * k - 1)) / fac((2 * k - 1))
        w2 = nsum(middle_term, [1, inf])

        return float(w2)

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
    def _negative_sharpe_ratio(a, *args):
        """
        Negative Sharpe ratio

        :param a: The entry threshold of the trading strategy
        :param args: Other parameters needed for the equation
        :return: The value of the equation
        """

        c, rf, theta, mu, sigma, scaler_func, w1, w2 = args
        m = 2*theta - a

        const_1 = (m - theta) * np.sqrt(2 * mu) / sigma
        const_2 = (a - theta) * np.sqrt(2 * mu) / sigma

        term_1 = m - a - c - rf
        term_2 = np.sqrt((m - a - c)**2)
        term_3 = mu*np.pi*(scaler_func(m) - scaler_func(a))
        term_4 = w1(const_1) - w1(const_2) - w2(const_1) + w2(const_2)

        return -1 * term_1 / term_2 * np.sqrt(term_3 / term_4)

    def plot_fig4(self, c_list: list):
        """
        Plot Fig. 4 in the paper.

        :param c_list: (list) A list contains transaction costs
        """

        a_list = []
        m_list = []
        for c in c_list:
            a, m = self.get_threshold_by_maximize_expected_return(c)
            a_list.append(a)
            m_list.append(m)

        plt.plot(c_list, a_list)
        plt.title("Optimal Trade Entry vs Trans. Costs")  # title
        plt.ylabel("a")  # y label
        plt.xlabel("c")  # x label

        plt.show()

        func = np.vectorize(self.expected_return)
        plt.plot(c_list, func(a_list, m_list, c_list))
        plt.title("Max E[Return] vs Trans. Costs")  # title
        plt.ylabel("E[Return]")  # y label
        plt.xlabel("c")  # x label

        plt.show()

    def plot_fig5(self, c: Union[float, int], rf_list: list):
        """
        Plot Fig. 5 in the paper.

        :param c: (float/int) The transaction costs of the trading strategy
        :param rf_list: (list) A list contains risk free rates
        """

        a_list = []
        m_list = []
        s_list = []
        for rf in rf_list:
            a, m = self.get_threshold_by_maximize_sharpe_ratio(c, rf)
            a_list.append(a)
            m_list.append(m)
            s_list.append((self.expected_return(a, m, c) - rf) / self.variance(a, m, c))

        plt.plot(rf_list, a_list)
        plt.title("Optimal Trade Entry vs Risk−free Rate")  # title
        plt.ylabel("a")  # y label
        plt.xlabel("rf")  # x label

        plt.show()

        plt.plot(rf_list, s_list)
        plt.title("Max Sharpe Ratio vs Risk−free Rate")  # title
        plt.ylabel("Sharpe Ratio")  # y label
        plt.xlabel("rf")  # x label

        plt.show()









