# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=missing-module-docstring, invalid-name, too-many-branches, too-many-statements
import numpy as np
from scipy import optimize, special
import matplotlib.pyplot as plt

from arbitragelab.time_series_approach.ou_optimal_threshold import OUModelOptimalThreshold
from arbitragelab.util import devadarsh


class OUModelOptimalThresholdBertram(OUModelOptimalThreshold):
    """
    This class implements the analytic solutions of the optimal trading thresholds for the series
    with mean-reverting properties. The methods are described in the following publication:
    `Bertram, W. K. (2010). Analytic solutions for optimal statistical arbitrage trading.
    Physica A: Statistical Mechanics and its Applications, 389(11): 2234–2243.
    <http://www.stagirit.org/sites/default/files/articles/a_0340_ssrn-id1505073.pdf>`_
    """

    def __init__(self):
        """
        Initializes the module parameters.
        """

        super().__init__()

        devadarsh.track('OUModelOptimalThresholdBertram')

    def expected_trade_length(self, a: float, m: float):
        """
        Calculates equation (9) in the paper to get the expected trade length.

        :param a: (float) The entry threshold of the trading strategy
        :param m: (float) The exit threshold of the trading strategy
        :return: (float) The expected trade length of the trading strategy
        """

        return (np.pi / self.mu) * (self._erfi_scaler(m) - self._erfi_scaler(a))

    def trade_length_variance(self, a: float, m: float):
        """
        Calculates equation (10) in the paper to get the variance of trade length.

        :param a: (float) The entry threshold of the trading strategy
        :param m: (float) The exit threshold of the trading strategy
        :return: (float) The variance of trade length of the trading strategy
        """

        const_1 = (m - self.theta) * np.sqrt(2 * self.mu) / self.sigma
        const_2 = (a - self.theta) * np.sqrt(2 * self.mu) / self.sigma

        term_1 = self._w1(const_1) - self._w1(const_2) - self._w2(const_1) + self._w2(const_2)
        term_2 = (self.mu) ** 2

        return term_1 / term_2

    def expected_return(self, a: float, m: float, c: float):
        """
        Calculates equation (5) in the paper to get the expected return.

        :param a: (float) The entry threshold of the trading strategy
        :param m: (float) The exit threshold of the trading strategy
        :param c: (float) The transaction costs of the trading strategy
        :return: (float) The expected return of the trading strategy
        """

        return (m - a - c) / self.expected_trade_length(a, m)

    def return_variance(self, a: float, m: float, c: float):
        """
        Calculates equation (6) in the paper to get the variance of return.

        :param a: (float) The entry threshold of the trading strategy
        :param m: (float) The exit threshold of the trading strategy
        :param c: (float) The transaction costs of the trading strategy
        :return: (float) The variance of return of the trading strategy
        """

        return (m - a - c) ** 2 * self.trade_length_variance(a, m) / (self.expected_trade_length(a, m) ** 3)

    def sharpe_ratio(self, a: float, m: float, c: float, rf: float):
        """
        Calculates equation (15) in the paper to get the Sharpe ratio.

        :param a: (float) The entry threshold of the trading strategy
        :param m: (float) The exit threshold of the trading strategy
        :param c: (float) The transaction costs of the trading strategy
        :param rf: (float) The risk free rate
        :return: (float) The Sharpe ratio of the strategy
        """

        r = rf / self.expected_trade_length(a, m)

        return (self.expected_return(a, m, c) - r) / np.sqrt(self.return_variance(a, m ,c))



    def get_threshold_by_maximize_expected_return(self, c: float, initial_guess: float = None):
        """
        Solves equation (13) in the paper to get the optimal trading thresholds.

        :param c: (float) The transaction costs of the trading strategy.
        :param initial_guess: (float) The initial guess of the entry threshold.
        :return: (tuple) The values of the optimal trading thresholds.
        """

        # equation (13) in the paper
        equation = lambda a: np.exp(self.mu * ((a - self.theta) ** 2) / (self.sigma ** 2)) * (2 * (a - self.theta) + c) \
                            - self.sigma * np.sqrt(np.pi / self.mu) * self._erfi_scaler(a)

        # Setting up the initial guess
        if initial_guess is None:
            initial_guess = self.theta - c - 1e-2

        root = optimize.fsolve(equation, initial_guess)[0]

        return root, 2 * self.theta - root

    def get_threshold_by_maximize_sharpe_ratio(self, c: float, rf: float, initial_guess: float = None):
        """
        Minimize -1 * Sharpe ratio to get the optimal trading thresholds.

        :param c: (float) The transaction costs of the trading strategy.
        :param rf: (float) The risk free rate.
        :param initial_guess: (float) The initial guess of the entry threshold.
        :return: (tuple) The values of the optimal trading thresholds.
        """

        negative_sharpe_ratio = lambda a: -1 * self.sharpe_ratio(a, 2 * self.theta - a, c, rf)
        negative_sharpe_ratio = np.vectorize(negative_sharpe_ratio)

        # Setting up the initial guess
        if initial_guess is None:
            initial_guess = self.theta - rf - c - 1e-2

        sol = optimize.minimize(negative_sharpe_ratio, initial_guess, method="Nelder-Mead").x[0]

        return sol, 2 * self.theta - sol

    def _erfi_scaler(self, const: float):
        """
        A helper function for simplifing equation expression

        :param const: (float) The input value of the function
        :return: (float) The output value of the function
        """

        return special.erfi((const - self.theta) * np.sqrt(self.mu) / self.sigma)

    def plot_target_vs_c(self, target: str, method: str, c_list: list, rf: float = 0):
        """
        Plots target versus transaction costs.

        :param target: (str) The target values to plot. The options are
            ["a", "m", "expected_return", "return_variance", "sharpe_ratio", "expected_trade_length", "trade_length_variance"].
        :param method: (str) The method for calculating the optimal thresholds. The options are
            ["maximize_expected_return", "maximize_sharpe_ratio"]
        :param c_list: (list) A list contains transaction costs.
        :param rf: (float) The risk free rate. It is only needed when the target is "sharpe_ratio"
            or when the method is "maximize_sharpe_ratio".
        :return: (plt.figure) Figure that plots target versus transaction costs.
        """

        a_list = []
        m_list = []
        rf_list = [rf] * len(c_list)

        if method == "maximize_expected_return":
            for c in c_list:
                a, m = self.get_threshold_by_maximize_expected_return(c)
                a_list.append(a)
                m_list.append(m)

        elif method == "maximize_sharpe_ratio":
            for c in c_list:
                a, m = self.get_threshold_by_maximize_sharpe_ratio(c, rf)
                a_list.append(a)
                m_list.append(m)

        else:
            raise Exception("Incorrect method. "
                            "Please use one of the options "
                            "[\"maximize_expected_return\", \"maximize_sharpe_ratio\"].")

        fig = plt.figure()

        if target == "a":
            plt.plot(c_list, a_list)
            plt.title("Optimal Entry Thresholds vs Trans. Costs")
            plt.ylabel("a")

        elif target == "m":
            plt.plot(c_list, m_list)
            plt.title("Optimal Exit Thresholds vs Trans. Costs")
            plt.ylabel("m")

        elif target == "expected_return":
            func = np.vectorize(self.expected_return)
            plt.plot(c_list, func(a_list, m_list, c_list))
            plt.title("Expected Returns vs Trans. Costs")
            plt.ylabel("Expected Return")

        elif target == "return_variance":
            func = np.vectorize(self.return_variance)
            plt.plot(c_list, func(a_list, m_list, c_list))
            plt.title("Variances of Return vs Trans. Costs")
            plt.ylabel("Variances of Return")

        elif target == "sharpe_ratio":
            func = np.vectorize(self.sharpe_ratio)
            plt.plot(c_list, func(a_list, m_list, c_list, rf_list))
            plt.title("Sharpe Ratios vs Trans. Costs")
            plt.ylabel("Sharpe Ratio")

        elif target == "expected_trade_length":
            func = np.vectorize(self.expected_trade_length)
            plt.plot(c_list, func(a_list, m_list))
            plt.title("Expected Trade Lengths vs Trans. Costs")
            plt.ylabel("Expected Trade Length")

        elif target == "trade_length_variance":
            func = np.vectorize(self.trade_length_variance)
            plt.plot(c_list, func(a_list, m_list))
            plt.title("Variance of Trade Lengths vs Trans. Costs")
            plt.ylabel("Variance of Trade Length")

        else:
            raise Exception("Incorrect target. "
                            "Please use one of the options "
                            "[\"a\", \"m\", \"expected_return\", \"return_variance\","
                            "\"sharpe_ratio\", \"expected_trade_length\", \"trade_length_variance\"].")

        plt.xlabel("Transaction Cost c")  # x label

        return fig

    def plot_target_vs_rf(self, target: str, method: str, rf_list: list, c: float):
        """
        Plots target versus risk free rates.

        :param target: (str) The target values to plot. The options are
            ["a", "m", "expected_return", "return_variance", "sharpe_ratio", "expected_trade_length", "trade_length_variance"].
        :param method: (str) The method for calculating the optimal thresholds. The options are
            ["maximize_expected_return", "maximize_sharpe_ratio"].
        :param rf_list: (list) A list contains risk free rates.
        :param c: (float) The transaction costs of the trading strategy.
        :return: (plt.figure) Figure that plots target versus risk free rates.
        """

        a_list = []
        m_list = []
        c_list = [c] * len(rf_list)

        if method == "maximize_expected_return":
            a, m = self.get_threshold_by_maximize_expected_return(c)
            a_list = [a] * len(rf_list)
            m_list = [m] * len(rf_list)

        elif method == "maximize_sharpe_ratio":
            for rf in rf_list:
                a, m = self.get_threshold_by_maximize_sharpe_ratio(c, rf)
                a_list.append(a)
                m_list.append(m)

        else:
            raise Exception("Incorrect method. "
                            "Please use one of the options "
                            "[\"maximize_expected_return\", \"maximize_sharpe_ratio\"].")

        fig = plt.figure()

        if target == "a":
            plt.plot(rf_list, a_list)
            plt.title("Optimal Entry Thresholds vs Risk−free Rates")
            plt.ylabel("a")

        elif target == "m":
            plt.plot(rf_list, m_list)
            plt.title("Optimal Exit Thresholds vs Risk−free Rates")
            plt.ylabel("m")

        elif target == "expected_return":
            func = np.vectorize(self.expected_return)
            plt.plot(rf_list, func(a_list, m_list, c_list))
            plt.title("Expected Returns vs Risk−free Rates")
            plt.ylabel("Expected Return")

        elif target == "return_variance":
            func = np.vectorize(self.return_variance)
            plt.plot(rf_list, func(a_list, m_list, c_list))
            plt.title("Variances of Return vs Risk−free Rates")
            plt.ylabel("Variances of Return")

        elif target == "sharpe_ratio":
            func = np.vectorize(self.sharpe_ratio)
            plt.plot(rf_list, func(a_list, m_list, c_list, rf_list))
            plt.title("Sharpe Ratios vs Risk−free Rates")
            plt.ylabel("Sharpe Ratio")

        elif target == "expected_trade_length":
            func = np.vectorize(self.expected_trade_length)
            plt.plot(rf_list, func(a_list, m_list))
            plt.title("Expected Trade Lengths vs Risk−free Rates")
            plt.ylabel("Expected Trade Length")

        elif target == "trade_length_variance":
            func = np.vectorize(self.trade_length_variance)
            plt.plot(rf_list, func(a_list, m_list))
            plt.title("Variance of Trade Lengths vs Risk−free Rates")
            plt.ylabel("Variance of Trade Length")

        else:
            raise Exception("Incorrect target. "
                            "Please use one of the options "
                            "[\"a\", \"m\", \"expected_return\", \"return_variance\","
                            "\"sharpe_ratio\", \"expected_trade_length\", \"trade_length_variance\"].")

        plt.xlabel("Risk−free Rate rf")  # x label

        return fig
