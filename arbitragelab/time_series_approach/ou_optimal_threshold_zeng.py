# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
The module implements the Zeng class for OU Optimal Threshold Model.
"""
# pylint: disable=invalid-name

import numpy as np
from scipy import optimize
from mpmath import nsum, inf, gamma, fac, cos, exp, ln, quad
import matplotlib.pyplot as plt

from arbitragelab.time_series_approach.ou_optimal_threshold import OUModelOptimalThreshold
from arbitragelab.util import segment


class OUModelOptimalThresholdZeng(OUModelOptimalThreshold):
    """
    This class implements the analytic solutions of the optimal trading thresholds for the series
    with mean-reverting properties. The methods are described in the following publication:
    `Zeng, Z. and Lee, C.-G. (2014). Pairs trading: optimal thresholds and profitability.
    Quantitative Finance, 14(11): 1881–1893
    <https://www.tandfonline.com/doi/pdf/10.1080/14697688.2014.917806>`_.
    """

    def __init__(self):
        """
        Initializes the module parameters.
        """

        super().__init__()

        segment.track('OUModelOptimalThresholdZeng')

    def expected_trade_length(self, a: float, b: float) -> float:
        """
        Calculates the expected trade length.

        :param a: (float) The entry threshold of the trading strategy.
        :param b: (float) The exit threshold of the trading strategy.
        :return: (float) The expected trade length of the trading strategy.
        """

        a_trans = self._transform_to_dimensionless(a)
        b_trans = self._transform_to_dimensionless(b)

        const_1 = max(a_trans, b_trans)
        const_2 = min(a_trans, b_trans)

        middle_term = lambda k: gamma((2 * k + 1) / 2) * ((1.414 * const_1) ** (2 * k + 1) -
                                                          (1.414 * const_2) ** (2 * k + 1)) / fac(2 * k + 1)
        term = nsum(middle_term, [0, inf]) / 2
        expected_trade_length = float(term) / self.mu

        return expected_trade_length

    def trade_length_variance(self, a: float, b: float) -> float:
        """
        Calculates the expected trade length.

        :param a: (float) The entry threshold of the trading strategy.
        :param b: (float) The exit threshold of the trading strategy.
        :return: (float) The expected trade length of the trading strategy.
        """

        a_trans = self._transform_to_dimensionless(a)
        b_trans = self._transform_to_dimensionless(b)

        const_1 = max(a_trans, b_trans)
        const_2 = min(a_trans, b_trans)

        term_1 = self._w1(const_1) - self._w1(const_2) - self._w2(const_1) + self._w2(const_2)
        term_2 = np.exp((const_2 ** 2 - const_1 ** 2) / 4) * (self._g_1(const_1, const_2) -
                                                              self._g_2(const_1, const_2))

        middle_term = lambda k: gamma(k) * ((1.414 * const_1) ** (2*k) - (1.414 * const_2) ** (2*k)) / fac(2*k)
        term_3 = float(nsum(middle_term, [1, inf])/2)

        trade_length_variance = (term_1 + (term_2 - (term_3 ** 2))) / (self.mu ** 2)

        return trade_length_variance

    def expected_return(self, a: float, b: float, c: float) -> float:
        """
        Calculates the expected return.

        :param a: (float) The entry threshold of the trading strategy.
        :param b: (float) The exit threshold of the trading strategy.
        :param c: (float) The transaction costs of the trading strategy.
        :return: (float) The expected return of the trading strategy.
        """

        return (abs(a - b) - c) / self.expected_trade_length(a, b)

    def return_variance(self, a: float, b: float, c: float) -> float:
        """
        Calculates the variance of return.

        :param a: (float) The entry threshold of the trading strategy.
        :param b: (float) The exit threshold of the trading strategy.
        :param c: (float) The transaction costs of the trading strategy.
        :return: (float) The variance of return of the trading strategy.
        """

        return (abs(a - b) - c) ** 2 * self.trade_length_variance(a, b) / (self.expected_trade_length(a, b) ** 3)

    def sharpe_ratio(self, a: float, b: float, c: float, rf: float) -> float:
        """
        Calculates the Sharpe ratio.

        :param a: (float) The entry threshold of the trading strategy.
        :param b: (float) The exit threshold of the trading strategy.
        :param c: (float) The transaction costs of the trading strategy.
        :param rf: (float) The risk free rate.
        :return: (float) The Sharpe ratio of the strategy.
        """

        r = rf / self.expected_trade_length(a, b)

        return (self.expected_return(a, b, c) - r) / np.sqrt(self.return_variance(a, b ,c))

    def get_threshold_by_conventional_optimal_rule(self, c: float, initial_guess: float = None) -> tuple:
        """
        Solves equation (20) in the paper to get the optimal trading thresholds.

        :param c: (float) The transaction costs of the trading strategy.
        :param initial_guess: (float) The initial guess of the entry threshold
            for a short position in the dimensionless system.
        :return: (tuple) The values of the optimal trading thresholds.
        """

        c_trans = c * np.sqrt((2 * self.mu)) / self.sigma
        args = (c_trans, np.vectorize(self._equation_term))

        # Setting up the initial guess
        if initial_guess is None:
            initial_guess = c_trans + 1e-2 * np.sqrt((2 * self.mu)) / self.sigma

        root = optimize.fsolve(self._equation_20, initial_guess, args=args)[0]
        a_s, b_s = self._back_transform_from_dimensionless(root), self._back_transform_from_dimensionless(0)
        a_l, b_l = self._back_transform_from_dimensionless(-root), self._back_transform_from_dimensionless(-0)

        return a_s, b_s, a_l, b_l

    def get_threshold_by_new_optimal_rule(self, c: float, initial_guess: float = None) -> tuple:
        """
        Solves equation (23) in the paper to get the optimal trading thresholds.

        :param c: (float) The transaction costs of the trading strategy.
        :param initial_guess: (float) The initial guess of the entry threshold
            for a short position in the dimensionless system.
        :return: (tuple) The values of the optimal trading thresholds.
        """

        c_trans = c * np.sqrt((2 * self.mu)) / self.sigma
        args = (c_trans, np.vectorize(self._equation_term))

        # Setting up the initial guess
        if initial_guess is None:
            initial_guess = c_trans + 1e-2 * np.sqrt((2 * self.mu)) / self.sigma

        root = optimize.fsolve(self._equation_23, initial_guess, args=args)[0]
        a_s, b_s = self._back_transform_from_dimensionless(root), self._back_transform_from_dimensionless(-root)
        a_l, b_l = self._back_transform_from_dimensionless(-root), self._back_transform_from_dimensionless(root)

        return a_s, b_s, a_l, b_l

    def _transform_to_dimensionless(self, const: float) -> float:
        """
        Transforms input value to the dimensionless system.

        :param const: The value to transform.
        :return: (float) The value in the dimensionless system.
        """

        return (const - self.theta) * np.sqrt((2 * self.mu)) / self.sigma

    def _back_transform_from_dimensionless(self, const: float) -> float:
        """
        Back transforms input value from the dimensionless system.

        :param const: The value in the dimensionless system.
        :return: (float) The original Value.
        """

        return const / np.sqrt((2 * self.mu)) * self.sigma + self.theta

    @staticmethod
    def _equation_term(const: float, index: int) -> float:
        """
        A helper function for simplifying equation expression.

        :param const: (float) The input value of the function.
        :param index: (int) It could be 0 or 1.
        :return: (float) The output value of the function.
        """

        middle_term = lambda k: gamma((2 * k + 1) / 2) * ((1.414 * const) **
                                                          (2 * k + index)) /fac(2 * k + index)
        term = nsum(middle_term, [0, inf])

        return float(term)

    @staticmethod
    def _equation_20(a: float, *args: tuple) -> float:
        """
        Equation (20) in the paper.

        :param a: (float) The entry threshold of the trading strategy.
        :param args: (tuple) Other parameters needed for the equation.
        :return: (float) The value of the equation.
        """

        c, equation_term = args

        return (1 / 2) * equation_term(a, 1) - (a - c) * (1.414 / 2) * equation_term(a, 0)

    @staticmethod
    def _equation_23(a: float, *args: tuple) -> float:
        """
        Equation (23) in the paper.

        :param a: (float) The entry threshold of the trading strategy.
        :param args: (tuple) Other parameters needed for the equation.
        :return: (float) The value of the equation.
        """

        c, equation_term = args

        return (1 / 2) * equation_term(a, 1) - (a - c / 2) * (1.414 / 2) * equation_term(a, 0)

    @staticmethod
    def _m(const: float) -> float:
        """
        A helper function for calculating the variance of trade length.

        :param const: (float) The input value of the function.
        :return: (float) The output value of the function.
        """

        return 2 * np.exp(-(const ** 2) / 4)

    @staticmethod
    def _m_first_order(const: float) -> float:
        """
        A helper function for calculating the variance of trade length.

        :param const: (float) The input value of the function.
        :return: (float) The output value of the function.
        """

        middle_term = lambda k: ln(k) * exp(-(k ** 2) / 2) * cos(const * k)
        term = quad(middle_term, [0, inf])

        return -2 * np.sqrt(2 / np.pi) * np.exp((const ** 2) / 4) * float(term)

    @staticmethod
    def _m_second_order(const: float) -> float:
        """
        A helper function for calculating the variance of trade length.

        :param const: (float) The input value of the function.
        :return: (float) The output value of the function.
        """

        middle_term = lambda k: (ln(k) ** 2) * exp(-(k ** 2) / 2) * cos(const * k)
        term = quad(middle_term, [0, inf])

        return 2 * np.sqrt(2 / np.pi) * np.exp((const ** 2) / 4) * float(term)  - ((np.pi ** 2) / 2) * np.exp(-(const ** 2) / 4)

    def _g_1(self, const_1: float, const_2: float):
        """
        A helper function for calculating the variance of trade length.

        :param const_1: (float) The first input value of the function.
        :param const_2: (float) The second input value of the function.
        :return: (float) The output value of the function.
        """

        numerator = self._m_second_order(const_2) * self._m(const_1) - self._m_first_order(const_1) * self._m_first_order(const_2)
        denominator = self._m(const_1) ** 2

        return numerator / denominator

    def _g_2(self, const_1: float, const_2: float) -> float:
        """
        A helper function for calculating the variance of trade length.

        :param const_1: (float) The first input value of the function.
        :param const_2: (float) The second input value of the function.
        :return: (float) The output value of the function.
        """

        numerator_1 = self._m_second_order(const_1) * self._m(const_2) + self._m_first_order(const_1) * self._m_first_order(const_2)
        denominator_1 = self._m(const_1) ** 2

        numerator_2 =  -2 * (self._m_first_order(const_1) ** 2) * self._m(const_2)
        denominator_2 = self._m(const_1) ** 3

        return numerator_1 / denominator_1 + numerator_2 / denominator_2

    def plot_target_vs_c(self, target: str, method: str, c_list: list, rf: float = 0) -> plt.figure:
        """
        Plots target versus transaction costs.

        :param target: (str) The target values to plot. The options are
            ["a_s", "b_s", "a_l", "b_l", "expected_return", "return_variance", "sharpe_ratio",
            "expected_trade_length", "trade_length_variance"].
        :param method: (str) The method for calculating the optimal thresholds. The options are
            ["conventional_optimal_rule", "new_optimal_rule"]
        :param c_list: (list) A list contains transaction costs.
        :param rf: (float) The risk free rate. It is only needed when the target is "sharpe_ratio".
        :return: (plt.figure) Figure that plots target versus transaction costs.
        """

        a_s_list = []
        b_s_list = []
        a_l_list = []
        b_l_list = []
        rf_list = [rf] * len(c_list)

        if method == "conventional_optimal_rule":
            for c in c_list:
                a_s, b_s, a_l, b_l = self.get_threshold_by_conventional_optimal_rule(c)
                a_s_list.append(a_s)
                b_s_list.append(b_s)
                a_l_list.append(a_l)
                b_l_list.append(b_l)

        elif method == "new_optimal_rule":
            for c in c_list:
                a_s, b_s, a_l, b_l = self.get_threshold_by_new_optimal_rule(c)
                a_s_list.append(a_s)
                b_s_list.append(b_s)
                a_l_list.append(a_l)
                b_l_list.append(b_l)

        else:
            raise Exception("Incorrect method. "
                            "Please use one of the options "
                            "[\"conventional_optimal_rule\", \"new_optimal_rule\"].")

        # Mapping target to the setting of the plot
        mapping = {
        "a_s": (a_s_list, "Optimal Entry Thresholds For a Short Position vs Trans. Costs", "a_s"),
        "b_s": (b_s_list, "Optimal Exit Thresholds For a Short Position vs Trans. Costs", "b_s"),
        "a_l": (a_l_list, "Optimal Exit Thresholds For a Long Position vs Trans. Costs", "a_l"),
        "b_l": (b_l_list, "Optimal Entry Thresholds For a Long Position vs Trans. Costs", "b_l"),

        "expected_return": (np.vectorize(self.expected_return)(a_s_list, b_s_list, c_list),
                            "Expected Returns vs Trans. Costs", "Expected Return"),

        "return_variance": (np.vectorize(self.return_variance)(a_s_list, b_s_list, c_list),
                            "Variances of Return vs Trans. Costs", "Variances of Return"),

        "sharpe_ratio": (np.vectorize(self.sharpe_ratio)(a_s_list, b_s_list, c_list, rf_list),
                         "Sharpe Ratios vs Trans. Costs", "Sharpe Ratio"),

        "expected_trade_length": (np.vectorize(self.expected_trade_length)(a_s_list, b_s_list),
                                  "Expected Trade Lengths vs Trans. Costs", "Expected Trade Length"),

        "trade_length_variance": (np.vectorize(self.trade_length_variance)(a_s_list, b_s_list),
                                  "Variance of Trade Lengths vs Trans. Costs", "Variance of Trade Length")
        }

        fig = plt.figure()

        if target in mapping.keys():
            y_values, title, label = mapping[target]
            plt.plot(c_list, y_values)
            plt.title(title)
            plt.ylabel(label)

        else:
            raise Exception("Incorrect target. "
                            "Please use one of the options "
                            "[\"a\", \"b\", \"expected_return\", \"return_variance\","
                            "\"sharpe_ratio\", \"expected_trade_length\", \"trade_length_variance\"].")

        plt.xlabel("Transaction Cost c")  # x label

        return fig

    def plot_sharpe_ratio_vs_rf(self, method: str, rf_list: list, c: float) -> plt.figure:
        """
        Plots the Sharpe ratio versus risk free rates.

        :param method: (str) The method for calculating the optimal thresholds. The options are
            ["conventional_optimal_rule", "new_optimal_rule"]
        :param rf_list: (list) A list contains risk free rates.
        :param c: (float) The transaction costs of the trading strategy.
        :return: (plt.figure) Figure that plots target versus risk free rates.
        """

        a_list = []
        b_list = []
        c_list = [c] * len(rf_list)

        if method == "conventional_optimal_rule":
            a, b, _, _ = self.get_threshold_by_conventional_optimal_rule(c)
            a_list = [a] * len(rf_list)
            b_list = [b] * len(rf_list)

        elif method == "new_optimal_rule":
            a, b, _, _ = self.get_threshold_by_new_optimal_rule(c)
            a_list = [a] * len(rf_list)
            b_list = [b] * len(rf_list)

        else:
            raise Exception("Incorrect method. "
                            "Please use one of the options "
                            "[\"conventional_optimal_rule\", \"new_optimal_rule\"].")

        fig = plt.figure()
        func = np.vectorize(self.sharpe_ratio)
        plt.plot(rf_list, func(a_list, b_list, c_list, rf_list))
        plt.title("Sharpe Ratios vs Risk−free Rates")
        plt.ylabel("Sharpe Ratio")
        plt.xlabel("Risk−free Rate rf")

        return fig
