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


class OUModelOptimalThreshold:
    """
    This class contains base functions for modules that calculate optimal O-U model trading thresholds through time-series approaches.
    """

    def __init__(self):
        """
        Initializes the module parameters.
        """

        self.theta = None  # The long-term mean of the O-U process
        self.mu = None  # The speed at which the values will regroup around the long-term mean
        self.sigma = None  # The amplitude of randomness of the O-U process

        # devadarsh.track('OUModelOptimalThreshold')

    def construct_ou_model_from_given_parameters(self, theta: float, mu: float, sigma: float):
        """
        Initializes the O-U process from given parameters.

        :param theta: (float/int) The long-term mean of the O-U process
        :param mu: (float/int) The speed at which the values will regroup around the long-term mean
        :param sigma: (float/int) The amplitude of randomness of the O-U process
        """

        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def fit_ou_model_to_data(self, data: Union[np.array, pd.DataFrame], data_frequency: str):
        """
        Fits the O-U process to log values of the given data.

        :param data: (np.array/pd.DataFrame) It could be a single time series or a time series of two assets prices. The dimensions should be either n x 1 or n x 2.
        :param data_frequency: (str) Data frequency ["D" - daily, "M" - monthly, "Y" - yearly].
        """

        # Setting delta parameter using data frequency
        self._fit_delta(data_frequency=data_frequency)

        if len(data.shape) == 1:  # If the input is a single time series
            self._fit_to_single_time_series(data=data)
        elif data.shape[1] == 2:  # If the input is time series of two assets prices
            self._fit_to_assets(data=data)
        else:
            raise Exception("The number of dimensions for input data is incorrect. "
                            "Please provide a 1 or 2-dimensional array or dataframe.")

    def _fit_data(self):
        """
        Checks the type of input data and returns its log values.

        :return: (np.array) Log values of input data.
        """

        # Checking the data type
        if isinstance(self.data, np.ndarray):
            self.data = self.data.transpose()
            data = self.data
        else:
            data = data.to_numpy().transpose()

        return np.log(data)

    def _fit_delta(self, data_frequency: str):
        """
        Sets the value of delta_t based on the input parameter.

        :param data_frequency: (str) Data frequency ["D" - daily, "M" - monthly, "Y" - yearly].
        """

        if data_frequency == "D":
            self.delta_t = 1 / 252
        elif data_frequency == "M":
            self.delta_t = 1 / 12
        elif data_frequency == "Y":
            self.delta_t = 1
        else:
            raise Exception("Incorrect data frequency. "
                            "Please use one of the options [\"D\", \"M\", \"Y\"].")

    def _fit_to_single_time_series(self, data: Union[np.array, pd.DataFrame] = None):
        """
        Fits the O-U process to a single time series.

        :param data: (np.array/pd.DataFrame) A single time series with dimensions n x 1.
        """

        if data is not None:
            self.data = data

        # Fitting the model
        parameters = self._optimal_coefficients(self._fit_data())

        # Setting the O-U process parameters
        self.theta = parameters[0]
        self.mu = parameters[1]
        self.sigma = parameters[2]

    @staticmethod
    def _get_spread(assets: np.array, beta: float):
        """
        Constructs a time series of spread based on log values of two given asset prices.

        :param prices: (np.array) A time series contains log values of two assets prices with dimensions n x 2.
        :param beta: (float) A coefficient representing the weight of the second asset.
        :return: (np.array) A time series of spread with dimensions n x 1.
        """

        # Calculated as: [log(Pt) - log(P0)] - beta * [log(Qt) - log(Q0)]
        spread = (assets[0][:] - assets[0][0]) - beta * (assets[1][:] - assets[1][0])

        return spread

    def _fit_to_assets(self, data: Union[np.array, pd.DataFrame] = None):
        """
        Fits the O-U process to a time series of two assets prices.

        :param data: (np.array/pd.DataFrame) A Time series of two assets prices with dimensions n x 2.
        """

        if data is not None:
            self.data = data

        # Lambda function that calculates the O-U process coefficients
        compute_coefficients = lambda x: self._optimal_coefficients(self._get_spread(self._fit_data(), x))

        # Speeding up the calculations
        vectorized = np.vectorize(compute_coefficients)
        linspace = np.linspace(.001, 1, 100)
        result = vectorized(linspace)

        # Picking the argmax of beta
        index = result[3].argmax()

        # Setting the O-U process parameters
        self.theta = result[0][index]
        self.mu = result[1][index]
        self.sigma = result[2][index]
        self.beta = linspace[index]

    def _optimal_coefficients(self, series: np.array):
        """
        Finds the O-U process coefficients.

        :param portfolio: (np.array) A time series to fit.
        :return: (tuple) O-U process coefficients (theta, mu, sigma)
        """

        # Setting bounds
        # Theta  R, mu > 0, sigma > 0
        bounds = ((None, None), (1e-5, None), (1e-5, None))

        theta_init = np.mean(series)

        # Initial guesses for theta, mu, sigma
        initial_guess = np.array((theta_init, 100, 100))

        result = so.minimize(self._compute_log_likelihood, initial_guess,
                             args=(portfolio, self.delta_t), bounds=bounds)

        # Unpacking optimal values
        theta, mu, sigma = result.x

        return theta, mu, sigma

    @staticmethod
    def _compute_log_likelihood(params: tuple, *args: tuple):
        """
        Computes the average Log Likelihood. (p.13)

        :param params: (tuple) A tuple of three elements representing theta, mu and sigma_squared.
        :param args: (tuple) All other values that to be passed to self._compute_log_likelihood()
        :return: (float) The average log likelihood from given parameters.
        """

        # Setting given parameters
        theta, mu, sigma = params
        X, dt = args
        n = len(X)

        # Calculating log likelihood
        sigma_tilde_squared = (sigma ** 2) * (1 - np.exp(-2 * mu * dt)) / (2 * mu)

        summation_term = sum((X[1:] - X[:-1] * np.exp(-mu * dt) - theta * (1 - np.exp(-mu * dt))) ** 2)

        summation_term = -summation_term / (2 * n * sigma_tilde_squared)

        log_likelihood = (-np.log(2 * np.pi) / 2) \
                         + (-np.log(np.sqrt(sigma_tilde_squared))) \
                         + summation_term

        return -log_likelihood


class OUModelOptimalThresholdBertram(OUModelOptimalThreshold):
    """
    This class implements the analytic solutions of the optimal trading thresholds for the series
    with mean-reverting properties. The methods are described in the following publication:
    Bertram, W. K. (2010). Analytic solutions for optimal statistical arbitrage trading.
    Physica A: Statistical Mechanics and its Applications, 389(11):2234–2243.
    Link: http://www.stagirit.org/sites/default/files/articles/a_0340_ssrn-id1505073.pdf

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

        super().__init__()

        # devadarsh.track('OUModelOptimalThresholdBertram')

    def expected_return(self, a: float, m: float, c: float):
        """
        Calculates equation (11) to get the expected return given trading thresholds.

        :param a: (float) The entry threshold of the trading strategy
        :param m: (float) The exit threshold of the trading strategy
        :param c: (float) The transaction costs of the trading strategy
        :return: (float) The expected return of the strategy.
        """

        return self.mu * (m - a - c) / (np.pi * (self._erfi_scaler(m) - self._erfi_scaler(a)))

    def variance(self, a: float, m: float, c: float):
        """
        Calculates equation (12) to get the variance given trading thresholds.

        :param a: (float) The entry threshold of the trading strategy
        :param m: (float) The exit threshold of the trading strategy
        :param c: (float) The transaction costs of the trading strategy
        :return: (float) The variance of the strategy.
        """

        const_1 = (m - self.theta) * np.sqrt(2 * self.mu) / self.sigma
        const_2 = (a - self.theta) * np.sqrt(2 * self.mu) / self.sigma

        term_1 = self.mu * ((m - a - c) ** 2)
        term_2 = self._w1(const_1) - self._w1(const_2) - self._w2(const_1) + self._w2(const_2)
        term_3 = (np.pi * (self._erfi_scaler(m) - self._erfi_scaler(a))) ** 3

        return term_1 * term_2 / term_3

    def get_threshold_by_maximize_expected_return(self, c: float):
        """
        Solves equation (13) in the paper to get the optimal trading thresholds.

        :param c: (float) The transaction costs of the trading strategy
        :return: (tuple) The value of the optimal trading thresholds
        """

        args = (c, self.theta, self.mu, self.sigma, self._erfi_scaler)
        initial_guess = self.theta - self.sigma
        root = optimize.fsolve(self._equation_13, initial_guess, args=args)[0]

        return root, 2 * self.theta - root

    def get_threshold_by_maximize_sharpe_ratio(self, c: float, rf: float):
        """
        Minimize -1 * Sharpe ratio to get the optimal trading thresholds.

        :param c: (float) The transaction costs of the trading strategy
        :param rf: (float) The risk free rate
        :return: (tuple) The value of the optimal trading thresholds
        """

        args = (
        c, rf, self.theta, self.mu, self.sigma, self._erfi_scaler, np.vectorize(self._w1), np.vectorize(self._w2))
        initial_guess = self.theta - self.sigma
        sol = optimize.minimize(self._negative_sharpe_ratio, initial_guess, args=args).x[0]

        return sol, 2 * self.theta - sol

    def _erfi_scaler(self, const: float):
        """
        A helper function for simplifing equation expression

        :param const: (float) The input value of the function
        :return: (float) The output value of the function
        """

        return special.erfi((const - self.theta) * np.sqrt(self.mu) / self.sigma)

    def _w1(self, const: float):
        """
        A helper function for simplifing equation expression

        :param const: (float) The input value of the function
        :return: (float) The output value of the function
        """

        common_term = lambda k: gamma(k / 2) * ((1.414 * const) ** k) / fac(k)
        term_1 = (nsum(common_term, [1, inf]) / 2) ** 2
        term_2 = (nsum(lambda k: common_term(k) * ((-1) ** k), [1, inf]) / 2) ** 2
        w1 = term_1 - term_2

        return float(w1)

    def _w2(self, const: float):
        """
        A helper function for simplifing equation expression

        :param const: (float) The input value of the function
        :return: (float) The output value of the function
        """

        middle_term = lambda k: (digamma((2 * k - 1) / 2) - digamma(1)) * gamma((2 * k - 1) / 2) * (
                    (1.414 * const) ** (2 * k - 1)) / fac((2 * k - 1))
        w2 = nsum(middle_term, [1, inf])

        return float(w2)

    @staticmethod
    def _equation_13(a: float, *args: tuple):
        """
        Equation (13) in the paper.

        :param a: (float) The entry threshold of the trading strategy
        :param args: (tuple) Other parameters needed for the equation
        :return: (float) The value of the equation
        """

        c, theta, mu, sigma, scaler_func = args
        return np.exp(mu * ((a - theta) ** 2) / (sigma ** 2)) * (2 * (a - theta) + c) - sigma * np.sqrt(
            np.pi / mu) * scaler_func(a)

    @staticmethod
    def _negative_sharpe_ratio(a: float, *args: tuple):
        """
        Negative Sharpe ratio

        :param a: (float) The entry threshold of the trading strategy
        :param args: (tuple) Other parameters needed for the equation
        :return: (float) The value of the negative Sharpe ratio
        """

        c, rf, theta, mu, sigma, scaler_func, w1, w2 = args
        m = 2 * theta - a

        const_1 = (m - theta) * np.sqrt(2 * mu) / sigma
        const_2 = (a - theta) * np.sqrt(2 * mu) / sigma

        term_1 = m - a - c - rf
        term_2 = np.sqrt((m - a - c) ** 2)
        term_3 = mu * np.pi * (scaler_func(m) - scaler_func(a))
        term_4 = w1(const_1) - w1(const_2) - w2(const_1) + w2(const_2)

        return -1 * term_1 / term_2 * np.sqrt(term_3 / term_4)

    def plot_optimal_trading_thresholds_c(self, c_list: list):
        """
        Calculates optimal trading thresholds by maximizing expected return and plots optimal trading thresholds versus transaction costs.

        :param c_list: (list) A list contains transaction costs
        :return: (plt.Figure) Figure that plots optimal trading thresholds versus transaction costs
        """

        a_list = []
        m_list = []
        for c in c_list:
            a, m = self.get_threshold_by_maximize_expected_return(c)
            a_list.append(a)
            m_list.append(m)

        fig = plt.figure()
        plt.plot(c_list, a_list)
        plt.plot(c_list, m_list)
        plt.title("Optimal Trade Entry vs Trans. Costs")  # title
        plt.ylabel("a / m")  # y label
        plt.xlabel("c")  # x label

        return fig

    def plot_maximum_expected_return(self, c_list: list):
        """
        Plots maximum expected returns versus transaction costs.

        :param c_list: (list) A list contains transaction costs
        :return: (plt.Figure) Figure that plots maximum expected returns versus transaction costs
        """

        a_list = []
        m_list = []
        for c in c_list:
            a, m = self.get_threshold_by_maximize_expected_return(c)
            a_list.append(a)
            m_list.append(m)

        func = np.vectorize(self.expected_return)

        fig = plt.figure()
        plt.plot(c_list, func(a_list, m_list, c_list))
        plt.title("Max E[Return] vs Trans. Costs")  # title
        plt.ylabel("E[Return]")  # y label
        plt.xlabel("c")  # x label

        return fig

    def plot_optimal_trading_thresholds_rf(self, c: float, rf_list: list):
        """
        Calculates optimal trading thresholds by maximizing Sharpe ratio and plots optimal trading thresholds versus risk free rates.

        :param c: (float) The transaction costs of the trading strategy
        :param rf_list: (list) A list contains risk free rates
        :return: (plt.Figure) Figure that plots optimal trading thresholds versus risk free rates
        """

        a_list = []
        m_list = []
        for rf in rf_list:
            a, m = self.get_threshold_by_maximize_sharpe_ratio(c, rf)
            a_list.append(a)
            m_list.append(m)

        fig = plt.figure()
        plt.plot(rf_list, a_list)
        plt.plot(rf_list, m_list)
        plt.title("Optimal Trade Entry vs Risk−free Rate")  # title
        plt.ylabel("a / m")  # y label
        plt.xlabel("rf")  # x label

        return fig

    def plot_maximum_sharpe_ratio(self, c: float, rf_list: list):
        """
        Plots maximum Sharpe ratios versus risk free rates.

        :param c: (float) The transaction costs of the trading strategy
        :param rf_list: (list) A list contains risk free rates
        :return: (plt.Figure) Figure that plots maximum Sharpe ratios versus risk free rates
        """

        s_list = []
        for rf in rf_list:
            a, m = self.get_threshold_by_maximize_sharpe_ratio(c, rf)
            s_list.append((self.expected_return(a, m, c) - rf) / self.variance(a, m, c))

        plt.plot(rf_list, s_list)
        plt.title("Max Sharpe Ratio vs Risk−free Rate")  # title
        plt.ylabel("Sharpe Ratio")  # y label
        plt.xlabel("rf")  # x label

        return fig


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




