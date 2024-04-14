"""
The module implements the base class for OU Optimal Threshold Model.
"""
# pylint: disable=invalid-name

from typing import Union

import numpy as np
import pandas as pd
from scipy import optimize
from mpmath import nsum, inf, gamma, digamma, fac


class OUModelOptimalThreshold:
    """
    This class contains base functions for modules that calculate optimal O-U model trading thresholds
    through time-series approaches.
    """

    def __init__(self):
        """
        Initializes the module parameters.
        """

        self.theta = None  # The long-term mean of the O-U process
        self.mu = None  # The speed at which the values will regroup around the long-term mean
        self.sigma = None  # The amplitude of randomness of the O-U process

        # Parameters for fitting function
        self.data = None  # Fitting data provided by the user
        self.delta_t = None  # Delta between observations, calculated in years
        self.beta = None  # Optimal ratio between two assets

    def construct_ou_model_from_given_parameters(self, theta: float, mu: float, sigma: float):
        """
        Initializes the O-U process from given parameters.

        :param theta: (float) The long-term mean of the O-U process.
        :param mu: (float) The speed at which the values will regroup around the long-term mean.
        :param sigma: (float) The amplitude of randomness of the O-U process.
        """

        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def fit_ou_model_to_data(self, data: Union[np.array, pd.DataFrame], data_frequency: str):
        """
        Fits the O-U process to log values of the given data.

        :param data: (np.array/pd.DataFrame) It could be a single time series or a time series of two assets prices.
            The dimensions should be either n x 1 or n x 2.
        :param data_frequency: (str) Data frequency ["D" - daily, "M" - monthly, "Y" - yearly].
        """

        # Setting delta parameter using data frequency
        self._fit_delta(data_frequency=data_frequency)

        if len(data.shape) == 1:  # The input is a single time series
            self._fit_to_single_time_series(data=data)

        elif data.shape[1] == 2:  # The input is time series of two assets prices
            self._fit_to_assets(data=data)

        else:
            raise Exception("The number of dimensions for input data is incorrect. "
                            "Please provide a 1 or 2-dimensional array or dataframe.")

    def _fit_data(self) -> np.array:
        """
        Checks the type of input data and returns its log values.

        :return: (np.array) Log values of input data.
        """

        # Checking the data type
        if isinstance(self.data, np.ndarray):
            data = self.data.transpose()
        else:
            data = self.data.to_numpy().transpose()

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

    def _fit_to_single_time_series(self, data: Union[np.array, pd.DataFrame]):
        """
        Fits the O-U process to a single time series.

        :param data: (np.array/pd.DataFrame) A single time series with dimensions n x 1.
        """

        self.data = data

        # Fitting the process
        parameters = self._optimal_coefficients(self._fit_data())

        # Setting the O-U process parameters
        self.theta = parameters[0]
        self.mu = parameters[1]
        self.sigma = parameters[2]

    @staticmethod
    def _get_spread(assets: np.array, beta: float) -> np.array:
        """
        Constructs a time series of spread based on log values of two given asset prices.

        :param assets: (np.array) A time series contains log values of two assets prices with dimensions n x 2.
        :param beta: (float) A coefficient representing the weight of the second asset.
        :return: (np.array) A time series of spread with dimensions n x 1.
        """

        # Calculated as: [log(Pt) - log(P0)] - beta * [log(Qt) - log(Q0)]
        spread = (assets[0][:] - assets[0][0]) - beta * (assets[1][:] - assets[1][0])

        return spread

    def _fit_to_assets(self, data: Union[np.array, pd.DataFrame]):
        """
        Fits the O-U process to a time series of two assets prices.

        :param data: (np.array/pd.DataFrame) A time series of two assets prices with dimensions n x 2.
        """

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

    def _optimal_coefficients(self, series: np.array) -> tuple:
        """
        Finds the O-U process coefficients.

        :param series: (np.array) A time series to fit.
        :return: (tuple) O-U process coefficients (theta, mu, sigma).
        """

        # Setting bounds
        # Theta  R, mu > 0, sigma > 0
        bounds = ((None, None), (1e-5, None), (1e-5, None))

        theta_init = np.mean(series)

        # Initial guesses for theta, mu, sigma
        initial_guess = np.array((theta_init, 100, 100))

        result = optimize.minimize(self._compute_log_likelihood, initial_guess,
                                   args=(series, self.delta_t), bounds=bounds)

        # Unpacking optimal values
        theta, mu, sigma = result.x

        # Undo negation
        max_log_likelihood = -result.fun

        return theta, mu, sigma, max_log_likelihood

    @staticmethod
    def _compute_log_likelihood(params: tuple, *args: tuple) -> float:
        """
        Computes the average Log Likelihood.

        :param params: (tuple) A tuple of three elements representing theta, mu and sigma.
        :param args: (tuple) All other values that to be passed to self._compute_log_likelihood().
        :return: (float) The value of log likelihood.
        """

        # Setting given parameters
        theta, mu, sigma = params
        X, dt = args
        n = len(X)

        # Calculating log likelihood
        sigma_tilde_squared = (sigma ** 2) * (1 - np.exp(-2 * mu * dt)) / (2 * mu)

        summation_term = sum((X[1:] - X[:-1] * np.exp(-mu * dt) - theta * (1 - np.exp(-mu * dt))) ** 2)

        summation_term = -summation_term / (2 * n * sigma_tilde_squared)

        log_likelihood = (-np.log(2 * np.pi) / 2) + (-np.log(np.sqrt(sigma_tilde_squared))) + summation_term

        return -log_likelihood

    @staticmethod
    def _w1(const: float) -> float:
        """
        A helper function for simplifying equation expression.

        :param const: (float) The input value of the function.
        :return: (float) The output value of the function.
        """

        common_term = lambda k: gamma(k / 2) * ((1.414 * const) ** k) / fac(k)
        term_1 = (nsum(common_term, [1, inf]) / 2) ** 2
        term_2 = (nsum(lambda k: common_term(k) * ((-1) ** k), [1, inf]) / 2) ** 2
        w1 = term_1 - term_2

        return float(w1)

    @staticmethod
    def _w2(const: float) -> float:
        """
        A helper function for simplifying equation expression.

        :param const: (float) The input value of the function.
        :return: (float) The output value of the function.
        """

        middle_term = lambda k: (digamma((2 * k - 1) / 2) - digamma(1)) * gamma((2 * k - 1) / 2) *\
                                ((1.414 * const) ** (2 * k - 1)) / fac((2 * k - 1))
        w2 = nsum(middle_term, [1, inf])

        return float(w2)
