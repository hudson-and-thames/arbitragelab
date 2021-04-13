# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module implements the optimal pairs trading strategy using Stochastic Control Approach.

This module is a realization of the methodology in the following paper:

`Mudchanatongsuk, S., Primbs, J.A. and Wong, W., 2008, June. Optimal pairs trading: A stochastic control approach.
<http://folk.ntnu.no/skoge/prost/proceedings/acc08/data/papers/0479.pdf>`__
"""
# pylint: disable=invalid-name, too-many-instance-attributes

import math
import numpy as np
import pandas as pd
import scipy.optimize as so


class OUModelMudchanatongsuk:
    """
    This class implements a stochastic control approach to the problem of pairs trading.

    We model the log-relationship between a pair of stock prices as an OrnsteinUhlenbeck process,
    and use this to formulate a portfolio optimization based stochastic control problem.
    We are able to obtain the optimal solution to this control problem in closed
    form via the corresponding Hamilton-Jacobi-Bellman equation. This closed form solution
    is calculated under a power utility on terminal wealth.
    The parameters in the model are calculated using closed form maximum-likelihood estimation formulas.
    """

    def __init__(self):
        """
        Initializes the parameters of the module.
        """

        # Characteristics of Training Data
        self.ticker_A = None  # Ticker Symbol of first stock
        self.ticker_B = None  # Ticker Symbol of second stock
        self.S = None  # Log price of stock B in spread
        self.spread = None  # Constructed spread from training data
        self.time_array = None  # Time indices of training data
        self.delta_t = 1 / 251  # Time difference between each index in data, calculated in years

        # Estimated params from training data
        self.sigma = None  # Standard deviation of stock B
        self.mu = None  # Drift of stock B
        self.k = None  # Rate of mean reversion of spread
        self.theta = None  # Long run mean of spread
        self.eta = None  # Standard deviation of spread
        self.rho = None  # Instantaneous correlation coefficient between two standard brownian motions

        # Params inputted by user
        self.gamma = None  # Parameter of utility function (self.gamma < 1)


    @staticmethod
    def _data_preprocessing(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function for input data preprocessing.

        :param prices: (pd.DataFrame) Pricing data of both stocks in spread.
        :return: (pd.DataFrame) Processed dataframe.
        """

        return prices.ffill()


    def fit(self, prices: pd.DataFrame):
        """
        This method uses inputted training data to calculate the spread and
        estimate the parameters of the corresponding OU process.

        The spread construction implementation follows Section II A in Mudchanatongsuk (2008).

        :param prices: (pd.DataFrame) Contains price series of both stocks in spread.
        """

        # Preprocessing
        prices = self._data_preprocessing(prices)

        # Setting instance attributes
        self.time_array = np.arange(0, len(prices)) * self.delta_t
        self.ticker_A, self.ticker_B = prices.columns[0], prices.columns[1]

        # Calculating the spread
        self.S = np.log(prices.loc[:, self.ticker_B])
        self.spread = np.log(prices.loc[:, self.ticker_A]) - self.S

        self.spread = self.spread.to_numpy()  # Converting from pd.Series to numpy array
        self.S = self.S.to_numpy()  # Converting from pd.Series to numpy array

        params = self._estimate_params_log_likelihood()
        self.sigma, self.mu, self.k, self.theta, self.eta, self.rho = params[:-1]


    def spread_calc(self, prices: pd.DataFrame) -> tuple:
        """
        This method calculates the spread on test data.

        :param prices: (pd.DataFrame) Contains price series of both stocks in spread.
        :return: (tuple) Consists of time remaining array and spread numpy array.
        """

        # Preprocessing
        prices = self._data_preprocessing(prices)
        t = np.arange(0, len(prices)) * self.delta_t
        tau = t[-1] - t

        # Calculating spread
        x = np.log(prices.iloc[:, 0]) - np.log(prices.iloc[:, 1])
        x = x.to_numpy()  # Converting from pd.Series to numpy array

        return tau, x


    def optimal_portfolio_weights(self, prices: pd.DataFrame, gamma: float = -100) -> np.array:
        """
        This method calculates the final optimal portfolio weights for the calculated spread.

        The calculation of weights follows Section III in Mudchanatongsuk (2008), specifically equation 28.

        :param prices: (pd.DataFrame) Contains price series of both stocks in spread.
        :param gamma: (float) Parameter of utility function (gamma < 1).
        :return: (np.array) Optimal weights array.
        """

        if self.sigma is None:
            raise Exception("Please run the fit method before calling this method.")

        if gamma >= 1:
            raise Exception("Please make sure value of gamma is less than 1.")

        # Setting instance attributes
        self.gamma = gamma

        # Calculating spread
        tau, x = self.spread_calc(prices)

        # Calculating the alpha and beta functions
        alpha_t, beta_t = self._alpha_beta_calc(tau)

        # Calculating the final optimal portfolio weights
        h = (1 / (1 - self.gamma)) * (beta_t + 2 * np.multiply(x, alpha_t)
                                      - self.k * (x - self.theta) / self.eta ** 2
                                      + self.rho * self.theta / self.eta + 0.5)

        return h


    def _alpha_beta_calc(self, tau: np.array) -> tuple:
        """
        This helper function computes the alpha and beta functions
        as given in equation 24 and 25 of Mudchanatongsuk (2008).

        :param tau: (np.array) Array with time till completion in years.
        :return: (tuple) Alpha and beta arrays.
        """

        sqrt_gamma = np.sqrt(1 - self.gamma)  # Repeating Calculation involving gamma
        exp_calc = np.exp(2 * self.k * tau / sqrt_gamma)  # Repeating Calculation involving gamma and tau series

        # Calculating the alpha function output
        alpha_t = self._alpha_calc(sqrt_gamma, exp_calc)
        # Calculating the beta function output
        beta_t = self._beta_calc(sqrt_gamma, exp_calc)

        return alpha_t, beta_t


    def _alpha_calc(self, sqrt_gamma: float, exp_calc: np.array) -> np.array:
        """
        This helper function computes the alpha function in equation 24.

        :param sqrt_gamma: (float) Repeating value.
        :param exp_calc: (np.array) Repeating series of values.
        :return: (np.array) Alpha array.
        """

        # The equation for calculation of alpha is split into two parts
        left_calc = self.k * (1 - sqrt_gamma) / (2 * (self.eta ** 2))

        right_calc = 2 * sqrt_gamma / (1 - sqrt_gamma - (1 + sqrt_gamma) * exp_calc)

        return left_calc * (1 + right_calc)


    def _beta_calc(self, sqrt_gamma: float, exp_calc: np.array) -> np.array:
        """
        This helper function computes the beta function in equation 25.

        :param sqrt_gamma: (float) Repeating value.
        :param exp_calc: (np.array) Repeating series of values.
        :return: (np.array) Beta array.
        """

        # The equation for calculation of beta is split into two parts
        left_calc = 1/(2 * (self.eta ** 2) * ((1 - sqrt_gamma) - (1 + sqrt_gamma) * exp_calc))

        right_calc = self.gamma * sqrt_gamma * (self.eta ** 2 + 2 * self.rho * self.sigma * self.eta) * ((1 - exp_calc) ** 2) - \
                     self.gamma * (self.eta ** 2 + 2 * self.rho * self.sigma * self.eta + 2 * self.k * self.theta) * (1 - exp_calc)

        return left_calc * right_calc


    def _estimate_params_log_likelihood(self) -> tuple:
        """
        Estimates parameters of model based on log likelihood maximization.

        :return: (tuple) Consists of final estimated params.
        """

        # Setting bounds
        # sigma, mu, k, theta, eta, rho
        bounds = ((1e-5, None), (None, None), (1e-5, None), (None, None), (1e-5, None), (-1 + 1e-5, 1 - 1e-5))

        # Setting initial value of theta to the mean of the spread
        theta_init = np.mean(self.spread)

        # Initial guesses for sigma, mu, k, theta, eta, rho
        initial_guess = np.array((1, 0, 1, theta_init, 1, 0.5))

        # Using scipy minimize to calculate the max log likelihood
        result = so.minimize(self._compute_log_likelihood, initial_guess,
                             args=(self.spread, self.S, self.delta_t), bounds=bounds, options={'maxiter': 100000})

        # Unpacking optimal values
        sigma, mu, k, theta, eta, rho = result.x

        # Undo negation
        max_log_likelihood = -result.fun

        return sigma, mu, k, theta, eta, rho, max_log_likelihood


    @staticmethod
    def _compute_log_likelihood(params: tuple, *args) -> float:
        """
        Helper function computes log likelihood function for a set of params.
        This implementation follows Appendix of Mudchanatongsuk (2008).

        :param params: (tuple) Contains values of set params.
        :return: (float) negation of log likelihood.
        """

        # Setting given parameters
        sigma, mu, k, theta, eta, rho = params
        X, S, dt = args

        # Setting y as given in Appendix
        y = np.array([X, S])

        # Calculating matrix as given in equation (41) in the paper
        matrix = np.zeros((2, 2))
        matrix[0, 0] = (eta ** 2) * (1 - np.exp(-2 * k * dt)) / (2 * k)
        matrix[0, 1] = rho * eta * sigma * (1 - np.exp(k * dt)) / k
        matrix[1, 0] = matrix[0, 1]
        matrix[1, 1] = sigma ** 2 * dt

        X = X[:-1]
        S = S[:-1]
        y = y[:, 1:]

        # Calculation follows equation (40) in the paper
        E_y = np.zeros((2, len(X)))
        E_y[0, :] = X * np.exp(-k * dt) + theta * (1 - np.exp(-k * dt))
        E_y[1, :] = S + (mu - 0.5 * sigma ** 2) * dt

        vec = y - E_y

        # Calculating the joint density function given in equation (39) in the paper
        with np.errstate(all='raise'):  # This was done due to np.nan's outputted in final result of scipy minimize
            try:
                f_y_denm = (2 * math.pi * np.sqrt(np.linalg.det(matrix)))

                f_y = np.exp(-0.5 * np.einsum('ij,ij->j', vec, np.linalg.inv(matrix) @ vec)) / f_y_denm
            except FloatingPointError:
                return 0

        # Calculating the final log likelihood given in equation (38) in the paper
        log_likelihood = np.log(f_y).sum()

        # Returning the negation as we are using scipy minimize
        return -log_likelihood


    @staticmethod
    def _calc_half_life(k: float) -> float:
        """
        Function returns half life of mean reverting spread from rate of mean reversion.

        :param k: (float) K value.
        :return: (float) Half life.
        """

        return np.log(2) / k  # Half life of shocks


    def describe(self) -> pd.Series:
        """
        Method returns values of instance attributes calculated from training data.

        :return: (pd.Series) series describing parameter values.
        """

        if self.sigma is None:
            raise Exception("Please run the fit method before calling describe.")

        # List defines the indexes of the final pandas object
        index = ['Ticker of first stock', 'Ticker of second stock',
                 'long-term mean of spread', 'rate of mean reversion of spread', 'standard deviation of spread', 'half-life of spread',
                 'Drift of stock B', 'standard deviation of stock B']

        # List defines the values of the final pandas object
        data = [self.ticker_A, self.ticker_B,
                self.theta, self.k, self.eta, self._calc_half_life(self.k),
                self.mu, self.sigma]

        # Combine data and indexes into the pandas Series
        output = pd.Series(data=data, index=index)

        return output
