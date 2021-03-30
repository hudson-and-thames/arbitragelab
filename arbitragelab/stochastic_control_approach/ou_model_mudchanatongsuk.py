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


class StochasticControlMudchanatongsuk:
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

        # Characteristics of Training Data.
        self.ticker_A = None # Ticker Symbol of first stock.
        self.ticker_B = None # Ticker Symbol of second stock.
        self.S = None # Log price of stock B in spread.
        self.spread = None # Constructed spread from training data.
        self.time_array = None # Time indices of training data.
        self.delta_t = 1 / 251 # Time difference between each index in data, calculated in years.

        # Estimated params from training data.
        self.sigma = None # Standard deviation of stock B.
        self.mu = None # Drift of stock B.
        self.k = None # Rate of mean reversion of spread.
        self.theta = None # Long run mean of spread.
        self.eta = None # Standard deviation of spread.
        self.rho = None # Instantaneous correlation coefficient between two standard brownian motions.

        # Params inputted by user.
        self.gamma = None # Parameter of utility function (self.gamma < 1)


    @staticmethod
    def _data_preprocessing(data):
        """
        Helper function for input data preprocessing.

        :param data: (pd.DataFrame) Pricing data of both stocks in spread.
        """

        return data.ffill()


    def fit(self, data: pd.DataFrame):
        """
        This method uses inputted training data to calculate the spread and
        estimate the parameters of the corresponding OU process.

        The spread construction implementation follows Section II A in Mudchanatongsuk (2008).

        :param data: (pd.DataFrame) Contains price series of both stocks in spread.
        """

        # Preprocessing
        data = self._data_preprocessing(data)

        # Setting instance attributes.
        self.time_array = np.arange(0, len(data)) * self.delta_t
        self.ticker_A, self.ticker_B = data.columns[0], data.columns[1]

        # Calculating the spread.
        self.S = np.log(data.loc[:, self.ticker_B])
        self.spread = np.log(data.loc[:, self.ticker_A]) - self.S

        self.spread = self.spread.to_numpy()  # Converting from pd.Series to numpy array.
        self.S = self.S.to_numpy()  # Converting from pd.Series to numpy array.
        #self._estimate_params()

        params = self._estimate_params_log_likelihood()
        #print(f"Params from likelihood : {params[:-1]}")
        self.sigma, self.mu, self.k,self.theta, self.eta, self.rho = params[:-1]


    # def _estimate_params(self):
    #     """
    #     This method implements the closed form solutions for estimators of the model parameters.
    #     These formulas for the estimators are given in Appendix of Mudchanatongsuk (2007).
    #     """
    #
    #     N = len(self.spread) - 1
    #
    #     m = (self.S[N] - self.S[0]) / N
    #
    #     S_squared = (np.power(np.diff(self.S), 2).sum() - 2 * m * (self.S[N] - self.S[0]) + N*m*m) / N
    #
    #
    #     p = (1 / (N * np.power(self.spread[:-1], 2).sum() - self.spread[:-1].sum() ** 2)) * \
    #         (N * np.multiply(self.spread[1:], self.spread[:-1]).sum() -
    #          (self.spread[N] - self.spread[0]) * np.sum(self.spread[:-1]) - self.spread[:-1].sum() ** 2)
    #     # TODO : This calculation of p is incorrect. Calculation of q is correct.
    #
    #     p = np.exp(-0.18565618 * self.delta_t) # Correct value of p calculated from likelihood params.
    #     print(p)
    #
    #     q = (self.spread[N] - self.spread[0] + (1 - p) * self.spread[:-1].sum()) / N
    #
    #     V_squared = (1 / N) * (self.spread[N] ** 2 - self.spread[0] ** 2 + (1 + p ** 2) * np.power(self.spread[:-1], 2).sum()
    #                            - 2 * p * np.multiply(self.spread[1:], self.spread[:-1]).sum() - N * q)
    #     # TODO : This calculation of V_squared is incorrect.
    #
    #     print(V_squared)
    #
    #     self.sigma = np.sqrt(S_squared / self.delta_t)
    #
    #     self.mu = (m / self.delta_t) + (0.5 * (self.sigma ** 2))
    #
    #     self.k = - np.log(p) / self.delta_t
    #
    #     self.theta = q / (1 - p)
    #
    #     V_squared = (0.3038416 ** 2) * (1 - p ** 2) / (2 * self.k) # Correct value of V_squared calculated from likelihood params.
    #
    #     print(V_squared)
    #
    #
    #     C = (1/(N * np.sqrt(V_squared * S_squared))) * (np.multiply(self.spread[1:], np.diff(self.S)).sum()
    #                                                   - p * np.multiply(self.spread[:-1], np.diff(self.S)).sum()
    #                                                   - m * (self.spread[N] - self.spread[0])
    #                                                   - m * (1 - p) * self.spread[:-1].sum())
    #
    #
    #
    #     self.eta = np.sqrt(2 * self.k * V_squared / (1 - p ** 2))
    #
    #     self.rho = self.k * C * np.sqrt(V_squared * S_squared) / (self.eta * self.sigma * (1 - p))
    #
    #     print(f"Params from closed form : {(self.sigma, self.mu, self.k, self.theta, self.eta, self.rho)}")


    def optimal_portfolio_weights(self, data: pd.DataFrame, gamma = -100):
        """
        This method calculates the final optimal portfolio weights for the calculated spread.

        The calculation of weights follows Section III in Mudchanatongsuk (2008), specifically equation 28.

        :param data: (pd.DataFrame) Contains price series of both stocks in spread.
        :param gamma: (float) Parameter of utility function (gamma < 1)
        """

        if gamma >= 1:
            raise Exception("Please make sure value of gamma is less than 1.")

        # Preprocessing
        data = self._data_preprocessing(data)

        # Setting instance attributes.
        self.gamma = gamma
        t = np.arange(0, len(data)) * self.delta_t
        tau = t[-1] - t

        # Calculating spread.
        x = np.log(data.iloc[:, 0]) - np.log(data.iloc[:, 1])
        x = x.to_numpy()  # Converting from pd.Series to numpy array.

        # Calculating the alpha and beta functions.
        alpha_t, beta_t = self._alpha_beta_calc(tau)

        # Calculating the final optimal portfolio weights.
        h = (1 / (1 - self.gamma)) * (beta_t + 2 * np.multiply(x, alpha_t)
                                      - self.k * (x - self.theta) / self.eta ** 2
                                      + self.rho * self.theta / self.eta + 0.5)

        return h


    def _alpha_beta_calc(self, tau):
        """
        This helper function computes the alpha and beta functions
        as given in equation 24 and 25 of Mudchanatongsuk (2008).

        :param tau: (np.array) Array with time till completion in years.
        """

        sqrt_gamma = np.sqrt(1 - self.gamma) # Repeating Calculation involving gamma.
        exp_calc = np.exp(2 * self.k * tau / sqrt_gamma) # Repeating Calculation involving gamma and tau series.

        alpha_t = self._alpha_calc(sqrt_gamma, exp_calc)
        beta_t = self._beta_calc(sqrt_gamma, exp_calc)

        return alpha_t, beta_t


    def _alpha_calc(self, sqrt_gamma, exp_calc):
        """
        This helper function computes the alpha function in equation 24.

        :param sqrt_gamma: (float) Repeating value.
        :param exp_calc: (np.array) Repeating series of values.
        """

        # The equation for calculation of alpha is split into two parts.
        left_calc = self.k * (1 - sqrt_gamma) / (2 * (self.eta ** 2))

        right_calc = 2 * sqrt_gamma / (1 - sqrt_gamma - (1 + sqrt_gamma) * exp_calc)

        return left_calc * (1 + right_calc)


    def _beta_calc(self, sqrt_gamma, exp_calc):
        """
        This helper function computes the beta function in equation 25.

        :param sqrt_gamma: (float) Repeating value.
        :param exp_calc: (np.array) Repeating series of values.
        """

        # The equation for calculation of beta is split into two parts.
        left_calc = 1/(2 * (self.eta ** 2) * ((1 - sqrt_gamma) - (1 + sqrt_gamma) * exp_calc))

        right_calc = self.gamma * sqrt_gamma * (self.eta ** 2 + 2 * self.rho * self.sigma * self.eta) * ((1 - exp_calc) ** 2) - \
                     self.gamma * (self.eta ** 2 + 2 * self.rho * self.sigma * self.eta + 2 * self.k * self.theta) * (1 - exp_calc)

        return left_calc * right_calc


    def _estimate_params_log_likelihood(self):
        """
        Estimates parameters of model based on log likelihood maximization.
        """

        # Setting bounds
        # sigma, mu, k, theta, eta, rho
        bounds = ((1e-5, None), (None, None), (1e-5, None), (None, None), (1e-5, None), (-1 + 1e-5, 1 - 1e-5))

        theta_init = np.mean(self.spread)

        # Initial guesses for sigma, mu, k, theta, eta, rho
        initial_guess = np.array((1, 0, 1, theta_init, 1, 0.5))

        result = so.minimize(self._compute_log_likelihood, initial_guess,
                             args=(self.spread, self.S, self.delta_t), bounds=bounds, options={'maxiter': 100000})

        # Unpacking optimal values
        sigma, mu, k, theta, eta, rho = result.x

        # Undo negation
        max_log_likelihood = -result.fun

        return sigma, mu, k, theta, eta, rho, max_log_likelihood


    @staticmethod
    def _compute_log_likelihood(params, *args):
        """
        Helper function computes log likelihood function for a set of params.
        This implementation follows Appendix of Mudchanatongsuk (2008).

        param params: (tuple) Contains values of set params.
        """

        # Setting given parameters
        sigma, mu, k ,theta, eta, rho = params
        X,S, dt = args

        y = np.array([X, S])

        matrix = np.zeros((2,2))
        matrix[0, 0] = (eta ** 2) * (1 - np.exp(-2 * k * dt)) / (2 * k)
        matrix[0, 1] = rho * eta * sigma * (1 - np.exp(k * dt)) / k
        matrix[1, 0] = matrix[0 ,1]
        matrix[1, 1] = sigma ** 2 * dt

        X = X[:-1]
        S = S[:-1]
        y = y[:, 1:]

        E_y = np.zeros((2, len(X)))
        E_y[0, :] = X * np.exp(-k * dt) + theta * (1 - np.exp(-k * dt))
        E_y[1, :] = S + (mu - 0.5 * sigma ** 2) * dt

        vec = y - E_y

        with np.errstate(all='raise'): # This was done due to np.nan's outputted in final result of scipy minimize.
            try:
                f_y_denm = (2 * math.pi * np.sqrt(np.linalg.det(matrix)))

                f_y = np.exp(-0.5 * np.einsum('ij,ij->j', vec, np.linalg.inv(matrix) @ vec)) / f_y_denm
            except FloatingPointError:
                return 0


        log_likelihood = np.log(f_y).sum()

        return -log_likelihood


    @staticmethod
    def _calc_half_life(k: float) -> float:
        """
        Function returns half life of mean reverting spread from rate of mean reversion.
        """

        return np.log(2) / k # Half life of shocks.


    def describe(self) -> pd.Series:
        """
        Method returns values of instance attributes calculated from training data.
        """

        if self.sigma is None:
            raise Exception("Please run the fit method before calling describe.")

        index = ['Ticker of first stock', 'Ticker of second stock',
                 'long-term mean of spread', 'rate of mean reversion of spread', 'standard deviation of spread', 'half-life of spread',
                 'Drift of stock B', 'standard deviation of stock B']

        data = [self.ticker_A, self.ticker_B,
                self.theta, self.k, self.eta, self._calc_half_life(self.k),
                self.mu, self.sigma]

        # Combine data and indexes into the pandas Series
        output = pd.Series(data=data, index=index)

        return output
