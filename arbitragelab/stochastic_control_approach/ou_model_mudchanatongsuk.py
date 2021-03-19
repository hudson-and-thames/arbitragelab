# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module implements the optimal pairs trading strategy using Stochastic Control Approach.

This module is a realization of the methodology in the following paper:
`Mudchanatongsuk, S., Primbs, J.A. and Wong, W., 2008, June. Optimal pairs trading: A stochastic control approach.
 In 2008 American control conference (pp. 1035-1039). IEEE.
<http://folk.ntnu.no/skoge/prost/proceedings/acc08/data/papers/0479.pdf>`__
"""

import math
import numpy as np
import pandas as pd
import scipy.optimize as so

class StochasticControlMudchanatongsuk:
    def __init__(self):

        self.ticker_A = None
        self.ticker_B = None
        self.S = None
        self.spread = None
        self.time_array = None

        self.delta_t = 1 / 251
        self.sigma = None
        self.mu = None
        self.k = None
        self.theta = None
        self.eta = None
        self.rho = None

        self.gamma = None

    @staticmethod
    def _data_preprocessing(data):

        return data.ffill()

    def fit(self, data: pd.DataFrame):

        # Preprocessing
        data = self._data_preprocessing(data)

        self.time_array = np.arange(0, len(data)) * self.delta_t
        self.ticker_A, self.ticker_B = data.columns[0], data.columns[1]
        self.S = np.log(data.loc[:, self.ticker_B])
        self.spread = np.log(data.loc[:, self.ticker_A]) - self.S

        self.spread = self.spread.to_numpy()  # Converting from pd.Series to numpy array.
        self.S = self.S.to_numpy()  # Converting from pd.Series to numpy array.
        #self._estimate_params() #TODO : V_squared estimator is returning a negative value which is incorrect. Need to check why?

        params = self._estimate_params_log_likelihood()
        print(params)
        self.sigma, self.mu, self.k,self.theta, self.eta, self.rho = params[:-1]


    def _estimate_params(self):
        """
        Closed Form Estimators of params.
        """

        N = len(self.spread) - 1

        m = (self.S[N] - self.S[0]) / N

        S_squared = (np.power(np.diff(self.S), 2).sum() - 2 * m * (self.S[N] - self.S[0]) + N*m*m) / N


        p = (1 / (N * np.power(self.spread[:-1], 2).sum() - self.spread[:-1].sum() ** 2)) * \
            (N * np.multiply(self.spread[1:], self.spread[:-1]).sum() -
             (self.spread[N] - self.spread[0]) * np.sum(self.spread[:-1]) - self.spread[:-1].sum() ** 2)

        q = (self.spread[N] - self.spread[0] + (1 - p) * self.spread[:-1].sum()) / N

        V_squared = (1 / N) * (self.spread[N] ** 2 - self.spread[0] ** 2 + (1 + p ** 2) * np.power(self.spread[:-1], 2).sum()
                               - 2 * p * np.multiply(self.spread[1:], self.spread[:-1]).sum() - N * q)


        C = (1/(N * np.sqrt(V_squared * S_squared))) * (np.multiply(self.spread[1:], np.diff(self.S)).sum()
                                                      - p * np.multiply(self.spread[:-1], np.diff(self.S)).sum()
                                                      - m * (self.spread[N] - self.spread[0])
                                                      - m * (1 - p) * self.spread[:-1].sum())

        self.sigma = np.sqrt(S_squared / self.delta_t)

        self.mu = (m / self.delta_t) + (0.5 * (self.sigma ** 2))

        self.k = - np.log(p) / self.delta_t

        self.theta = q / (1 - p)

        self.eta = np.sqrt(2 * self.k * V_squared / (1 - p ** 2))

        self.rho = self.k * C * np.sqrt(V_squared * S_squared) / (self.eta * self.sigma * (1 - p))


    def optimal_portfolio_weights(self, data: pd.DataFrame, gamma = -100):

        # Preprocessing
        data = self._data_preprocessing(data)

        self.gamma = gamma
        t = np.arange(0, len(data)) * self.delta_t
        tau = t[-1] - t
        x = np.log(data.iloc[:, 0]) - np.log(data.iloc[:, 1])
        x = x.to_numpy()  # Converting from pd.Series to numpy array.

        alpha_t, beta_t = self._alpha_beta_calc(tau)

        h = (1 / (1 - self.gamma)) * (beta_t + 2 * np.multiply(x, alpha_t)
                                      - self.k * (x - self.theta) / self.eta ** 2
                                      + self.rho * self.theta / self.eta + 0.5)

        return h


    def _alpha_beta_calc(self, tau):

        sqrt_gamma = np.sqrt(1 - self.gamma)
        exp_calc = np.exp(2 * self.k * tau / sqrt_gamma)

        alpha_t = self._alpha_calc(sqrt_gamma, exp_calc)
        beta_t = self._beta_calc(sqrt_gamma, exp_calc)

        return alpha_t, beta_t


    def _alpha_calc(self, sqrt_gamma, exp_calc):

        left_calc = self.k * (1 - sqrt_gamma) / (2 * (self.eta ** 2))

        right_calc = 2 * sqrt_gamma / (1 - sqrt_gamma - (1 + sqrt_gamma) * exp_calc)

        return left_calc * (1 + right_calc)


    def _beta_calc(self, sqrt_gamma, exp_calc):

        left_calc = 1/(2 * (self.eta ** 2) * ((1 - sqrt_gamma) - (1 + sqrt_gamma)*exp_calc))

        right_calc = self.gamma * sqrt_gamma * (self.eta ** 2 + 2*self.rho*self.sigma*self.eta) * ((1 - exp_calc) ** 2) - \
                     self.gamma * (self.eta ** 2 + 2*self.rho*self.sigma*self.eta + 2*self.k*self.theta) * (1 - exp_calc)

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
            except FloatingPointError as r:
                return 0


        log_likelihood = np.log(f_y).sum()

        return -log_likelihood
