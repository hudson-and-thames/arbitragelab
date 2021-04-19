# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module is a realization of the methodology in the following paper:
`Liu, J. and Timmermann, A., 2013. Optimal convergence trade strategies. The Review of Financial Studies, 26(4), pp.1048-1086.
<https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.905.236&rep=rep1&type=pdf>`__
"""
# pylint: disable=invalid-name

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arbitragelab.cointegration_approach import JohansenPortfolio


class OptimalConvergence:
    """
    Implementation of Optimal Convergence Trade Strategies.
    """

    def __init__(self):
        """
        Initializes the parameters of the module.
        """

        # Estimated from error-correction model
        self.ticker_A = None
        self.ticker_B = None
        self.delta_t = 1 / 252
        self.lambda_1 = None
        self.lambda_2 = None

        # Moment based estimates
        self.b_squared = None
        self.sigma_squared = None
        self.beta = None

        # Parameters inputted by user
        self.gamma = None  # gamma should be positive
        self.r = None
        self.mu_m = None
        self.sigma_m = None


    def fit(self, prices: pd.DataFrame, delta_t: float = 1 / 252):
        """

        :param prices: (pd.DataFrame) Contains price series of both stocks in spread.
        :param delta_t: (float) Time difference between each index of data, calculated in years.
        """

        #TODO: Need to input market index data and orthogonalize the price data with market index before using cointegration.
        # Refer Table 1 in paper.

        # Setting instance attributes
        self.delta_t = delta_t
        self.ticker_A, self.ticker_B = prices.columns[0], prices.columns[1]

        prices = self._data_preprocessing(prices)

        # As mentioned in the paper, the vector of linear weights are calculated using co-integrating regression
        j_portfolio = JohansenPortfolio()
        j_portfolio.fit(np.log(prices), det_order=0)  # Fitting the log prices
        j_cointegration_vectors = j_portfolio.cointegration_vectors  # Stores the calculated weights for the pair of stocks in the spread

        self.lambda_1, self.lambda_2 = j_cointegration_vectors.values[0]
        # TODO : Is using Johanssen test correct here for estimating lambda's.

        # TODO: Construct "moment based estimates" for b, sigma and beta.


    def unconstrained_portfolio_weights_continuous(self, prices_1, prices_2, mu_m, sigma_m, gamma, r):
        """
        Implementation of Proposition 1.

        :param r:
        :param gamma:
        :param prices_1:
        :param prices_2:
        :param mu_m:
        :param sigma_m:
        :return:
        """

        if gamma <= 0:
            raise Exception("The value of gamma should be positive.")

        self.mu_m = mu_m
        self.sigma_m = sigma_m
        self.gamma = gamma
        self.r = r

        x, tau = self._x_tau_calc(prices_1, prices_2)

        C_t = self._C_calc(tau)

        matrix = np.zeros((2,2))
        matrix[0, 0] = self.sigma_squared + self.b_squared
        matrix[0, 1] = - self.sigma_squared
        matrix[1, 0] = matrix[0, 1]
        matrix[1, 1] = matrix[0, 0]

        C_matrix = np.zeros((2, len(tau)))
        C_matrix[0, :] = - self.lambda_1 + self.b_squared * C_t
        C_matrix[1, :] =   self.lambda_2 - self.b_squared * C_t

        phi = (1 / (self.gamma * (2 * self.sigma_squared + self.b_squared) * self.b_squared)) \
              * (matrix @ C_matrix) * x

        phi_1 = phi[0, :]
        phi_2 = phi[1, :]

        phi_m = (self.mu_m / (self.gamma * self.sigma_m ** 2)) - (phi_1 + phi_2) * self.beta

        return phi_1, phi_2, phi_m


    def delta_neutral_portfolio_weights_continuous(self, prices_1, prices_2, mu_m, sigma_m, gamma, r):
        """
        Implementation of Proposition 2.

        :param r:
        :param gamma:
        :param prices_1:
        :param prices_2:
        :param mu_m:
        :param sigma_m:
        :return:
        """

        if gamma <= 0:
            raise Exception("The value of gamma should be positive.")

        self.mu_m = mu_m
        self.sigma_m = sigma_m
        self.gamma = gamma
        self.r = r

        x, tau = self._x_tau_calc(prices_1, prices_2)

        D_t = self._D_calc(tau)

        phi_1 = (-(self.lambda_1 + self.lambda_2) * x + 2 * self.b_squared * D_t * x) / (2 * self.gamma * self.b_squared)

        phi_2 = -phi_1

        phi_m = self.mu_m / (self.gamma * self.sigma_m ** 2)

        return phi_1, phi_2, phi_m


    def wealth_gain_continuous(self, prices_1, prices_2, mu_m, sigma_m, gamma, r):
        """
        Implementation of Proposition 4.

        :param r:
        :param gamma:
        :param prices_1:
        :param prices_2:
        :param mu_m:
        :param sigma_m:
        :return:
        """

        if gamma <= 0:
            raise Exception("The value of gamma should be positive.")

        self.mu_m = mu_m
        self.sigma_m = sigma_m
        self.gamma = gamma
        self.r = r

        x, tau = self._x_tau_calc(prices_1, prices_2)

        u_x_t = self._u_func_continuous_calc(x, tau)
        v_x_t = self._v_func_continuous_calc(x, tau)

        R = np.exp((u_x_t - v_x_t) / (1 - self.gamma))

        return R


    def wealth_process(self):
        # TODO : To construct the final wealth from portfolio weights, we would need the market index data.
        #  Refer Section 2 in paper.
        pass


    def _x_tau_calc(self, prices_1, prices_2):
        """
        Calculates the error correction term x given in equation (4) and the time remaining in years.
        :param prices_1:
        :param prices_2:
        :return:
        """

        t = np.arange(0, len(prices_1)) * self.delta_t
        tau = t[-1] - t  # Stores time remaining till closure (In years)

        x = np.log(prices_1 / prices_2)

        return x, tau


    def _lambda_x_calc(self):
        """
        Helper function calculates lambda_x.
        :return:
        """

        lambda_x = self.lambda_1 + self.lambda_2  # This should be always positive
        return lambda_x


    def _xi_calc(self):
        """
        Helper function which calculates xi, present in Appendix A.1.
        Xi is used in the calculations of A and C functions.
        :return:
        """

        lambda_x = self._lambda_x_calc()

        inner_term = ((self.lambda_1 ** 2 + self.lambda_2 ** 2) * (self.sigma_squared + self.b_squared)
                      + 2 * self.lambda_1 * self.lambda_2 * self.sigma_squared) / (
                                 self.b_squared + 2 * self.sigma_squared)

        xi = np.sqrt(lambda_x ** 2 - 2 * inner_term * (1 - self.gamma))

        return xi, lambda_x


    def _C_calc(self, tau):
        """
        Implementation of function C given in Appendix A.1.
        :param tau:
        :return:
        """

        xi, lambda_x = self._xi_calc()

        C_plus = (lambda_x + xi) / (2 * self.b_squared)
        C_minus = (lambda_x - xi) / (2 * self.b_squared)

        exp_term = np.exp((2 * self.b_squared / self.gamma) * (C_plus - C_minus) * tau)

        C = C_minus * (exp_term - 1) / (exp_term - (C_minus / C_plus))

        return C


    def _D_calc(self, tau):
        """
        Implementation of function D given in Appendix A.2.
        :param tau:
        :return:
        """

        lambda_x = self._lambda_x_calc()
        sqrt_term = np.sqrt(self.gamma)

        D_plus = (lambda_x / (2 * self.b_squared)) * (1 + sqrt_term)
        D_minus = (lambda_x / (2 * self.b_squared)) * (1 - sqrt_term)

        exp_term = np.exp(2 * lambda_x * tau / sqrt_term)

        D = (1 - exp_term) / ((1 / D_plus) - (exp_term / D_minus))

        return D


    def _A_calc(self, tau):
        """
        Implementation of function A given in Appendix A.1.
        :param tau:
        :return:
        """

        xi, lambda_x = self._xi_calc()

        A = self._A_B_helper(lambda_x, tau, xi)

        return A


    def _B_calc(self, tau):
        """
        Implementation of function B given in Appendix A.2.
        :param tau:
        :return:
        """

        lambda_x = self._lambda_x_calc()
        eta = lambda_x * np.sqrt(self.gamma)

        B = self._A_B_helper(lambda_x, tau, eta)

        return B


    def _A_B_helper(self, lambda_x, tau, rep_term):
        """
        Helper function implements the common formulae present in A and B function calculations.

        :param lambda_x:
        :param tau:
        :param rep_term:
        :return:
        """

        inner_exp_term = (rep_term / self.gamma) * tau
        exp_term_1 = np.exp(inner_exp_term)
        exp_term_2 = np.exp(-inner_exp_term)

        first_term = self.r + (1 / (2 * self.gamma)) * (self.mu_m ** 2 / self.sigma_m ** 2)
        log_term = np.log((lambda_x / 2) * ((exp_term_1 - exp_term_2) / rep_term) + 0.5 * (exp_term_1 + exp_term_2))

        result = first_term * (1 - self.gamma) * tau + (lambda_x / 2) * tau - (self.gamma / 2) * log_term

        return result


    def _u_func_continuous_calc(self, x, tau):
        """
        Implementation of Lemma 1.
        :param x:
        :param tau:
        :return:
        """

        C_t = self._C_calc(tau)
        A_t = self._A_calc(tau)

        u = A_t + 0.5 * C_t * np.power(x, 2)

        return u


    def _v_func_continuous_calc(self, x, tau):
        """
        Implementation of Lemma 2.
        :param x:
        :param tau:
        :return:
        """

        D_t = self._D_calc(tau)
        B_t = self._B_calc(tau)

        v = B_t + 0.5 * D_t * np.power(x, 2)

        return v


    @staticmethod
    def _data_preprocessing(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function for input data preprocessing.

        :param prices: (pd.DataFrame) Pricing data of both stocks in spread.
        :return: (pd.DataFrame) Processed dataframe.
        """

        return prices.ffill()
