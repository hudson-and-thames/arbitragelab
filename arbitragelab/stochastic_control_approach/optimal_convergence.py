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

class OptimalConvergence:
    def __init__(self):

        self.gamma = None # gamma should be positive
        self.lambda_1 = None
        self.lambda_2 = None
        self.b_squared = None
        self.sigma_squared = None
        self.delta_t = 1 / 252


    def unconstrained_portfolio_weights_continuous(self, prices_1, prices_2, mu_m, sigma_m, beta):
        """
        Implementation of Proposition 1.

        :param prices_1:
        :param prices_2:
        :param mu_m:
        :param sigma_m:
        :param beta:
        :return:
        """

        t = np.arange(0, len(prices_1)) * self.delta_t
        tau = t[-1] - t  # Stores time remaining till closure (In years)

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
              * (matrix @ C_matrix) * np.log(prices_1 / prices_2)

        phi_1 = phi[0, :]
        phi_2 = phi[1, :]

        phi_m = (mu_m / (self.gamma * sigma_m ** 2)) - (phi_1 + phi_2) * beta

        return phi_1, phi_2, phi_m


    def delta_neutral_portfolio_weights_continuous(self, prices_1, prices_2, mu_m, sigma_m):
        """
        Implementation of Proposition 2.

        :param prices_1:
        :param prices_2:
        :param mu_m:
        :param sigma_m:
        :param beta:
        :return:
        """

        t = np.arange(0, len(prices_1)) * self.delta_t
        tau = t[-1] - t  # Stores time remaining till closure (In years)

        D_t = self._D_calc(tau)

        log_term = np.log(prices_1 / prices_2)

        phi_1 = (-(self.lambda_1 + self.lambda_2) * log_term + 2 * self.b_squared * D_t * log_term)\
                / (2 * self.gamma * self.b_squared)

        phi_2 = -phi_1

        phi_m = mu_m / (self.gamma * sigma_m ** 2)

        return phi_1, phi_2, phi_m


    def _C_calc(self, tau):
        """
        Implementation of Appendix A.1.
        :param tau:
        :return:
        """

        lambda_x = self.lambda_1 + self.lambda_2 # This should be always positive

        inner_term = ((self.lambda_1 ** 2 + self.lambda_2 ** 2) * (self.sigma_squared + self.b_squared)
                      + 2 * self.lambda_1 * self.lambda_2 * self.sigma_squared) / (self.b_squared + 2 * self.sigma_squared)

        sqrt_term = np.sqrt(lambda_x ** 2 - 2 * inner_term * (1 - self.gamma))

        C_plus = (lambda_x + sqrt_term) / (2 * self.b_squared)
        C_minus = (lambda_x - sqrt_term) / (2 * self.b_squared)


        exp_term = np.exp((2 * self.b_squared / self.gamma) * (C_plus - C_minus) * tau)

        C = C_minus * (exp_term - 1) / (exp_term - (C_minus / C_plus))

        return C


    def _D_calc(self, tau):
        """
        Implementation of Appendix A.2.
        :param tau:
        :return:
        """

        lambda_x = self.lambda_1 + self.lambda_2 # This should be always positive
        sqrt_term = np.sqrt(self.gamma)

        D_plus = (lambda_x / (2 * self.b_squared)) * (1 + sqrt_term)
        D_minus = (lambda_x / (2 * self.b_squared)) * (1 - sqrt_term)

        exp_term = np.exp(2 * lambda_x * tau / sqrt_term)

        D = (1 - exp_term) / ((1 / D_plus) - (exp_term / D_minus))

        return D
