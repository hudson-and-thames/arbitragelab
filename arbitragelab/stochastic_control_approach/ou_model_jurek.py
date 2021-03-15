# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module implements the optimal dynamic portfolio strategy for arbitrageurs with a finite horizon and
non-myopic preferences facing a mean-reverting arbitrage opportunity using OU process.

This module is a realization of the methodology in the following paper:
`Jurek, J.W. and Yang, H., 2007, April. Dynamic portfolio selection in arbitrage. In EFA 2006 Meetings Paper.
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=882536>`__
"""

import numpy as np
import pandas as pd

class StochasticControlJurek:
    def __init__(self):

        #Training Data.
        self.ticker_A = None
        self.ticker_B = None
        self.spread = None
        self.time_array = None

        self.delta_t = 1 / 251   #TODO : Is this value correct for this paper?
        # Estimated params from training data.
        self.sigma = None
        self.mu = None
        self.k = None

        # Params inputted by user.
        self.r = None
        self.gamma = None


    def fit(self, data: pd.DataFrame):

        self.time_array = np.arange(0, len(data)) / self.delta_t
        self.ticker_A, self.ticker_B = data.columns[0], data.columns[1]

        returns_df = data.pct_change()
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().dropna()

        total_return_indices = data.copy()
        total_return_indices.iloc[0, :] = 1
        total_return_indices.iloc[1:, :] = pd.DataFrame.cumprod(1 + returns_df, axis=0)

        # df = data.iloc[:round(len(data) * 0.1), :].mean(axis=0)
        # mispricing_percent = df[self.ticker_B] / df[self.ticker_A]
        # print(mispricing_percent)
        # total_return_indices[self.ticker_A]  *= mispricing_percent

        #TODO : Need to figure out weights for return indices for spread calcuation

        self.spread = total_return_indices[self.ticker_A] - total_return_indices[self.ticker_B]

        self._estimate_params()


    def _estimate_params(self):

        N = len(self.spread)

        self.mu = self.spread.mean()

        self.k = (-1 / self.delta_t) * np.log(np.multiply(self.spread[1:] - self.mu, self.spread[:-1] - self.mu).sum()
                                              / np.power(self.spread[1:] - self.mu, 2).sum())

        sigma_calc_sum = np.power(self.spread[1:] - self.mu - np.exp(-self.k*self.delta_t)*(self.spread[:-1] - self.mu)
                                  / np.exp(-self.k*self.delta_t), 2).sum()
        #TODO : Check for error in paper for this summation?

        self.sigma = np.sqrt(2*self.k*sigma_calc_sum/((np.exp(2*self.k*self.delta_t) - 1)*(N - 2)))


    def optimal_portfolio_weights(self, data: pd.DataFrame, utility_type = 1, r = 0.05, gamma = 1):
        """
        utility_type = 1 implies agent with utility of terminal wealth.
                        gamma = 1 implies log utility investor.
                        gamma != 1 implies general CRRA investor.

        utility_type = 2 implies agent whose utility is defined over intermediate consumption
         with agent’s preferences described by Epstein-Zin recursive utility with phi = 1.
                    gamma = 1 reduces to standard log utility.
                    gamma ! = 1
        """

        self.r = r
        self.gamma = gamma
        t = np.arange(0, len(data)) / self.delta_t
        tau = t[:-1] - t

        W = np.zeros(len(t)) #TODO : Need to figure out how to calculate W
        S = np.zeros(len(t)) #TODO : Calculate this spread after the spread calc in fit is finalized.

        N = None
        # The optimal weights equation is the same for both types of utility functions.
        # For gamma = 1, the outputs weights are identical for both types of utility functions,
        # whereas for gamma != 1, the calculation of A and B functions are different.
        if self.gamma == 1:
            N = ((self.k * (self.mu - S) - self.r * S) / (self.sigma ** 2)) * W

        else:
            # For the case of gamma = 1, A & B are not used in the final optimal portfolio calculation,
            # so the corresponding calculations are skipped.

            A = None
            B = None
            if utility_type == 1:
                A, B = self._AB_calc_1(tau)

            elif utility_type == 2:
                A, B = self._AB_calc_2(tau)

            N = ((self.k * (self.mu - S) - self.r * S) / (self.sigma ** 2) + (2 * A * S + B) / self.gamma) * W

        return N


    def _AB_calc_1(self, tau):
        """
        For utility_type = 1.
        Follows Appendix A.2 in the paper.
        """

        c_1 = 2 * self.sigma ** 2 / self.gamma
        c_2 = -(self.k / self.gamma + self.r * (1 - self.gamma) / self.gamma)
        c_3 = 0.5 * ((1 - self.gamma) / self.gamma) * (((self.k + self.r) / self.sigma) ** 2)
        det = 4 * (self.k ** 2 - self.r ** 2 * (1 - self.gamma)) / self.gamma
        gamma_0 = 1 - (self.k / self.r) ** 2

        A = self._A_calc_1(tau, c_1, c_2, c_3, det, gamma_0)
        B = self._B_calc_1(tau, c_1, c_2, c_3, det, gamma_0)
        return A, B


    def _A_calc_1(self, tau, c_1, c_2, c_3, det, gamma_0):

        A = None
        if 0 < self.gamma < gamma_0:
            A = -c_2 / c_1 + (np.sqrt(-det) / (2 * c_1)) * np.tan(np.sqrt(-det) * tau / 2 + np.arctan(2 * c_2 / np.sqrt(-det)))

        elif self.gamma == gamma_0:
            A = -(c_2 / c_1) * (1 + 1 / (c_2 * tau - 1))

        elif gamma_0 < self.gamma < 1:
            # coth = 1/tanh and arccoth(x) = arctanh(1/x)
            # For reference : https://www.efunda.com/math/hyperbolic/hyperbolic.cfm

            A = -c_2 / c_1 + (np.sqrt(det) / (2 * c_1)) * (1 / np.tanh(-np.sqrt(det) * tau / 2 + np.arctanh(np.sqrt(det) / (2 * c_2))))

        elif self.gamma > 1:
            A = -c_2 / c_1 + (np.sqrt(det) / (2 * c_1)) * np.tanh(-np.sqrt(det) * tau / 2 + np.arctanh(2 * c_2 / np.sqrt(det)))

        return A


    def _B_calc_1(self, tau, c_1, c_2, c_3, det, gamma_0):

        #Note : When self.mu = 0, c_4 and c_5 become zero and consequently B becomes zero.

        c_4 = 2 * self.k * self.mu / self.gamma
        c_5 = -((self.k + self.r) / self.sigma ** 2) * ((1 - self.gamma) / self.gamma) * self.k * self.mu

        B = None
        if 0 < self.gamma < gamma_0:
            phi_1 = np.sqrt(-det) * (np.cos(np.sqrt(-det) * tau / 2) - 1) + 2 * c_2 * np.sin(np.sqrt(-det) * tau / 2)
            phi_2 = np.arctanh(np.tan(0.25 * (np.sqrt(-det) * tau - 2 * np.arctan(2 * c_2 / np.sqrt(-det))))) + \
                    np.arctanh(np.tan(0.5 * np.arctan(2 * c_2 / np.sqrt(-det))))

            B = c_4 * phi_1 / (c_1 * np.sqrt(-det)) + (4 * phi_2 / np.sqrt(-det)) * (c_5 - c_4 / c_1) * \
                np.cos(np.sqrt(-det) * tau / 2 - np.arctan(c_2 / np.sqrt(-det)))

        elif self.gamma == gamma_0:
            B = (c_1 * c_5 * (c_2 * tau - 2) - (c_2 ** 2) * c_4) * tau / (2 * c_1 * (c_2 * tau - 1))

        elif self.gamma > gamma_0:
            B = (4 * (c_2 * c_5 - c_3 * c_4 + (c_3 * c_4 - c_2 * c_5) * np.cosh(np.sqrt(det) * tau / 2))
                 + 2 * c_5 * np.sqrt(det) * np.sinh(np.sqrt(det) * tau / 2)) \
                / (det * np.cosh(np.sqrt(det) * tau / 2) - 2 * c_2 * np.sqrt(det) * np.sinh(np.sqrt(det) * tau / 2))

        return B


    def _AB_calc_2(self, tau):
        """
        For utility_type = 2.
        Follows appendix B.1 in the paper.
        """

        c_1 = 2 * self.sigma ** 2 / self.gamma
        #TODO : Figure out how to calculate beta and subsequently c_2, c_3, etc.
        c_2 = 0
        c_3 = 0
        det = 0
        gamma_0 = 0

        A = self._A_calc_2(tau, c_1, c_2, c_3, det, gamma_0)
        B = self._B_calc_2(tau)
        return A, B


    def _A_calc_2(self, tau, c_1, c_2, c_3, det, gamma_0):

        # Same calculation as for general CRRA Investor.
        return self._A_calc_1(tau, c_1, c_2, c_3, det, gamma_0)


    def _B_calc_2(self, tau):
        pass
