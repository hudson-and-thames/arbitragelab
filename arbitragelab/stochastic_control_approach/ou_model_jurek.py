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
import matplotlib.pyplot as plt

from arbitragelab.cointegration_approach.johansen import JohansenPortfolio
from arbitragelab.cointegration_approach.engle_granger import EngleGrangerPortfolio

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
        self.beta = None

    def _calc_total_return_indices(self, data):
        returns_df = data.pct_change()
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().dropna()

        total_return_indices = data.copy()
        total_return_indices.iloc[0, :] = 1
        total_return_indices.iloc[1:, :] = pd.DataFrame.cumprod(1 + returns_df, axis=0)

        return total_return_indices



    def fit(self, data: pd.DataFrame):

        self.time_array = np.arange(0, len(data)) * self.delta_t
        self.ticker_A, self.ticker_B = data.columns[0], data.columns[1]

        total_return_indices = self._calc_total_return_indices(data)

        eg_portfolio = EngleGrangerPortfolio()
        eg_portfolio.fit(total_return_indices, add_constant=True)
        eg_adf_statistics = eg_portfolio.adf_statistics
        eg_cointegration_vectors = eg_portfolio.cointegration_vectors

        if eg_adf_statistics.loc['statistic_value', 0] > eg_adf_statistics.loc['95%', 0]:
            print(eg_adf_statistics)
            raise Exception("ADF statistic test failure.")

        self.eg_scaled_vectors = eg_cointegration_vectors.loc[0] / abs(eg_cointegration_vectors.loc[0]).sum()

        self.spread = (total_return_indices * self.eg_scaled_vectors).sum(axis=1)
        self.spread.plot()
        plt.show()

        self._estimate_params()
        print(self.k)
        print(self.sigma)
        print(self.mu)

        # TODO : Test the null hypothesis of no mean reversion using estimated k by running a Monte Carlo bootstrap experiment.
        #  For details look at para 3 in Section IV A of the paper.


    def _estimate_params(self):

        N = len(self.spread)

        self.mu = self.spread.mean()

        self.k = (-1 / self.delta_t) * np.log(np.multiply(self.spread[1:] - self.mu, self.spread[:-1] - self.mu).sum()
                                              / np.power(self.spread[1:] - self.mu, 2).sum())

        sigma_calc_sum = np.power((self.spread[1:] - self.mu - np.exp(-self.k * self.delta_t) * (self.spread[:-1] - self.mu))
                                  / np.exp(-self.k * self.delta_t), 2).sum()

        self.sigma = np.sqrt(2 * self.k * sigma_calc_sum / ((np.exp(2 * self.k * self.delta_t) - 1) * (N - 2)))


    def optimal_portfolio_weights(self, data: pd.DataFrame, beta, utility_type = 1, r = 0.05, gamma = 1):
        """

        Implementation of Theorem 1 and Theorem 2 in Jurek (2007).

        utility_type = 1 implies agent with utility of terminal wealth.
                        gamma = 1 implies log utility investor.
                        gamma != 1 implies general CRRA investor.

        utility_type = 2 implies agent whose utility is defined over intermediate consumption
         with agent’s preferences described by Epstein-Zin recursive utility with psi(elasticity of intertemporal substitution) = 1.
                    gamma = 1 reduces to standard log utility.
                    gamma ! = 1, gamma > 1 implies more risk averse investors, whereas gamma < 1 implies more risk tolerance.

        What is beta?
        1)subjective rate of time preference.

        2)beta signifies the constant fraction of total wealth an investor chooses to consume. This is analogous to a hedge fund investor
        who cares both about terminal wealth in some risk-averse way and consumes a constant fraction, β,
        of assets under management (the management fee).

        For utility_type = 2, C(Consumption) = beta * W
        """

        self.r = r
        self.gamma = gamma
        self.beta = beta
        t = np.arange(0, len(data)) * self.delta_t
        tau = t[-1] - t

        total_return_indices = self._calc_total_return_indices(data)

        W = np.ones(len(t)) # Wealth is normalized to one. #TODO : Is this the correct way?
        S = (total_return_indices * self.eg_scaled_vectors).sum(axis=1) #TODO : Do we use trained linear weights here ?

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

        return N / W # We return the optimal allocation of spread asset scaled by wealth.


    def _AB_calc_1(self, tau):
        """
        For utility_type = 1.
        Follows Appendix A.2 in the paper.
        """

        c_1 = 2 * self.sigma ** 2 / self.gamma

        c_2 = -(self.k / self.gamma + self.r * (1 - self.gamma) / self.gamma)

        c_3 = 0.5 * ((1 - self.gamma) / self.gamma) * (((self.k + self.r) / self.sigma) ** 2)

        disc = 4 * (self.k ** 2 - self.r ** 2 * (1 - self.gamma)) / self.gamma
        #Note : discriminant is always positive for gamma > 1.

        gamma_0 = 1 - (self.k / self.r) ** 2
        #Note : gamma_0 is not always > 0.


        A = self._A_calc_1(tau, c_1, c_2, c_3, disc, gamma_0)
        B = self._B_calc_1(tau, c_1, c_2, c_3, disc, gamma_0)

        return A, B


    def _A_calc_1(self, tau, c_1, c_2, c_3, disc, gamma_0):

        #Note : A<=0 and decreasing in tau for gamma > 1, and vice versa for gamma < 1.

        A = None
        if 0 < self.gamma < gamma_0:
            A = -c_2 / c_1 + (np.sqrt(-disc) / (2 * c_1)) * np.tan(np.sqrt(-disc) * tau / 2 + np.arctan(2 * c_2 / np.sqrt(-disc)))

        elif self.gamma == gamma_0:
            A = -(c_2 / c_1) * (1 + 1 / (c_2 * tau - 1))

        elif gamma_0 < self.gamma < 1:
            # coth = 1/tanh and arccoth(x) = arctanh(1/x)
            # For reference : https://www.efunda.com/math/hyperbolic/hyperbolic.cfm

            A = -c_2 / c_1 + (np.sqrt(disc) / (2 * c_1)) * (1 / np.tanh(-np.sqrt(disc) * tau / 2 + np.arctanh(np.sqrt(disc) / (2 * c_2))))

        elif self.gamma > 1:
            A = -c_2 / c_1 + (np.sqrt(disc) / (2 * c_1)) * np.tanh(-np.sqrt(disc) * tau / 2 + np.arctanh(2 * c_2 / np.sqrt(disc)))

        return A


    def _B_calc_1(self, tau, c_1, c_2, c_3, disc, gamma_0):

        #Note : When self.mu = 0, c_4 and c_5 become zero and consequently B becomes zero.

        c_4 = 2 * self.k * self.mu / self.gamma

        c_5 = -((self.k + self.r) / self.sigma ** 2) * ((1 - self.gamma) / self.gamma) * self.k * self.mu

        B = None
        if 0 < self.gamma < gamma_0:
            phi_1 = np.sqrt(-disc) * (np.cos(np.sqrt(-disc) * tau / 2) - 1) + 2 * c_2 * np.sin(np.sqrt(-disc) * tau / 2)

            phi_2 = np.arctanh(np.tan(0.25 * (np.sqrt(-disc) * tau - 2 * np.arctan(2 * c_2 / np.sqrt(-disc))))) + \
                    np.arctanh(np.tan(0.5 * np.arctan(2 * c_2 / np.sqrt(-disc))))

            B = c_4 * phi_1 / (c_1 * np.sqrt(-disc)) + (4 * phi_2 / np.sqrt(-disc)) * (c_5 - c_4 / c_1) * \
                np.cos(np.sqrt(-disc) * tau / 2 - np.arctan(c_2 / np.sqrt(-disc)))

        elif self.gamma == gamma_0:
            B = (c_1 * c_5 * (c_2 * tau - 2) - (c_2 ** 2) * c_4) * tau / (2 * c_1 * (c_2 * tau - 1))

        elif self.gamma > gamma_0:
            B = (4 * (c_2 * c_5 - c_3 * c_4 + (c_3 * c_4 - c_2 * c_5) * np.cosh(np.sqrt(disc) * tau / 2))
                 + 2 * c_5 * np.sqrt(disc) * np.sinh(np.sqrt(disc) * tau / 2)) \
                / (disc * np.cosh(np.sqrt(disc) * tau / 2) - 2 * c_2 * np.sqrt(disc) * np.sinh(np.sqrt(disc) * tau / 2))

        return B


    def _AB_calc_2(self, tau):
        """
        For utility_type = 2.
        Follows appendix B.1 in the paper.
        """

        c_1 = 2 * self.sigma ** 2 / self.gamma

        c_2 = (self.gamma * (2 * self.r - self.beta) - 2 * (self.k + self.r)) / (2 * self.gamma)

        c_3 = ((self.k + self.r) ** 2) * (1 - self.gamma) / (2 * self.gamma * (self.sigma ** 2))

        disc = ((2 * self.k + self.beta) ** 2 + (self.gamma - 1) * ((-2 * self.r + self.beta) ** 2)) / self.gamma

        gamma_0 = 4 * (self.k + self.r) * (self.r - self.beta - self.k) / ((2 * self.r - self.beta) ** 2)


        A = self._A_calc_2(tau, c_1, c_2, c_3, disc, gamma_0)
        B = self._B_calc_2(tau, c_1, c_2, c_3, disc, gamma_0)

        return A, B


    def _A_calc_2(self, tau, c_1, c_2, c_3, disc, gamma_0):

        # Same calculation as for general CRRA Investor.
        return self._A_calc_1(tau, c_1, c_2, c_3, disc, gamma_0)


    def _B_calc_2(self, tau, c_1, c_2, c_3, disc, gamma_0):

        # Note : When self.mu = 0, c_4 and c_6 become zero and consequently B becomes zero.

        c_4 = 2 * self.k * self.mu / self.gamma

        c_5 = -(self.k + self.r * (1 - self.gamma) + self.beta * self.gamma) / (2 * self.gamma)

        c_6 = self.k * (self.k + self.r) * (self.gamma - 1) * self.mu / (self.gamma * self.sigma ** 2)

        B = None

        rep_exp_1 = np.exp(tau * c_2)
        rep_exp_2 = np.exp(tau * c_5)

        if 0 < self.gamma < gamma_0:

            rep_phrase_1 = np.sqrt(c_1 * c_3 - c_2 ** 2)
            rep_phrase_2 = np.sqrt(c_1 * c_3) / rep_phrase_1
            rep_phrase_3 = rep_phrase_1 * tau + np.arctan(c_2 / rep_phrase_1)

            denominator = c_1 * rep_phrase_2 * (c_1 * c_3 + c_5 * (c_5 - 2 * c_2))

            term_1 = rep_exp_1 * rep_phrase_2 * c_4 * c_5 * (c_2 - rep_phrase_1 * np.tan(rep_phrase_3))

            term_2 = rep_exp_1 * rep_phrase_2 - 2 * rep_exp_2 * (1 / np.cos(rep_phrase_3))

            term_3 = rep_exp_2 * (1 / np.cos(rep_phrase_3)) - rep_exp_1 * rep_phrase_2

            term_4 = rep_exp_1 * rep_phrase_2 * rep_phrase_1 * np.tan(rep_phrase_3)

            term_5 = -term_3


            B = np.exp(-tau * c_2) * (term_1
                                      + c_1 * (c_6 * (c_2 * term_2
                                                      + term_3 * c_5
                                                      + term_4)
                                               -c_3 * term_5 * c_4)) / denominator


        elif self.gamma == gamma_0:

            denominator = c_1 * rep_exp_1 * (tau * c_2 - 1) * (c_2 - c_5) ** 2

            term_1 = rep_exp_1 * tau * c_4 * c_2 ** 3

            term_2 = (c_4 * (rep_exp_1 * (tau * c_5 + 1) - rep_exp_2) + rep_exp_1 * tau * c_1 * c_6) * c_2 ** 2

            term_3 = c_1 * (rep_exp_1 * tau * c_5 + 2 * (rep_exp_1 - rep_exp_2)) * c_6 * c_2

            term_4 = (rep_exp_1 - rep_exp_2) * c_1 * c_5 * c_6


            B = (-term_1 + term_2 - term_3
                 + term_4) / denominator

        elif gamma_0 < self.gamma < 1:

            rep_phrase_1 = np.sqrt(c_1 * c_3 / c_2 ** 2)
            rep_phrase_2 = np.sqrt(c_2 ** 2 - c_1 * c_3)
            rep_phrase_3 = np.arctanh(rep_phrase_2 / c_2) - tau * rep_phrase_2
            # arccoth(x) = arctanh(1/x)

            denominator = c_2 * rep_phrase_1 * (c_1 * c_3 + c_5 * (c_5 - 2 * c_2))

            term_1 = rep_exp_1 * rep_phrase_1 * c_6 * c_2 ** 2

            term_2 = rep_exp_1 * c_3 * rep_phrase_1 * c_4

            term_3 = 2 * rep_exp_2 * (1 / np.sinh(rep_phrase_3)) - rep_phrase_2 * (1 / np.tanh(rep_phrase_3)) * rep_phrase_1
            # csch = 1/sinh, coth = 1/tanh

            term_4 = rep_exp_1 * rep_phrase_1 * c_5

            term_5 = (1 / np.sinh(rep_phrase_3)) * (c_3 * c_4 * (rep_exp_2 * rep_phrase_2 - rep_exp_1 * np.sinh(tau * rep_phrase_2) * c_5)
                                          + rep_exp_2 * rep_phrase_2 * c_5 * c_6)
            # csch = 1/sinh


            B = np.exp(-tau * c_2) * (term_1 - (term_2 + (rep_phrase_2 * term_3 + term_4) * c_6) * c_2
                                      + term_5) / denominator

        elif self.gamma > 1:

            rep_phrase_1 = np.sqrt(-c_1 * c_3 / disc) # Repeated Phrase 1
            rep_phrase_2 = 0.5 * np.sqrt(disc) * tau - np.arctanh(2 * c_2 / np.sqrt(disc)) # Repeated Phrase 2

            denominator = 2 * c_1 * rep_phrase_1 * (c_1 * c_3 + c_5 * (c_5 - 2 * c_2)) # Denominator

            term_1 = 2 * rep_exp_1 * rep_phrase_1 * c_4 * c_5 * (c_2 + 0.5 * np.sqrt(disc) * np.tanh(rep_phrase_2))

            term_2 = 2 * rep_exp_1 * rep_phrase_1 - 2 * rep_exp_2 * (1 / np.cosh(rep_phrase_2))
            # sech = 1 / cosh

            term_3 = rep_exp_2 * (1 / np.cosh(rep_phrase_2)) - 2 * rep_exp_1 * rep_phrase_1

            term_4 = rep_exp_1 * np.sqrt(disc) * rep_phrase_1 * np.tanh(rep_phrase_2)

            term_5 = -term_3


            B = np.exp(-tau * c_2) * (term_1
                                      + c_1 * ( c_6 * (c_2 * term_2
                                                       + term_3 * c_5
                                                       - term_4)
                                                - c_3 * term_5 * c_4)) / denominator


        return B
