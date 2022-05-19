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
# pylint: disable=invalid-name, too-many-instance-attributes, too-many-locals

import warnings
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arbitragelab.cointegration_approach.engle_granger import EngleGrangerPortfolio
from arbitragelab.util import segment


class OUModelJurek:
    """
    This class derives the optimal dynamic strategy for arbitrageurs with a finite horizon
    and non-myopic preferences facing a mean-reverting arbitrage opportunity (e.g. an equity pairs trade).

    To capture the presence of horizon and divergence risk, we model the dynamics of the mispricing
    using a mean-reverting OU process. Under this process, although the mispricing is guaranteed
    to be eliminated at some future date, the timing of convergence, as well as the maximum magnitude
    of the mispricing prior to convergence, are uncertain. With this assumption, we are able to derive
    the arbitrageur's optimal dynamic portfolio policy for a set of general, non-myopic preference
    specifications, including CRRA utility defined over wealth at a finite horizon and Epstein-Zin
    utility defined over intermediate cash flows (e.g. fees).
    """

    def __init__(self):
        """
        Initializes the parameters of the module.
        """

        # Characteristics of Training Data
        self.ticker_A = None  # Ticker Symbol of first stock
        self.ticker_B = None  # Ticker Symbol of second stock
        self.spread = None  # Constructed spread from training data
        self.scaled_spread_weights = None  # Scaled weights for the prices in spread
        self.time_array = None  # Time indices of training data
        self.delta_t = None  # Time difference between each index in data, calculated in years

        # Estimated params from training data
        self.sigma = None  # Standard deviation of spread
        self.mu = None  # Long run mean of spread
        self.k = None  # Rate of mean reversion

        # Params inputted by user
        self.r = None  # Rate of returns
        self.gamma = None  # Coefficient of relative risk aversion
        self.beta = None  # Rate of time preference

        segment.track('OUModelJurek')


    @staticmethod
    def _calc_total_return_indices(prices: pd.DataFrame) -> pd.DataFrame:
        """
        This method calculates the total return indices from pricing data.
        This calculation follows Section IV A in Jurek (2007).

        :param prices: (pd.DataFrame) Contains price series of both stocks in spread.
        :return: (pd.DataFrame) Total return indices.
        """

        # Calculating the daily returns
        returns_df = prices.pct_change()
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().dropna()

        total_return_indices = prices.copy()
        total_return_indices.iloc[0, :] = 1
        total_return_indices.iloc[1:, :] = pd.DataFrame.cumprod(1 + returns_df, axis=0)

        return total_return_indices


    def fit(self, prices: pd.DataFrame, delta_t: float = 1 / 252, adf_test: bool = False, significance_level: float = 0.95):
        """
        This method uses inputted training data to calculate the spread and
        estimate the parameters of the corresponding OU process.

        The spread construction implementation follows Section IV A in Jurek (2007).

        :param prices: (pd.DataFrame) Contains price series of both stocks in spread.
        :param delta_t: (float) Time difference between each index of data, calculated in years.
        :param adf_test: (bool) Flag which defines whether the adf statistic test should be conducted.
        :param significance_level: (float) This significance level is used in the ADF statistic test.
            Value can be one of the following: (0.90, 0.95, 0.99).
        """

        if len(prices) < (10 / delta_t):
            warnings.warn("Please make sure length of training data is greater than 10 years. "
                          "This is the time period used to fit the model in the original paper.")

        # Setting instance attributes
        self.delta_t = delta_t
        self.ticker_A, self.ticker_B = prices.columns[0], prices.columns[1]

        # Calculating the total return indices from pricing data
        total_return_indices = self._calc_total_return_indices(prices)
        self.time_array = np.arange(0, len(total_return_indices)) * self.delta_t

        # As mentioned in the paper, the vector of linear weights are calculated using co-integrating regression
        eg_portfolio = EngleGrangerPortfolio()
        eg_portfolio.fit(total_return_indices, add_constant=True)  # Fitting the total return indices
        eg_adf_statistics = eg_portfolio.adf_statistics  # Stores the results of the ADF statistic test
        eg_cointegration_vectors = eg_portfolio.cointegration_vectors  # Stores the calculated weights for the pair of stocks in the spread

        if adf_test is True and eg_adf_statistics.loc['statistic_value', 0] > eg_adf_statistics.loc[f'{int(significance_level * 100)}%', 0]:
            # Making sure that the data passes the ADF statistic test
            print(eg_adf_statistics)
            warnings.warn("ADF statistic test failure.")

        # Scaling the weights such that they sum to 1
        self.scaled_spread_weights = eg_cointegration_vectors.loc[0] / abs(eg_cointegration_vectors.loc[0]).sum()

        # Calculating the final spread value from the scaled weights
        self.spread = (total_return_indices * self.scaled_spread_weights).sum(axis=1)

        self.spread = self.spread.to_numpy()  # This conversion seems to have changed the estimated values

        params = self._estimate_params(self.spread)
        self.mu, self.k, self.sigma = params


    def _estimate_params(self, spread: np.array) -> tuple:
        """
        This method implements the closed form solutions for estimators of the model parameters.
        These formulas for the estimators are given in Appendix E of Jurek (2007).

        :param spread: (np.array) Price series of the constructed spread.
        :return: (tuple) Consists of estimated params.
        """

        N = len(spread)

        # Mean estimator
        mu = spread.mean()

        # Estimator for rate of mean reversion
        k = (-1 / self.delta_t) * np.log(np.multiply(spread[1:] - mu, spread[:-1] - mu).sum()
                                              / np.power(spread[1:] - mu, 2).sum())

        # Part of sigma estimation formula
        sigma_calc_sum = np.power((spread[1:] - mu - np.exp(-k * self.delta_t) * (spread[:-1] - mu))
                                  / np.exp(-k * self.delta_t), 2).sum()

        # Estimator for standard deviation
        sigma = np.sqrt(2 * k * sigma_calc_sum / ((np.exp(2 * k * self.delta_t) - 1) * (N - 2)))

        return mu, k, sigma


    def spread_calc(self, prices: pd.DataFrame) -> tuple:
        """
        This method calculates the spread on test data using the scaled weights from training data.

        :param prices: (pd.DataFrame) Contains price series of both stocks in spread.
        :return: (tuple) Consists of time remaining array and spread numpy array.
        """

        # Calculating the total return indices from pricing data
        total_return_indices = self._calc_total_return_indices(prices)
        t = np.arange(0, len(total_return_indices)) * self.delta_t
        tau = t[-1] - t  # Stores time remaining till closure (In years)

        # Calculating the spread with weights calculated from training data
        S = (total_return_indices * self.scaled_spread_weights).sum(axis=1)

        S = S.to_numpy()

        return tau, S


    def optimal_portfolio_weights(self, prices: pd.DataFrame, utility_type: int = 1, gamma: float = 1,
                                  beta: float = 0.01, r: float = 0.05) -> np.array:
        """
        Implementation of Theorem 1 and Theorem 2 in Jurek (2007).

        This method implements the optimal portfolio strategy for two types of investors with varying utility functions.

        The first type of investor is represented by utility_type = 1.
        This agent has constant relative risk aversion preferences(CRRA investor) with utility defined over terminal wealth.
        For this type of investor,

        gamma = 1 implies log utility investor, and
        gamma != 1 implies general CRRA investor.

        The second type of investor is represented by utility_type = 2.
        This agent has utility defined over intermediate consumption,
        with agent’s preferences described by Epstein-Zin recursive utility with psi(elasticity of intertemporal substitution) = 1.
        For this type of investor,

        gamma = 1 reduces to standard log utility investor, and
        gamma > 1 implies more risk averse investors,
        whereas gamma < 1 implies more risk tolerance in comparison to investor with log utility.

        What is beta?
        Beta signifies the constant fraction of total wealth an investor chooses to consume. This is analogous to a hedge fund investor
        who cares both about terminal wealth in some risk-averse way and consumes a constant fraction, beta,
        of assets under management (the management fee).
        For utility_type = 2, C(Consumption) = beta * W.

        :param prices: (pd.DataFrame) Contains price series of both stocks in spread.
        :param utility_type: (int) Flag signifies type of investor preferences. (Either 1 or 2).
        :param gamma: (float) coefficient of relative risk aversion. (gamma > 0).
        :param beta: (float) Subjective rate of time preference. (Only required for utility_type = 2).
        :param r: (float) Rate of Returns. (r > 0).
        :return: (np.array) Scaled optimal portfolio weights.
        """

        if self.sigma is None:
            raise Exception("Please run the fit method before calling this method.")

        if gamma <= 0:
            raise Exception("The value of gamma should be positive.")

        if utility_type not in [1, 2]:
            raise Exception("Please make sure utility_type is either 1 or 2.")

        # Setting instance attributes
        self.r = r
        self.gamma = gamma
        self.beta = beta

        tau, S = self.spread_calc(prices)

        W = np.ones(len(tau))  # Wealth is normalized to one

        N = None
        # The optimal weights equation is the same for both types of utility functions
        # For gamma = 1, the outputs weights are identical for both types of utility functions,
        # whereas for gamma != 1, the calculation of A and B functions in the equation are different
        if self.gamma == 1:
            N = ((self.k * (self.mu - S) - self.r * S) / (self.sigma ** 2)) * W

        else:
            # For the case of gamma = 1, A & B are not used in the final optimal portfolio calculation,
            # so the corresponding calculations are skipped

            A = None
            B = None
            # A and B functions are used in the final weights calculation
            if utility_type == 1:
                A, B = self._AB_calc_1(tau)

            else:
                A, B = self._AB_calc_2(tau)

            N = ((self.k * (self.mu - S) - self.r * S) / (self.gamma * self.sigma ** 2) + (2 * A * S + B) / self.gamma) * W

        return N / W  # We return the optimal allocation of spread asset scaled by wealth


    def optimal_portfolio_weights_fund_flows(self, prices: pd.DataFrame, f: float, gamma: float = 1, r: float = 0.05) -> np.array:
        """
        Implementation of Theorem 4 in Jurek (2007).

        This method calculates the optimal portfolio allocation of an agent with
        constant relative risk aversion with utility defined over terminal wealth,
        (utility_type = 1) , in the presence of fund flows.

        Note: For all values of gamma, the presence of fund flows affect the general portfolio rule
        only by introducing a coefficient of proportionality.

        Also note, this method is only applicable for investors with utility defined over terminal wealth.

        What is f?

        f is the coefficient of proportionality for fund flows and is the drift term in the
        stochastic process equation for fund flows. (Refer Appendix C)

        :param prices: (pd.DataFrame) Contains price series of both stocks in spread.
        :param gamma: (float) coefficient of relative risk aversion. (gamma > 0).
        :param f: (float) coefficient of proportionality (f > 0).
        :param r: (float) Rate of Returns. (r > 0).
        :return: (np.array) Optimal weights with fund flows.
        """

        # Calculating the scaled optimal portfolio weights
        N = self.optimal_portfolio_weights(prices, utility_type=1, gamma=gamma, r=r)

        return (1 / (1 + f)) * N


    def stabilization_region(self, prices: pd.DataFrame, utility_type: int = 1, gamma: float = 1, beta: float = 0.01, r: float = 0.05) -> tuple:
        """
        Implementation of Theorem 3 in Jurek (2007).

        :param prices: (pd.DataFrame) Contains price series of both stocks in spread.
        :param utility_type: (int) Flag signifies type of investor preferences.
        :param gamma: (float) coefficient of relative risk aversion.
        :param beta: (float) Subjective rate of time preference. (Only required for utility_type = 2).
        :param r: (float) Rate of Returns.
        :return: (tuple) Tuple of numpy arrays for spread, min bound and max bound.
        """

        if self.sigma is None:
            raise Exception("Please run the fit method before calling this method.")

        if gamma <= 0:
            raise Exception("The value of gamma should be positive.")

        if utility_type not in [1, 2]:
            raise Exception("Please make sure utility_type is either 1 or 2.")

        # Setting instance attributes
        self.r = r
        self.gamma = gamma
        self.beta = beta

        # Calculating the time left array and the spread
        tau, S = self.spread_calc(prices)

        if self.gamma == 1:
            #  Calculation of A and B functions are not done in case of gamma = 1 (Refer Appendix A.1 and B.2)

            warnings.warn("Calculation of stabilization region is not implemented for gamma = 1.")
            return None

        A = None
        B = None
        # A and B functions are used in the final weights calculation
        if utility_type == 1:
            A, B = self._AB_calc_1(tau)

        else:
            A, B = self._AB_calc_2(tau)

        # Calculating phi (Refer Equation 17 in Jurek (2007))
        phi = (2 * A / self.gamma) - ((self.k + self.r) / (self.gamma * self.sigma ** 2))
        # Note : phi < 0.

        # term_1 and term_2 calculations are part of the constraint equation calculation below
        term_1 = (self.k * self.mu + (self.sigma ** 2) * B) / (self.gamma * self.sigma ** 2)
        term_2 = np.sqrt(-phi)

        # Initializing the upper bound and lower bound arrays
        max_bound = np.zeros(len(tau))
        min_bound = np.zeros(len(tau))

        # Iterating over each time step in the array
        for ind in range(len(tau)):
            s = cp.Variable() # Setting an optimization variable for spread
            # Defining the constraint equation involving spread
            constraint = [cp.abs(phi[ind] * s + term_1[ind]) <= term_2[ind] - 1e-6]

            # Solving the maximization problem to calculate the upper bound
            prob_max = cp.Problem(cp.Maximize(s), constraint)
            prob_max.solve()
            max_bound[ind] = prob_max.value

            # Solving the minimization problem to calculate the lower bound
            prob_min = cp.Problem(cp.Minimize(s), constraint)
            prob_min.solve()
            min_bound[ind] = prob_min.value

        return S, min_bound, max_bound


    def _AB_calc_1(self, tau: np.array) -> tuple:
        """
        This helper function computes the A and B functions for investors with utility_type = 1.
        The implementation follows Appendix A.2 in the paper.

        :param tau: (np.array) Array with time till completion in years.
        :return: (tuple) A and B arrays.
        """

        # Calculating value of variable c_1 in Appendix A.2.1
        c_1 = 2 * self.sigma ** 2 / self.gamma

        # Calculating value of variable c_2 in Appendix A.2.1
        c_2 = -(self.k / self.gamma + self.r * (1 - self.gamma) / self.gamma)

        # Calculating value of variable c_3 in Appendix A.2.1
        c_3 = 0.5 * ((1 - self.gamma) / self.gamma) * (((self.k + self.r) / self.sigma) ** 2)

        # Calculating value of discriminant in Appendix A.2.1
        disc = 4 * (self.k ** 2 - self.r ** 2 * (1 - self.gamma)) / self.gamma
        # Note: discriminant is always positive for gamma > 1

        # Calculating value of variable gamma_0 in Appendix A.2.1
        gamma_0 = 1 - (self.k / self.r) ** 2
        # Note: gamma_0 is not always > 0


        A = self._A_calc_1(tau, c_1, c_2, disc, gamma_0)  # Calculating value of function A
        B = self._B_calc_1(tau, c_1, c_2, c_3, disc, gamma_0)  # Calculating value of function B

        return A, B


    def _A_calc_1(self, tau: np.array, c_1: float, c_2: float, disc: float, gamma_0: float) -> np.array:
        """
        This method calculates the value of function A as described in the paper for investor with utility_type = 1.
        The implementation follows Appendix A.2.1 in the paper.

        Value of function A <= 0 and decreasing in tau for gamma > 1, and vice versa for gamma < 1.

        :param tau: (np.array) Array with time till completion in years.
        :param c_1: (float) Value of variable c_1 in Appendix A.2.1.
        :param c_2: (float) Value of variable c_2 in Appendix A.2.1.
        :param disc: (float) Value of discriminant in Appendix A.2.1.
        :param gamma_0: (float) Value of variable gamma_0 in Appendix A.2.1.
        :return: (np.array) A array.
        """

        A = None
        error_margin = 1e-4  # Error margin around gamma_0

        if 0 < self.gamma < gamma_0 - error_margin:

            A = -c_2 / c_1 + (np.sqrt(-disc) / (2 * c_1)) * np.tan(np.sqrt(-disc) * tau / 2 + np.arctan(2 * c_2 / np.sqrt(-disc)))

        elif gamma_0 - error_margin <= self.gamma <= gamma_0 + error_margin:

            A = -(c_2 / c_1) * (1 + 1 / (c_2 * tau - 1))

        elif gamma_0 + error_margin < self.gamma < 1:
            # coth = 1/tanh and arccoth(x) = arctanh(1/x)
            # For reference : https://www.efunda.com/math/hyperbolic/hyperbolic.cfm

            A = -c_2 / c_1 + (np.sqrt(disc) / (2 * c_1)) * (1 / np.tanh(-np.sqrt(disc) * tau / 2 + np.arctanh(np.sqrt(disc) / (2 * c_2))))

        else:
            # Case when self.gamma > 1

            A = -c_2 / c_1 + (np.sqrt(disc) / (2 * c_1)) * np.tanh(-np.sqrt(disc) * tau / 2 + np.arctanh(2 * c_2 / np.sqrt(disc)))

        return A


    def _B_calc_1(self, tau: np.array, c_1: float, c_2: float, c_3: float, disc: float, gamma_0: float) -> np.array:
        """
        This method calculates the value of function B as described in the paper for investor with utility_type = 1.
        The implementation follows Appendix A.2.2 in the paper.

        When self.mu = 0, c_4 and c_5 become zero and consequently B becomes zero.

        :param tau: (np.array) Array with time till completion in years.
        :param c_1: (float) Value of variable c_1 in Appendix A.2.1.
        :param c_2: (float) Value of variable c_2 in Appendix A.2.1.
        :param c_3: (float) Value of variable c_3 in Appendix A.2.1.
        :param disc: (float) Value of discriminant in Appendix A.2.1.
        :param gamma_0: (float) Value of variable gamma_0 in Appendix A.2.1.
        :return: (np.array) B array.
        """

        # Calculating value of variable c_4 in Appendix A.2.2
        c_4 = 2 * self.k * self.mu / self.gamma

        # Calculating value of variable c_5 in Appendix A.2.2
        c_5 = -((self.k + self.r) / self.sigma ** 2) * ((1 - self.gamma) / self.gamma) * self.k * self.mu

        B = None
        error_margin = 1e-4  # Error margin around gamma_0

        if 0 < self.gamma < gamma_0 - error_margin:

            # Calculating value of function phi_1 in Appendix A.2.2
            phi_1 = np.sqrt(-disc) * (np.cos(np.sqrt(-disc) * tau / 2) - 1) + 2 * c_2 * np.sin(np.sqrt(-disc) * tau / 2)

            # Calculating value of function phi_2 in Appendix A.2.2
            phi_2 = np.arctanh(np.tan(0.25 * (np.sqrt(-disc) * tau - 2 * np.arctan(2 * c_2 / np.sqrt(-disc))))) + \
                    np.arctanh(np.tan(0.5 * np.arctan(2 * c_2 / np.sqrt(-disc))))

            B = c_4 * phi_1 / (c_1 * np.sqrt(-disc)) + (4 * phi_2 / np.sqrt(-disc)) * (c_5 - c_4 / c_1) * \
                np.cos(np.sqrt(-disc) * tau / 2 - np.arctan(c_2 / np.sqrt(-disc)))

        elif gamma_0 - error_margin <= self.gamma <= gamma_0 + error_margin:

            B = (c_1 * c_5 * (c_2 * tau - 2) - (c_2 ** 2) * c_4) * tau / (2 * c_1 * (c_2 * tau - 1))

        else:
            # Case when self.gamma > gamma_0 + error_margin

            B = (4 * (c_2 * c_5 - c_3 * c_4 + (c_3 * c_4 - c_2 * c_5) * np.cosh(np.sqrt(disc) * tau / 2))
                 + 2 * c_5 * np.sqrt(disc) * np.sinh(np.sqrt(disc) * tau / 2)) \
                / (disc * np.cosh(np.sqrt(disc) * tau / 2) - 2 * c_2 * np.sqrt(disc) * np.sinh(np.sqrt(disc) * tau / 2))

        return B


    def _AB_calc_2(self, tau: np.array) -> tuple:
        """
        This helper function computes the A and B functions for investors with utility_type = 2.
        The implementation follows Appendix B.1 in the paper.

        :param tau: (np.array) Array with time till completion in years.
        :return: (tuple) A and B arrays.
        """

        # Calculating value of variable c_1 in Appendix B.1.1
        c_1 = 2 * self.sigma ** 2 / self.gamma

        # Calculating value of variable c_2 in Appendix B.1.1
        c_2 = (self.gamma * (2 * self.r - self.beta) - 2 * (self.k + self.r)) / (2 * self.gamma)

        # Calculating value of variable c_3 in Appendix B.1.1
        c_3 = ((self.k + self.r) ** 2) * (1 - self.gamma) / (2 * self.gamma * (self.sigma ** 2))

        # Calculating value of discriminant in Appendix B.1.1
        disc = ((2 * self.k + self.beta) ** 2 + (self.gamma - 1) * ((-2 * self.r + self.beta) ** 2)) / self.gamma

        # Calculating value of variable gamma_0 in Appendix B.1.1
        gamma_0 = 4 * (self.k + self.r) * (self.r - self.beta - self.k) / ((2 * self.r - self.beta) ** 2)


        A = self._A_calc_2(tau, c_1, c_2, disc, gamma_0)  # Calculating value of function A
        B = self._B_calc_2(tau, c_1, c_2, c_3, disc, gamma_0)  # Calculating value of function B

        return A, B


    def _A_calc_2(self, tau: np.array, c_1: float, c_2: float, disc: float, gamma_0: float) -> np.array:
        """
        This method calculates the value of function A as described in the paper for investor with utility_type = 2.
        The implementation follows Appendix B.1.1 in the paper.

        The formulas for calculating function A given in Appendix B.1.1 are the same as those in Appendix A.2.1,
        with the values of variables differing.

        :param tau: (np.array) Array with time till completion in years.
        :param c_1: (float) Value of variable c_1 in Appendix B.1.1.
        :param c_2: (float) Value of variable c_2 in Appendix B.1.1.
        :param disc: (float) Value of discriminant in Appendix B.1.1.
        :param gamma_0: (float) Value of variable gamma_0 in Appendix B.1.1.
        :return: (np.array) A array.
        """

        # Same calculation as for general CRRA Investor
        return self._A_calc_1(tau, c_1, c_2, disc, gamma_0)


    def _B_calc_2(self, tau: np.array, c_1: float, c_2: float, c_3: float, disc: float, gamma_0: float) -> np.array:
        """
        This method calculates the value of function B as described in the paper for investor with utility_type = 2.
        The implementation follows Appendix B.1.2 in the paper.

        When self.mu = 0, c_4 and c_6 become zero and consequently B becomes zero.

        :param tau: (np.array) Array with time till completion in years.
        :param c_1: (float) Value of variable c_1 in Appendix B.1.1.
        :param c_2: (float) Value of variable c_2 in Appendix B.1.1.
        :param c_3: (float) Value of variable c_3 in Appendix B.1.1.
        :param disc: (float) Value of discriminant in Appendix B.1.1.
        :param gamma_0: (float) Value of variable gamma_0 in Appendix B.1.1.
        :return: (np.array) B array.
        """

        # Calculating value of variable c_4 in Appendix B.1.2
        c_4 = 2 * self.k * self.mu / self.gamma

        # Calculating value of variable c_5 in Appendix B.1.2
        c_5 = -(self.k + self.r * (1 - self.gamma) + self.beta * self.gamma) / (2 * self.gamma)

        # Calculating value of variable c_6 in Appendix B.1.2
        c_6 = self.k * (self.k + self.r) * (self.gamma - 1) * self.mu / (self.gamma * self.sigma ** 2)

        B = None
        error_margin = 1e-4  # Error margin around gamma_0

        rep_exp_1 = np.exp(tau * c_2)  # Repeating Exponential Form with variable c_2
        rep_exp_2 = np.exp(tau * c_5)  # Repeating Exponential Form with variable c_5

        if 0 < self.gamma < gamma_0 - error_margin:
            # Implementation of Case I in Appendix B.1.2

            B = self._B_calc_2_I(c_1, c_2, c_3, c_4, c_5, c_6, rep_exp_1, rep_exp_2, tau)

        elif gamma_0 - error_margin <= self.gamma <= gamma_0 + error_margin:
            # Implementation of Case II in Appendix B.1.2

            B = self._B_calc_2_II(c_1, c_2, c_4, c_5, c_6, rep_exp_1, rep_exp_2, tau)

        elif gamma_0 + error_margin < self.gamma < 1:
            # Implementation of Case III in Appendix B.1.2

            B = self._B_calc_2_III(c_1, c_2, c_3, c_4, c_5, c_6, rep_exp_1, rep_exp_2, tau)

        else:
            # Case when self.gamma > 1
            # Implementation of Case IV in Appendix B.1.2

            B = self._B_calc_2_IV(c_1, c_2, c_3, c_4, c_5, c_6, disc, rep_exp_1, rep_exp_2, tau)

        return B


    @staticmethod
    def _B_calc_2_IV(c_1: float, c_2: float, c_3: float, c_4: float, c_5: float, c_6: float, disc: float,
                     rep_exp_1: np.array, rep_exp_2: np.array, tau: np.array) -> np.array:
        """
        Method calculates value of function B according to Case IV in Appendix B.1.2.

        :param c_1: (float) Value of variable c_1 in Appendix B.1.1.
        :param c_2: (float) Value of variable c_2 in Appendix B.1.1.
        :param c_3: (float) Value of variable c_3 in Appendix B.1.1.
        :param c_4: (float) Value of variable c_4 in Appendix B.1.2.
        :param c_5: (float) Value of variable c_5 in Appendix B.1.2.
        :param c_6: (float) Value of variable c_6 in Appendix B.1.2.
        :param disc: (float) Value of discriminant in Appendix B.1.1.
        :param rep_exp_1: (np.array) Repeating Exponential Form with variable c_2.
        :param rep_exp_2: (np.array) Repeating Exponential Form with variable c_5.
        :param tau: (np.array) Array with time till completion in years.
        :return: (np.array) Final value of B.
        """

        rep_phrase_1 = np.sqrt(-c_1 * c_3 / disc)  # Repeated Phrase 1
        rep_phrase_2 = 0.5 * np.sqrt(disc) * tau - np.arctanh(2 * c_2 / np.sqrt(disc))  # Repeated Phrase 2
        denominator = 2 * c_1 * rep_phrase_1 * (c_1 * c_3 + c_5 * (c_5 - 2 * c_2))  # Denominator in final equation

        # The final equation for B is split into 5 terms
        term_1 = 2 * rep_exp_1 * rep_phrase_1 * c_4 * c_5 * (c_2 + 0.5 * np.sqrt(disc) * np.tanh(rep_phrase_2))

        term_2 = 2 * rep_exp_1 * rep_phrase_1 - 2 * rep_exp_2 * (1 / np.cosh(rep_phrase_2))
        # sech = 1 / cosh

        term_3 = rep_exp_2 * (1 / np.cosh(rep_phrase_2)) - 2 * rep_exp_1 * rep_phrase_1

        term_4 = rep_exp_1 * np.sqrt(disc) * rep_phrase_1 * np.tanh(rep_phrase_2)

        term_5 = -term_3

        # Calculating the final value of function B
        B = np.exp(-tau * c_2) * (term_1
                                  + c_1 * (c_6 * (c_2 * term_2
                                                  + term_3 * c_5
                                                  - term_4)
                                           - c_3 * term_5 * c_4)) / denominator

        return B


    @staticmethod
    def _B_calc_2_III(c_1: float, c_2: float, c_3: float, c_4: float, c_5: float, c_6: float,
                      rep_exp_1: np.array, rep_exp_2: np.array, tau: np.array) -> np.array:
        """
        Method calculates value of function B according to Case III in Appendix B.1.2.

        :param c_1: (float) Value of variable c_1 in Appendix B.1.1.
        :param c_2: (float) Value of variable c_2 in Appendix B.1.1.
        :param c_3: (float) Value of variable c_3 in Appendix B.1.1.
        :param c_4: (float) Value of variable c_4 in Appendix B.1.2.
        :param c_5: (float) Value of variable c_5 in Appendix B.1.2.
        :param c_6: (float) Value of variable c_6 in Appendix B.1.2.
        :param rep_exp_1: (np.array) Repeating Exponential Form with variable c_2.
        :param rep_exp_2: (np.array) Repeating Exponential Form with variable c_5.
        :param tau: (np.array) Array with time till completion in years.
        :return: (np.array) Final value of B.
        """

        rep_phrase_1 = np.sqrt(c_1 * c_3 / c_2 ** 2)  # Repeated Phrase 1
        rep_phrase_2 = np.sqrt(c_2 ** 2 - c_1 * c_3)  # Repeated Phrase 2
        rep_phrase_3 = np.arctanh(rep_phrase_2 / c_2) - tau * rep_phrase_2  # Repeated Phrase 3
        # arccoth(x) = arctanh(1/x)
        denominator = c_2 * rep_phrase_1 * (c_1 * c_3 + c_5 * (c_5 - 2 * c_2))  # Denominator in final equation

        # The final equation for B is split into 5 terms
        term_1 = rep_exp_1 * rep_phrase_1 * c_6 * c_2 ** 2

        term_2 = rep_exp_1 * c_3 * rep_phrase_1 * c_4

        term_3 = 2 * rep_exp_2 * (1 / np.sinh(rep_phrase_3)) - rep_phrase_2 * (1 / np.tanh(rep_phrase_3)) * rep_phrase_1
        # csch = 1/sinh, coth = 1/tanh

        term_4 = rep_exp_1 * rep_phrase_1 * c_5

        term_5 = (1 / np.sinh(rep_phrase_3)) * (
                    c_3 * c_4 * (rep_exp_2 * rep_phrase_2 - rep_exp_1 * np.sinh(tau * rep_phrase_2) * c_5)
                    + rep_exp_2 * rep_phrase_2 * c_5 * c_6)
        # csch = 1/sinh

        # Calculating the final value of function B
        B = np.exp(-tau * c_2) * (term_1 - (term_2 + (rep_phrase_2 * term_3 + term_4) * c_6) * c_2
                                  + term_5) / denominator

        return B


    @staticmethod
    def _B_calc_2_II(c_1: float, c_2: float, c_4: float, c_5: float, c_6: float,
                     rep_exp_1: np.array, rep_exp_2: np.array, tau: np.array) -> np.array:
        """
        Method calculates value of function B according to Case II in Appendix B.1.2.

        :param c_1: (float) Value of variable c_1 in Appendix B.1.1.
        :param c_2: (float) Value of variable c_2 in Appendix B.1.1.
        :param c_4: (float) Value of variable c_4 in Appendix B.1.2.
        :param c_5: (float) Value of variable c_5 in Appendix B.1.2.
        :param c_6: (float) Value of variable c_6 in Appendix B.1.2.
        :param rep_exp_1: (np.array) Repeating Exponential Form with variable c_2.
        :param rep_exp_2: (np.array) Repeating Exponential Form with variable c_5.
        :param tau: (np.array) Array with time till completion in years.
        :return: (np.array) Final value of B.
        """

        denominator = c_1 * rep_exp_1 * (tau * c_2 - 1) * (c_2 - c_5) ** 2  # Denominator in final equation

        # The final equation for B is split into 4 terms
        term_1 = rep_exp_1 * tau * c_4 * c_2 ** 3

        term_2 = (c_4 * (rep_exp_1 * (tau * c_5 + 1) - rep_exp_2) + rep_exp_1 * tau * c_1 * c_6) * c_2 ** 2

        term_3 = c_1 * (rep_exp_1 * tau * c_5 + 2 * (rep_exp_1 - rep_exp_2)) * c_6 * c_2

        term_4 = (rep_exp_1 - rep_exp_2) * c_1 * c_5 * c_6

        # Calculating the final value of function B
        B = (-term_1 + term_2 - term_3
             + term_4) / denominator

        return B


    @staticmethod
    def _B_calc_2_I(c_1: float, c_2: float, c_3: float, c_4: float, c_5: float, c_6: float,
                    rep_exp_1: np.array, rep_exp_2: np.array, tau: np.array) -> np.array:
        """
        Method calculates value of function B according to Case I in Appendix B.1.2.

        :param c_1: (float) Value of variable c_1 in Appendix B.1.1.
        :param c_2: (float) Value of variable c_2 in Appendix B.1.1.
        :param c_3: (float) Value of variable c_3 in Appendix B.1.1.
        :param c_4: (float) Value of variable c_4 in Appendix B.1.2.
        :param c_5: (float) Value of variable c_5 in Appendix B.1.2.
        :param c_6: (float) Value of variable c_6 in Appendix B.1.2.
        :param rep_exp_1: (np.array) Repeating Exponential Form with variable c_2.
        :param rep_exp_2: (np.array) Repeating Exponential Form with variable c_5.
        :param tau: (np.array) Array with time till completion in years.
        :return: (np.array) Final value of B.
        """

        rep_phrase_1 = np.sqrt(c_1 * c_3 - c_2 ** 2)  # Repeating Phrase 1
        rep_phrase_2 = np.sqrt(c_1 * c_3) / rep_phrase_1  # Repeating Phrase 2
        rep_phrase_3 = rep_phrase_1 * tau + np.arctan(c_2 / rep_phrase_1)  # Repeating Phrase 3
        denominator = c_1 * rep_phrase_2 * (c_1 * c_3 + c_5 * (c_5 - 2 * c_2))  # Denominator in final equation

        # The final equation for B is split into 5 terms
        term_1 = rep_exp_1 * rep_phrase_2 * c_4 * c_5 * (c_2 - rep_phrase_1 * np.tan(rep_phrase_3))

        term_2 = rep_exp_1 * rep_phrase_2 - 2 * rep_exp_2 * (1 / np.cos(rep_phrase_3))

        term_3 = rep_exp_2 * (1 / np.cos(rep_phrase_3)) - rep_exp_1 * rep_phrase_2

        term_4 = rep_exp_1 * rep_phrase_2 * rep_phrase_1 * np.tan(rep_phrase_3)

        term_5 = -term_3

        # Calculating the final value of function B
        B = np.exp(-tau * c_2) * (term_1
                                  + c_1 * (c_6 * (c_2 * term_2
                                                  + term_3 * c_5
                                                  + term_4)
                                           - c_3 * term_5 * c_4)) / denominator

        return B


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

        :return: (pd.Series) Series describing parameter values.
        """

        if self.sigma is None:
            raise Exception("Please run the fit method before calling describe.")

        # List defines the indexes of the final pandas object
        index = ['Ticker of first stock', 'Ticker of second stock', 'Scaled Spread weights',
                 'long-term mean', 'rate of mean reversion', 'standard deviation', 'half-life']

        # List defines the values of the final pandas object
        data = [self.ticker_A, self.ticker_B, np.round(self.scaled_spread_weights.values, 3),
                self.mu, self.k, self.sigma, self._calc_half_life(self.k)]

        # Combine data and indexes into the pandas Series
        output = pd.Series(data=data, index=index)

        return output


    def plot_results(self, prices: pd.DataFrame, num_test_windows: int = 5, delta_t: float = 1 / 252, utility_type: int = 1,
                     gamma: float = 10, beta: float = 0.01, r: float = 0.05, f: float = 0.1, figsize: tuple = (8, 4), fontsize: int = 8):
        # pylint: disable=too-many-arguments, too-many-statements
        """
        Method plots out of sample performance of the model on specified number of test windows.
        We use a backward looking rolling window as training data and its size depends on the number of test windows chosen.
        The length of training data is fixed for all test windows.

        For example, if the total data is of length 16 years, with the number of test windows set to 5,
        the length of training data would be 16 - (5 + 1) = 10. (The last year is not considered for testing).

        This method plots the stabilization region, optimal portfolio weights with and without fund flows, and the
        evolution of the wealth process with initial wealth normalized to 1.

        :param prices: (pd.DataFrame) Contains price series of both stocks in spread with dates as index.
        :param num_test_windows: (int) Number of out of sample testing windows to plot.
        :param delta_t: (float) Time difference between each index of prices, calculated in years.
        :param utility_type: (int) Flag signifies type of investor preferences.
        :param gamma: (float) coefficient of relative risk aversion.
        :param beta: (float) Subjective rate of time preference. (Only required for utility_type = 2).
        :param r: (float) Rate of Returns.
        :param f: (float) Coefficient of proportionality (assumed to be positive).
        :param figsize: (tuple) Input to matplotlib figsize parameter for plotting.
        :param fontsize: (int) general matplotlib font size for plotting.
        """

        # Setting font size of plots
        plt.rcParams.update({'font.size': fontsize})

        # Price series preprocessing
        prices = prices.ffill()

        if not np.issubdtype(prices.index.dtype, np.datetime64):
            raise Exception("Please make sure index of dataframe is datetime type.")

        if len(prices) < (10 / delta_t):
            raise Exception("Please make sure length of input data is greater than 10 years. "
                            "This is the time period used to fit the model in the paper.")

        # Getting the list of years in input data
        years = prices.index.year.unique()

        # Initializing a dataframe which stores stabilization region results
        stab_result_dataframe = pd.DataFrame(index=prices.loc[str(years[-(num_test_windows + 1)]):str(years[-1])].index,
                                             columns=['Spread', 'lower bound', 'upper bound'])
        # Initializing a dataframe which stores optimal weights
        optimal_result_dataframe = pd.DataFrame(index=prices.loc[str(years[-(num_test_windows + 1)]):str(years[-1])].index,
                                                columns=['Weights'])
        # Initializing a dataframe which stores optimal weights in the case with fund flows
        optimal_fund_flows_result_dataframe = pd.DataFrame(index=prices.loc[str(years[-(num_test_windows + 1)]):
                                                                          str(years[-1])].index, columns=['Weights'])
        # Initializing a dataframe which stores wealth
        wealth_dataframe = pd.DataFrame(index=prices.loc[str(years[-(num_test_windows + 1)]):str(years[-1])].index,
                                                columns=['Wealth'])

        ind = 0
        W_initial = 1  # Initial wealth normalized to 1
        # Iterating over the test windows
        for year in np.arange(years[-(num_test_windows + 1)], years[-1], 1):

            # Setting the train and test data
            data_train_dataframe = prices.loc[str(year - (len(years) - (num_test_windows + 1))):str(year - 1)]
            data_test_dataframe = prices.loc[str(year)]

            self.fit(data_train_dataframe, delta_t=delta_t)

            optimal_weights = self.optimal_portfolio_weights(data_test_dataframe, gamma=gamma,
                                                             utility_type=utility_type, beta=beta, r=r)
            optimal_fund_flow_weights = self.optimal_portfolio_weights_fund_flows(data_test_dataframe, gamma=gamma,
                                                                                  f=f, r=r)
            S, min_bound, max_bound = self.stabilization_region(data_test_dataframe, gamma=gamma,
                                                                utility_type=utility_type, beta=beta, r=r)

            W = np.zeros(len(data_test_dataframe))
            W[0] = W_initial

            if utility_type == 1:
                for i in range(len(data_test_dataframe) - 1):
                    # Calculating the wealth process for CRRA investor. Follows equation (3) in Appendix A
                    W[i + 1] = W[i] + W[i] * optimal_weights[i] * (S[i + 1] - S[i]) + r * W[i] * (
                                1 - optimal_weights[i] * S[i]) * delta_t
            else:
                for i in range(len(data_test_dataframe) - 1):
                    # Calculating the wealth process for investor with intermediate consumption
                    # Follows equation (39) in Appendix A
                    # Here we add an extra term which denotes consumption
                    W[i + 1] = W[i] + W[i] * optimal_weights[i] * (S[i + 1] - S[i]) + r * W[i] * (
                                1 - optimal_weights[i] * S[i]) * delta_t - beta * W[i] * delta_t

            W_initial = W[-1]

            # Adding the final results to their corresponding dataframes for each test window
            stab_result_dataframe.iloc[ind:ind + len(S), :] = np.array([S, min_bound, max_bound]).T
            optimal_result_dataframe.iloc[ind:ind + len(S), :] = np.array([optimal_weights]).T
            optimal_fund_flows_result_dataframe.iloc[ind:ind + len(S), :] = np.array([optimal_fund_flow_weights]).T
            wealth_dataframe.iloc[ind:ind + len(S), :] = np.array([W]).T
            ind += len(S)

        # Plotting the stabilization bound plot
        ax = stab_result_dataframe.plot(style=['c-', 'r:', 'r:'], legend=False, linewidth=1.0, figsize=figsize)
        ax.xaxis.grid(color='grey', linestyle=':', linewidth=0.6)
        ax.set_ylabel('Spread')
        ax.set_title("Evolution of spread with stabilization bound")
        plt.show()

        # Plotting the optimal weights allocation plot
        ax = optimal_result_dataframe.plot(style=['c-'], legend=False, linewidth=1.0, figsize=figsize)
        ax.xaxis.grid(color='grey', linestyle=':', linewidth=0.6)
        ax.set_ylabel('Optimal Weights')
        ax.set_title('Optimal allocation to the spread asset scaled by wealth')
        plt.show()

        # Plotting the optimal weights allocation plot in the case with fund flows
        ax = optimal_fund_flows_result_dataframe.plot(style=['c-'], legend=False, linewidth=1.0, figsize=figsize)
        ax.xaxis.grid(color='grey', linestyle=':', linewidth=0.6)
        ax.set_ylabel('Optimal Weights with fund flows')
        ax.set_title('Optimal allocation to the spread asset with fund flows scaled by wealth')
        plt.show()

        # Plotting the wealth plot
        ax = wealth_dataframe.plot(style=['c-'], legend=False, linewidth=1.0, figsize=figsize)
        ax.xaxis.grid(color='grey', linestyle=':', linewidth=0.6)
        ax.set_ylabel('Wealth')
        ax.set_title('Evolution of wealth over lifetime of simulation, with initial wealth normalized to 1')
        plt.show()
