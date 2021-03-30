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

from arbitragelab.cointegration_approach.engle_granger import EngleGrangerPortfolio


class StochasticControlJurek:
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

        # Characteristics of Training Data.
        self.ticker_A = None # Ticker Symbol of first stock.
        self.ticker_B = None # Ticker Symbol of second stock.
        self.spread = None # Constructed spread from training data.
        self.scaled_spread_weights = None # Scaled weights for the prices in spread.
        self.time_array = None # Time indices of training data.
        self.delta_t = None # Time difference between each index in data, calculated in years.

        # Estimated params from training data.
        self.sigma = None # Standard deviation of spread.
        self.mu = None # Long run mean of spread.
        self.k = None # Rate of mean reversion.

        # Params inputted by user.
        self.r = None # Rate of returns.
        self.gamma = None # Coefficient of relative risk aversion.
        self.beta = None # Rate of time preference.


    @staticmethod
    def _calc_total_return_indices(data : pd.DataFrame) -> pd.DataFrame:
        """
        This method calculates the total return indices from pricing data.
        This calculation follows Section IV A in Jurek (2007).

        :param data: (pd.DataFrame) Contains price series of both stocks in spread.
        """

        #Calculating the daily returns.
        returns_df = data.pct_change()
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().dropna()

        total_return_indices = data.copy()
        total_return_indices.iloc[0, :] = 1
        total_return_indices.iloc[1:, :] = pd.DataFrame.cumprod(1 + returns_df, axis=0)

        return total_return_indices


    def fit(self, data: pd.DataFrame, delta_t: float = 1 / 252, adf_test: bool = False, significance_level: float = 0.95):
        """
        This method uses inputted training data to calculate the spread and
        estimate the parameters of the corresponding OU process.

        The spread construction implementation follows Section IV A in Jurek (2007).

        :param data: (pd.DataFrame) Contains price series of both stocks in spread.
        :param delta_t: (float) Time difference between each index of data, calculated in years.
        :param adf_test: (bool) Flag which defines whether the adf statistic test should be conducted.
        :param significance_level: (float) This significance level is used in the ADF statistic test.
            Value can be one of the following: (0.90, 0.95, 0.99).
        """

        if len(data) < (10 / delta_t):
            warnings.warn("Please make sure length of training data is greater than 10 years.")

        # Setting instance attributes.
        self.delta_t = delta_t
        self.ticker_A, self.ticker_B = data.columns[0], data.columns[1]

        #Calculating the total return indices from pricing data.
        total_return_indices = self._calc_total_return_indices(data)
        self.time_array = np.arange(0, len(total_return_indices)) * self.delta_t

        # As mentioned in the paper, the vector of linear weights are calculated using co-integrating regression.
        eg_portfolio = EngleGrangerPortfolio()
        eg_portfolio.fit(total_return_indices, add_constant=True)  # Fitting the total return indices.
        eg_adf_statistics = eg_portfolio.adf_statistics  # Stores the results of the ADF statistic test.
        eg_cointegration_vectors = eg_portfolio.cointegration_vectors # Stores the calculated weights for the pair o stocks in the spread.

        if adf_test is True and eg_adf_statistics.loc['statistic_value', 0] > eg_adf_statistics.loc[f'{int(significance_level * 100)}%', 0]:
            # Making sure that the data passes the ADF statistic test.
            print(eg_adf_statistics)
            warnings.warn("ADF statistic test failure.")

        # Scaling the weights such that they sum to 1.
        self.scaled_spread_weights = eg_cointegration_vectors.loc[0] / abs(eg_cointegration_vectors.loc[0]).sum()

        self.spread = (total_return_indices * self.scaled_spread_weights).sum(axis=1)

        self.spread = self.spread.to_numpy() # TODO : This conversion seems to have changed the estimated values. Not sure why?

        params = self._estimate_params(self.spread)
        self.mu, self.k, self.sigma = params

        #self._check_estimations()


    def _estimate_params(self, spread: np.array):
        """
        This method implements the closed form solutions for estimators of the model parameters.
        These formulas for the estimators are given in Appendix E of Jurek (2007).

        :param: (np.array) Price series of the constructed spread.
        """

        N = len(spread)

        # Mean estimator.
        mu = spread.mean()

        # Estimator for rate of mean reversion.
        k = (-1 / self.delta_t) * np.log(np.multiply(spread[1:] - mu, spread[:-1] - mu).sum()
                                              / np.power(spread[1:] - mu, 2).sum())

        sigma_calc_sum = np.power((spread[1:] - mu - np.exp(-k * self.delta_t) * (spread[:-1] - mu))
                                  / np.exp(-k * self.delta_t), 2).sum()

        #Estimator for standard deviation.
        sigma = np.sqrt(2 * k * sigma_calc_sum / ((np.exp(2 * k * self.delta_t) - 1) * (N - 2)))

        return mu, k, sigma


    # def _check_estimations(self):
    #     """
    #     Testing against null of random walk for rate of mean reversion k.
    #     """
    #
    #     num_paths = 100000
    #
    #     output_params = np.zeros((num_paths, 3))
    #     for i in range(num_paths):
    #         white_noise_process = self.sigma * np.random.randn(len(self.spread)) + self.mu
    #         output_params[i, :] = self._estimate_params(white_noise_process)
    #
    #     plt.hist(output_params[:, 1], bins=20)
    #     plt.show()
    #     #TODO : This is incomplete.


    def _spread_calc(self, data: pd.DataFrame):
        """
        This method calculates the spread on test data using the scaled weights from training data.

        :param data: (pd.DataFrame) Contains price series of both stocks in spread.
        """

        #Calculating the total return indices from pricing data.
        total_return_indices = self._calc_total_return_indices(data)
        t = np.arange(0, len(total_return_indices)) * self.delta_t
        tau = t[-1] - t  # Stores time remaining till closure. (In years)

        # Calculating the spread with weights calculated from training data.
        S = (total_return_indices * self.scaled_spread_weights).sum(axis=1)

        S = S.to_numpy() # TODO : This conversion in fit seems to have changed the estimated values.

        return tau, S


    def optimal_portfolio_weights(self, data: pd.DataFrame, utility_type: int = 1, gamma: float = 1, beta: float = 0.1, r: float = 0.05):
        """
        Implementation of Theorem 1 and Theorem 2 in Jurek (2007).

        This method implements the optimal portfolio strategy for two types of investors with varying utility functions.

        The first type of investor is represented by utility_type = 1.
        This agent has constant relative risk aversion preferences(CRRA investor) with utility defined over terminal wealth.
        For this type of investor,  gamma = 1 implies log utility investor, and
                                    gamma != 1 implies general CRRA investor.

        The second type of investor is represented by utility_type = 2.
        This agent has utility defined over intermediate consumption,
        with agent’s preferences described by Epstein-Zin recursive utility with psi(elasticity of intertemporal substitution) = 1.
        For this type of investor,  gamma = 1 reduces to standard log utility investor, and
                                    gamma > 1 implies more risk averse investors,
                                    whereas gamma < 1 implies more risk tolerance in comparison to investor with log utility.

        What is beta?
        Beta signifies the constant fraction of total wealth an investor chooses to consume. This is analogous to a hedge fund investor
        who cares both about terminal wealth in some risk-averse way and consumes a constant fraction, β,
        of assets under management (the management fee).
        For utility_type = 2, C(Consumption) = beta * W.

        :param data: (pd.DataFrame) Contains price series of both stocks in spread.
        :param utility_type: (int) Flag signifies type of investor preferences.
        :param gamma: (float) coefficient of relative risk aversion.
        :param beta: (float) Subjective rate of time preference. (Only required for utility_type = 2).
        :param r: (float) Rate of Returns.
        """

        # Setting instance attributes.
        self.r = r
        self.gamma = gamma
        self.beta = beta

        tau, S = self._spread_calc(data)

        W = np.ones(len(tau))  # Wealth is normalized to one.

        N = None
        # The optimal weights equation is the same for both types of utility functions.
        # For gamma = 1, the outputs weights are identical for both types of utility functions,
        # whereas for gamma != 1, the calculation of A and B functions in the equation are different.
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

            N = ((self.k * (self.mu - S) - self.r * S) / (self.gamma * self.sigma ** 2) + (2 * A * S + B) / self.gamma) * W

        return N / W # We return the optimal allocation of spread asset scaled by wealth.


    def optimal_portfolio_weights_fund_flows(self, data: pd.DataFrame, f: float, gamma: float = 1, r: float = 0.05):
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

        :param data: (pd.DataFrame) Contains price series of both stocks in spread.
        :param gamma: (float) coefficient of relative risk aversion.
        :param f: (float) coefficient of proportionality (assumed to be positive).
        :param r: (float) Rate of Returns.
        """

        N = self.optimal_portfolio_weights(data, utility_type=1, gamma=gamma, r=r)

        return (1 / (1 + f)) * N


    def stabilization_region_calc(self, data: pd.DataFrame, utility_type: int = 1, gamma: float = 1, beta: float = 0.1, r: float = 0.05):
        """
        Implementation of Theorem 3 in Jurek (2007).

        :param data: (pd.DataFrame) Contains price series of both stocks in spread.
        :param utility_type: (int) Flag signifies type of investor preferences.
        :param gamma: (float) coefficient of relative risk aversion.
        :param beta: (float) Subjective rate of time preference. (Only required for utility_type = 2).
        :param r: (float) Rate of Returns.
        """

        # Setting instance attributes.
        self.r = r
        self.gamma = gamma
        self.beta = beta

        tau, S = self._spread_calc(data)

        if self.gamma == 1:
            #  Calculation of A and B functions are not done in case of gamma = 1. (Refer Appendix A.1 and B.2).

            warnings.warn("Calculation of stabilization region is not implemented for gamma = 1.")
            return None

        A = None
        B = None
        if utility_type == 1:
            A, B = self._AB_calc_1(tau)

        elif utility_type == 2:
            A, B = self._AB_calc_2(tau)

        # Calculating phi (Refer Equation 17 in Jurek (2007)).
        phi = (2 * A / self.gamma) - ((self.k + self.r) / (self.gamma * self.sigma ** 2))
        # Note : phi < 0.

        term_1 = (self.k * self.mu + (self.sigma ** 2) * B) / (self.gamma * self.sigma ** 2)
        term_2 = np.sqrt(-phi)

        max_bound = np.zeros(len(tau))
        min_bound = np.zeros(len(tau))

        for ind in range(len(tau)):
            s = cp.Variable()
            constraint = [cp.abs(phi[ind] * s + term_1[ind]) <= term_2[ind] - 1e-6]

            prob_max = cp.Problem(cp.Maximize(s), constraint)
            prob_max.solve()
            max_bound[ind] = prob_max.value

            prob_min = cp.Problem(cp.Minimize(s), constraint)
            prob_min.solve()
            min_bound[ind] = prob_min.value

        return S, min_bound, max_bound


    def _AB_calc_1(self, tau):
        """
        This helper function computes the A and B functions for investors with utility_type = 1.
        The implementation follows Appendix A.2 in the paper.

        :param tau: (np.array) Array with time till completion in years.
        """

        # Calculating value of variable c_1 in Appendix A.2.1.
        c_1 = 2 * self.sigma ** 2 / self.gamma

        # Calculating value of variable c_2 in Appendix A.2.1.
        c_2 = -(self.k / self.gamma + self.r * (1 - self.gamma) / self.gamma)

        # Calculating value of variable c_3 in Appendix A.2.1.
        c_3 = 0.5 * ((1 - self.gamma) / self.gamma) * (((self.k + self.r) / self.sigma) ** 2)

        # Calculating value of discriminant in Appendix A.2.1.
        disc = 4 * (self.k ** 2 - self.r ** 2 * (1 - self.gamma)) / self.gamma
        # Note: discriminant is always positive for gamma > 1.

        # Calculating value of variable gamma_0 in Appendix A.2.1.
        gamma_0 = 1 - (self.k / self.r) ** 2
        # Note: gamma_0 is not always > 0.


        A = self._A_calc_1(tau, c_1, c_2, disc, gamma_0)  # Calculating value of function A.
        B = self._B_calc_1(tau, c_1, c_2, c_3, disc, gamma_0)  # Calculating value of function B.

        return A, B


    def _A_calc_1(self, tau, c_1, c_2, disc, gamma_0):
        """
        This method calculates the value of function A as described in the paper for investor with utility_type = 1.
        The implementation follows Appendix A.2.1 in the paper.

        Value of function A <= 0 and decreasing in tau for gamma > 1, and vice versa for gamma < 1.

        :param tau: (np.array) Array with time till completion in years.
        :param c_1: (float) Value of variable c_1 in Appendix A.2.1.
        :param c_2: (float) Value of variable c_2 in Appendix A.2.1.
        :param disc: (float) Value of discriminant in Appendix A.2.1.
        :param gamma_0: (float) Value of variable gamma_0 in Appendix A.2.1.
        """

        A = None
        error_margin = 1e-4  # Error margin around gamma_0.

        if 0 < self.gamma < gamma_0 - error_margin:

            A = -c_2 / c_1 + (np.sqrt(-disc) / (2 * c_1)) * np.tan(np.sqrt(-disc) * tau / 2 + np.arctan(2 * c_2 / np.sqrt(-disc)))

        elif gamma_0 - error_margin <= self.gamma <= gamma_0 + error_margin:

            A = -(c_2 / c_1) * (1 + 1 / (c_2 * tau - 1))

        elif gamma_0 + error_margin < self.gamma < 1:
            # coth = 1/tanh and arccoth(x) = arctanh(1/x)
            # For reference : https://www.efunda.com/math/hyperbolic/hyperbolic.cfm

            A = -c_2 / c_1 + (np.sqrt(disc) / (2 * c_1)) * (1 / np.tanh(-np.sqrt(disc) * tau / 2 + np.arctanh(np.sqrt(disc) / (2 * c_2))))

        elif self.gamma > 1:

            A = -c_2 / c_1 + (np.sqrt(disc) / (2 * c_1)) * np.tanh(-np.sqrt(disc) * tau / 2 + np.arctanh(2 * c_2 / np.sqrt(disc)))

        return A


    def _B_calc_1(self, tau, c_1, c_2, c_3, disc, gamma_0):
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
        """

        # Calculating value of variable c_4 in Appendix A.2.2.
        c_4 = 2 * self.k * self.mu / self.gamma

        # Calculating value of variable c_5 in Appendix A.2.2.
        c_5 = -((self.k + self.r) / self.sigma ** 2) * ((1 - self.gamma) / self.gamma) * self.k * self.mu

        B = None
        error_margin = 1e-4  # Error margin around gamma_0.

        if 0 < self.gamma < gamma_0 - error_margin:

            # Calculating value of function phi_1 in Appendix A.2.2.
            phi_1 = np.sqrt(-disc) * (np.cos(np.sqrt(-disc) * tau / 2) - 1) + 2 * c_2 * np.sin(np.sqrt(-disc) * tau / 2)

            # Calculating value of function phi_2 in Appendix A.2.2.
            phi_2 = np.arctanh(np.tan(0.25 * (np.sqrt(-disc) * tau - 2 * np.arctan(2 * c_2 / np.sqrt(-disc))))) + \
                    np.arctanh(np.tan(0.5 * np.arctan(2 * c_2 / np.sqrt(-disc))))

            B = c_4 * phi_1 / (c_1 * np.sqrt(-disc)) + (4 * phi_2 / np.sqrt(-disc)) * (c_5 - c_4 / c_1) * \
                np.cos(np.sqrt(-disc) * tau / 2 - np.arctan(c_2 / np.sqrt(-disc)))

        elif gamma_0 - error_margin <= self.gamma <= gamma_0 + error_margin:

            B = (c_1 * c_5 * (c_2 * tau - 2) - (c_2 ** 2) * c_4) * tau / (2 * c_1 * (c_2 * tau - 1))

        elif self.gamma > gamma_0 + error_margin:

            B = (4 * (c_2 * c_5 - c_3 * c_4 + (c_3 * c_4 - c_2 * c_5) * np.cosh(np.sqrt(disc) * tau / 2))
                 + 2 * c_5 * np.sqrt(disc) * np.sinh(np.sqrt(disc) * tau / 2)) \
                / (disc * np.cosh(np.sqrt(disc) * tau / 2) - 2 * c_2 * np.sqrt(disc) * np.sinh(np.sqrt(disc) * tau / 2))

        return B


    def _AB_calc_2(self, tau):
        """
        This helper function computes the A and B functions for investors with utility_type = 2.
        The implementation follows Appendix B.1 in the paper.

        :param tau: (np.array) Array with time till completion in years.
        """

        # Calculating value of variable c_1 in Appendix B.1.1.
        c_1 = 2 * self.sigma ** 2 / self.gamma

        # Calculating value of variable c_2 in Appendix B.1.1.
        c_2 = (self.gamma * (2 * self.r - self.beta) - 2 * (self.k + self.r)) / (2 * self.gamma)

        # Calculating value of variable c_3 in Appendix B.1.1.
        c_3 = ((self.k + self.r) ** 2) * (1 - self.gamma) / (2 * self.gamma * (self.sigma ** 2))

        # Calculating value of discriminant in Appendix B.1.1.
        disc = ((2 * self.k + self.beta) ** 2 + (self.gamma - 1) * ((-2 * self.r + self.beta) ** 2)) / self.gamma

        # Calculating value of variable gamma_0 in Appendix B.1.1.
        gamma_0 = 4 * (self.k + self.r) * (self.r - self.beta - self.k) / ((2 * self.r - self.beta) ** 2)


        A = self._A_calc_2(tau, c_1, c_2, disc, gamma_0)  # Calculating value of function A.
        B = self._B_calc_2(tau, c_1, c_2, c_3, disc, gamma_0)  # Calculating value of function B.

        return A, B


    def _A_calc_2(self, tau, c_1, c_2, disc, gamma_0):
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
        """

        # Same calculation as for general CRRA Investor.
        return self._A_calc_1(tau, c_1, c_2, disc, gamma_0)


    def _B_calc_2(self, tau, c_1, c_2, c_3, disc, gamma_0):
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
        """

        # Calculating value of variable c_4 in Appendix B.1.2.
        c_4 = 2 * self.k * self.mu / self.gamma

        # Calculating value of variable c_5 in Appendix B.1.2.
        c_5 = -(self.k + self.r * (1 - self.gamma) + self.beta * self.gamma) / (2 * self.gamma)

        # Calculating value of variable c_6 in Appendix B.1.2.
        c_6 = self.k * (self.k + self.r) * (self.gamma - 1) * self.mu / (self.gamma * self.sigma ** 2)

        B = None
        error_margin = 1e-4  # Error margin around gamma_0.

        rep_exp_1 = np.exp(tau * c_2)  # Repeating Exponential Form with variable c_2.
        rep_exp_2 = np.exp(tau * c_5)  # Repeating Exponential Form with variable c_5.

        if 0 < self.gamma < gamma_0 - error_margin:
            # Implementation of Case I in Appendix B.1.2.

            rep_phrase_1 = np.sqrt(c_1 * c_3 - c_2 ** 2)  # Repeating Phrase 1.
            rep_phrase_2 = np.sqrt(c_1 * c_3) / rep_phrase_1  # Repeating Phrase 2.
            rep_phrase_3 = rep_phrase_1 * tau + np.arctan(c_2 / rep_phrase_1)  # Repeating Phrase 3.

            denominator = c_1 * rep_phrase_2 * (c_1 * c_3 + c_5 * (c_5 - 2 * c_2))  # Denominator in final equation.

            # The final equation for B is split into 5 terms.

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


        elif gamma_0 - error_margin <= self.gamma <= gamma_0 + error_margin:
            # Implementation of Case II in Appendix B.1.2.

            denominator = c_1 * rep_exp_1 * (tau * c_2 - 1) * (c_2 - c_5) ** 2  # Denominator in final equation.

            # The final equation for B is split into 4 terms.

            term_1 = rep_exp_1 * tau * c_4 * c_2 ** 3

            term_2 = (c_4 * (rep_exp_1 * (tau * c_5 + 1) - rep_exp_2) + rep_exp_1 * tau * c_1 * c_6) * c_2 ** 2

            term_3 = c_1 * (rep_exp_1 * tau * c_5 + 2 * (rep_exp_1 - rep_exp_2)) * c_6 * c_2

            term_4 = (rep_exp_1 - rep_exp_2) * c_1 * c_5 * c_6


            B = (-term_1 + term_2 - term_3
                 + term_4) / denominator


        elif gamma_0 + error_margin < self.gamma < 1:
            # Implementation of Case III in Appendix B.1.2.

            rep_phrase_1 = np.sqrt(c_1 * c_3 / c_2 ** 2)  # Repeated Phrase 1.
            rep_phrase_2 = np.sqrt(c_2 ** 2 - c_1 * c_3)  # Repeated Phrase 2.
            rep_phrase_3 = np.arctanh(rep_phrase_2 / c_2) - tau * rep_phrase_2  # Repeated Phrase 3.
            # arccoth(x) = arctanh(1/x)

            denominator = c_2 * rep_phrase_1 * (c_1 * c_3 + c_5 * (c_5 - 2 * c_2))  # Denominator in final equation.

            # The final equation for B is split into 5 terms.

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
            # Implementation of Case IV in Appendix B.1.2.

            rep_phrase_1 = np.sqrt(-c_1 * c_3 / disc)  # Repeated Phrase 1.
            rep_phrase_2 = 0.5 * np.sqrt(disc) * tau - np.arctanh(2 * c_2 / np.sqrt(disc))  # Repeated Phrase 2.

            denominator = 2 * c_1 * rep_phrase_1 * (c_1 * c_3 + c_5 * (c_5 - 2 * c_2))  # Denominator in final equation.

            # The final equation for B is split into 5 terms.

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

        index = ['Ticker of first stock', 'Ticker of second stock', 'Scaled Spread weights',
                 'long-term mean', 'rate of mean reversion', 'standard deviation', 'half-life']

        data = [self.ticker_A, self.ticker_B, np.round(self.scaled_spread_weights.values, 3),
                self.mu, self.k, self.sigma, self._calc_half_life(self.k)]

        # Combine data and indexes into the pandas Series
        output = pd.Series(data=data, index=index)

        return output
