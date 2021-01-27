# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=missing-module-docstring, invalid-name, no-name-in-module

import warnings
import numpy as np
from scipy.optimize import root_scalar, fsolve
from scipy.special import ive, hyp1f1, gamma
import matplotlib.pyplot as plt
import scipy.optimize as so
import pandas as pd
from matplotlib.figure import Figure

from arbitragelab.optimal_mean_reversion.ou_model import OrnsteinUhlenbeck
from arbitragelab.util import devadarsh


class CoxIngersollRoss(OrnsteinUhlenbeck):
    """
    This class implements the algorithms for optimal stopping and optimal switching problems for
    assets with mean-reverting tendencies based on the Cox-Ingersoll-Ross model mentioned in the
    following publication:'Tim Leung and Xin Li Optimal Mean reversion Trading: Mathematical Analysis
    and Practical Applications(November 26, 2015)'
    <https://www.amazon.com/Optimal-Mean-Reversion-Trading-Mathematical/dp/9814725919>`_

    Constructing a portfolio with mean-reverting properties is usually attempted by
    simultaneously taking a position in two highly correlated or co-moving assets and is
    labeled as "pairs trading". One of the most important problems faced by investors is
    to determine when to open and close a position.

    Optimal stopping problem is established on the premise of maximizing the output of
    one optimal enter-exit pair of trades, optimal switching, on the other hand, aims
    to maximize the cumulative reward of an infinite number of trades made at respective optimal levels.

    To find the liquidation and entry price levels for both optimal switching and optimal stopping
    for each approach we formulate an optimal double-stopping problem that gives the optimal entry
    and exit level rules.
    """

    def __init__(self):
        """
        Initializes the module
        """

        super().__init__()
        self.B_value = None
        devadarsh.track('CoxIngersollRoss')

    def fit(self, data: pd.DataFrame, data_frequency: str, discount_rate: tuple, transaction_cost: tuple,
            start: str = None, end: str = None, stop_loss: float = None):
        """
        Fits the Cox-Ingersoll-Ross model to given data and assigns the discount rates,
        transaction costs and stop-loss level for further exit or entry-level calculation.

        :param data: (np.array/pd.DataFrame) An array with time series of portfolio prices / An array with
            time series of of two assets prices. The dimensions should be either nx1 or nx2.
        :param data_frequency: (str) Data frequency ["D" - daily, "M" - monthly, "Y" - yearly].
        :param discount_rate: (float/tuple) A discount rate either for both entry and exit time
            or a list/tuple of discount rates with exit rate and entry rate in respective order.
        :param transaction_cost: (float/tuple) A transaction cost either for both entry and exit time
            or a list/tuple of transaction costs with exit cost and entry cost in respective order.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        :param stop_loss: (float/int) A stop-loss level - the position is assumed to be closed
            immediately upon reaching this pre-defined price level.
        """

        # Using this structure to rewrite the method docstring
        OrnsteinUhlenbeck.fit(self, data, data_frequency, discount_rate, transaction_cost, start,
                              end, stop_loss)

    def fit_to_portfolio(self, data: np.array = None, start: str = None, end: str = None):
        """
        Fits the Cox-Ingersoll-Ross model to time series for portfolio prices.

        :param data: (np.array) All given prices of two assets to construct a portfolio from.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        """

        # Using this structure to rewrite the method docstring
        OrnsteinUhlenbeck.fit_to_portfolio(self, data, start, end)

    def fit_to_assets(self, data: np.array = None, start: str = None, end: str = None):
        """
        Creates the optimal portfolio in terms of the Cox-Ingersoll-Ross model
        from two given time series for asset prices and fits the values of the model's parameters. (p.13)

        :param data: (np.array) All given prices of two assets to construct a portfolio from.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        """

        # Using this structure to rewrite the method docstring
        OrnsteinUhlenbeck.fit_to_assets(self, data, start, end)

    def cir_model_simulation(self, n: int, theta_given: float = None, mu_given: float = None,
                             sigma_given: float = None, delta_t_given: float = None) -> np.array:
        """
        Simulates values of a CIR process with given parameters or parameters fitted to our data.

        :param n: (int) Number of simulated values.
        :param theta_given: (float) Long-term mean.
        :param mu_given: (float) Mean reversion speed.
        :param sigma_given: (float) The amplitude of randomness in the system.
        :param delta_t_given: (float) Delta between observations, calculated in years.
        :return: (np.array) simulated portfolio prices.
        """

        # Initializing the variable for process values
        x = np.zeros(n)

        # Checking whether to use given parameters or parameters of the fitted model

        # Use given data parameters
        if all(param is not None for param in
               [theta_given, mu_given, sigma_given, delta_t_given]):
            x[0] = theta_given
            theta = theta_given
            mu = mu_given
            sigma = sigma_given
            delta_t = delta_t_given

        else:  # Use fitted data parameters
            x[0] = self.theta
            theta = self.theta
            mu = self.mu
            sigma = np.sqrt(self.sigma_square)
            delta_t = self.delta_t

        # Simulating the CIR process values
        for i in range(n - 1):
            x[i + 1] = (x[i] + mu * (theta - x[i]) * delta_t
                        + sigma * np.sqrt(delta_t * x[i]) * np.random.randn())

        return x

    @staticmethod
    def _compute_log_likelihood(params: tuple, *args: tuple) -> float:
        """
        Computes the average Log Likelihood.

        From Borodin and Salminen (2002).

        :param params: (tuple) A tuple of three elements representing theta, mu and sigma_squared.
        :param args: (tuple) All other values that are passed to self._compute_log_likelihood()
        :return: (float) The average log-likelihood from given parameters.
        """

        # Setting given parameters
        theta, mu, sigma_squared = params
        Y, dt = args
        n = len(Y)

        # Calculating log likelihood
        q = (2 * mu * theta) / sigma_squared - 1

        c = (2 * mu) / (sigma_squared * (1 - np.exp(-mu * dt)))

        u = c * Y[:-1] * np.exp(-mu * dt)

        v = c * Y[1:]

        z = 2 * np.sqrt(u * v)

        log_likelihood = (n - 1) * np.log(c) - sum(u + v - 0.5 * q * np.log(v / u) - np.log(ive(q, z)) - z)

        return -log_likelihood

    def optimal_coefficients(self, portfolio: np.array) -> tuple:
        """
        Finds the optimal CIR model coefficients depending
        on the portfolio prices time series given. (p.13)

        :param portfolio: (np.array) Portfolio prices.
        :return: (tuple) Optimal parameters (theta, mu, sigma_square, max_LL).
        """

        # Setting bounds
        # Theta  R, mu > 0, sigma_squared > 0
        bounds = ((1e-5, None), (1e-5, None), (1e-5, None))

        theta_init = np.mean(portfolio)

        # Initial guesses for theta, mu, sigma
        initial_guess = np.array((theta_init, 1e-3, 1e-3))

        result = so.minimize(self._compute_log_likelihood, initial_guess,
                             args=(portfolio, self.delta_t), bounds=bounds)

        # Unpacking optimal values
        theta, mu, sigma_square = result.x

        # Undo negation
        max_log_likelihood = -result.fun

        return theta, mu, sigma_square, max_log_likelihood

    def _F(self, price: float, rate: float) -> float:
        """
        Calculates helper function to further define the exit/enter level. (p.84)

        :param price: (float) Portfolio price.
        :param rate: (float) Discounting rate.
        :return: (float) Value of F function.
        """

        # Setting up the function variables
        a = rate / self.mu

        b = (2 * self.mu * self.theta) / self.sigma_square

        z = (2 * self.mu * price) / self.sigma_square

        # Calculating the Kummer's function
        output = hyp1f1(a, b, z)

        return output

    def _G(self, price: float, rate: float) -> float:
        """
        Calculates helper function to further define the exit/enter level. (p.84)

        :param price: (float) Portfolio price.
        :param rate: (float) Discounting rate.
        :return: (float) Value of G function.
        """

        # Setting up the function variables
        a = rate / self.mu

        b = (2 * self.mu * self.theta) / self.sigma_square

        z = (2 * self.mu * price) / self.sigma_square

        # Calculating the Tricomi's function
        output = ((gamma(1 - b) * hyp1f1(a, b, z)) / gamma(a - b + 1)
                  + (gamma(b - 1) * hyp1f1(a - b + 1, 2 - b, z) * (z ** (1 - b))) / gamma(a))

        return output

    def optimal_liquidation_level(self) -> float:
        """
        Calculates the optimal liquidation portfolio level. (p.85)

        :return: (float) Optimal liquidation portfolio level.
        """

        # If the liquidation level wasn't calculated before, setting it
        if self.liquidation_level[0] is None:

            equation = lambda price: (self._F(price, self.r[0]) - (price - self.c[0])
                                      * self._F_derivative(price, self.r[0]))

            bracket = [0, self.theta + 6 * np.sqrt(self.sigma_square)]

            sol = root_scalar(equation, bracket=bracket)

            output = sol.root

            self.liquidation_level[0] = output

        # If was pre-calculated, using it
        else:

            output = self.liquidation_level[0]

        return output

    def optimal_entry_level(self) -> float:
        """
        Calculates the optimal entry portfolio level. (p.86)

        :return: (float) Optimal entry portfolio level.
        """

        # If the entry level wasn't calculated before, setting it
        if self.entry_level[0] is None:

            equation = lambda price: (self._G(price, self.r[1])
                                      * (self._V_derivative(price) - 1)
                                      - self._G_derivative(price, self.r[1])
                                      * (self.V(price) - price - self.c[1]))

            bracket = [1e-4,
                       self.optimal_liquidation_level()]

            sol = root_scalar(equation, bracket=bracket)

            output = sol.root

            self.entry_level[0] = output

        # If was pre-calculated, using it
        else:

            output = self.entry_level[0]

        return output

    def _critical_constants(self) -> list:
        """
        Calculates critical constants for the CIR model.

        :return: (list) Critical constants of the CIR model.
        """
        # Calculating the critical constants for exit and entry
        output = [(self.mu * self.theta + self.r[0] * self.c[0]) / (self.mu + self.r[0]),
                  (self.mu * self.theta - self.r[1] * self.c[1]) / (self.mu + self.r[1])]

        return output

    def _check_optimal_switching(self) -> bool:
        """
        Checks if it is optimal to re-enter the market.

        :return: (bool) The result of the test.
        """

        # Calculating the optimal liquidation level
        b = self.optimal_liquidation_level()

        # Calculating the optimal entry critical constant
        y_b = self._critical_constants()[1]

        output = False

        # Performing the test
        if y_b > 0 and self.c[1] < (b - self.c[0]) / self._F(b, self.r[0]):
            output = True

        return output

    def _optimal_switching_equations(self, x: list) -> list:
        """
        Defines the system of equations needed to calculate the optimal entry and exit levels for
        the optimal switching problem.

        :param x: (list) A list of two variables.
        :return: (list) A list of values of two equations.
        """

        G = lambda x: self._G(x, self.r[0])

        G_d = lambda x: self._G_derivative(x, self.r[0])

        F = lambda x: self._F(x, self.r[0])

        F_d = lambda x: self._F_derivative(x, self.r[0])

        output = [((G(x[0]) - (x[0] + self.c[1]) * G_d(x[0]))
                   / (F_d(x[0]) * G(x[0]) - F(x[0]) * G_d(x[0]))
                   - ((G(x[1]) - (x[1] - self.c[0]) * G_d(x[1]))
                      / (F_d(x[1]) * G(x[1]) - F(x[1]) * G_d(x[1])))),
                  ((F(x[0]) - (x[0] + self.c[1]) * F_d(x[0]))
                   / (F_d(x[0]) * G(x[0]) - F(x[0]) * G_d(x[0]))
                   - (F(x[1]) - (x[1] - self.c[0]) * F_d(x[1]))
                   / (F_d(x[1]) * G(x[1]) - F(x[1]) * G_d(x[1])))]

        return output

    def optimal_switching_levels(self) -> np.array:
        """
        Calculates the optimal switching levels.

        :return: (np.array) Optimal switching levels.
        """

        # Checks if the optimal levels were calculated before
        if self.entry_level[1] is None and self.liquidation_level[1] is None:
            # Checking if all the conditions are satisfied
            if not self._check_optimal_switching():
                warnings.warn("It is not optimal to enter the market")
                output = [None, self.optimal_liquidation_level()]

            else:

                # Setting the boundaries
                upper_bound, lower_bound = [self.optimal_liquidation_level(), self.optimal_entry_level()]

                # Solving the equations
                output = fsolve(self._optimal_switching_equations, [lower_bound, upper_bound])

                self.entry_level[1] = output[0]
                self.liquidation_level[1] = output[1]
        else:
            output = np.array([self.entry_level[1], self.liquidation_level[1]])

        return output

    def cir_plot_levels(self, data: pd.DataFrame, switching: bool = False) -> Figure:
        """
        Plots the found optimal exit and entry levels on the graph
        alongside with the given data.

        :param data: (np.array/pd.DataFrame) Time series of portfolio prices.
        :param switching: (bool) A flag whether to take stop-loss level into account when showcasing the results.
        :return: (plt.Figure) Figure with optimal exit and entry levels.
        """

        portfolio = data

        if switching:
            # Plotting entry and exit levels calculated by default
            fig = plt.figure()
            plt.plot(portfolio, label='portfolio price', color="#023778")
            plt.axhline(self.optimal_liquidation_level(), label="optimal liquidation level",
                        linestyle='-', color='#FE5F55')
            plt.axhline(self.optimal_entry_level(), label="optimal entry level",
                        linestyle='-', color='#9FC490')
            plt.axhline(self.optimal_switching_levels()[1], label="optimal liquidation level for switching",
                        linestyle='--', color='#FE5F55')
            plt.axhline(self.optimal_switching_levels()[0], label="optimal entry level for switching",
                        linestyle='--', color='#C0DFA1')
            plt.legend()
            plt.title('Default optimal levels and optimal switching')

        else:
            fig = plt.figure()
            plt.plot(portfolio, label='portfolio price', color="#023778")
            plt.axhline(self.optimal_liquidation_level(), label="optimal liquidation level",
                        linestyle='-', color='#FE5F55')
            plt.axhline(self.optimal_entry_level(), label="optimal entry level",
                        linestyle='-', color='#9FC490')
            plt.legend()
            plt.title('Default optimal levels')

        return fig

    def cir_description(self, switching: bool = False) -> pd.Series:
        """
        Returns all the general parameters of the model, training interval timestamps if provided,
        the goodness of fit, allocated trading costs and discount rates, which stands for the optimal
        ratio between two assets in the created portfolio, default optimal levels calculated.
        If re-entering the market is optimal shows optimal switching levels.

        :param switching: (bool) Flag that signals whether to output switching data.
        :return: (pd.Series) Summary data for all model parameters and optimal levels.
        """

        # Calculating the default data values
        data = [self.training_period, self.theta, self.mu, np.sqrt(self.sigma_square),
                self.r, self.c, self.B_value,
                self.optimal_entry_level(), self.optimal_liquidation_level()]
        # Setting the names for the data indexes
        index = ['training period', 'long-term mean', 'speed of reversion', 'volatility',
                 'discount rates', 'transaction costs', 'beta',
                 'optimal entry level', 'optimal liquidation level']

        # If re-entering the position is optimal - account for additional values
        if switching:
            data.extend([[round(self.optimal_switching_levels()[1], 5)],
                         round(self.optimal_switching_levels()[0], 5)])
            index.extend(['optimal switching entry level',
                          'optimal switching liquidation level'])

        # Combine data and indexes into the pandas Series
        output = pd.Series(data=data, index=index)

        return output
