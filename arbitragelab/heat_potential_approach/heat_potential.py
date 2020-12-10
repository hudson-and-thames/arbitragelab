# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

# pylint: disable=missing-module-docstring, invalid-name, too-many-locals, too-many-arguments
#import warnings
from typing import Callable
import numpy as np
#import matplotlib.pyplot as plt
import scipy.optimize as so
import pandas as pd



class HeatPotentials():
    """
    This class implements the algorithm for finding
    profit-taking and stop-loss levels for p/l
    that follow an Ornstein-Uhlenbeck process using the
    heat potential method. The following implementation is based on the work
    of Alexandr Lipton and Marcos Lopez de Prado "A closed-form solution for
    optimal mean-reverting trading strategies"<https://ssrn.com/abstract=3534445>`_
    """

    def __init__(self):

        self.theta = None
        self.optimal_profit = None
        self.optimal_stop_loss = None
        self.delta_grid = None
        self.sharpe = None
        self.max_trade_duration = None
        self.mu = None

    def fit(self, ou_params: list, delta_grid: float, max_trade_duration=None) -> None:
        """
        Fits the steady-state distribution to given OU model, assigns the
        grid density with respect to t, maximum duration of the trade.
        Calculates and assigns values for optimal stop-loss and profit-taking levels.
        :param ou_params: (list) parameters of the OU model. [theta, mu, sigma]
        :param delta_grid: Grid density with respect to t.
        :param max_trade_duration: maximum duration of the trade.
        """
        # theta, mu, sigma = ou.optimal_coefficients(ou_data)

        theta, self.mu, sigma = ou_params
        self.delta_grid = delta_grid

        if max_trade_duration is not None:
            self.max_trade_duration = self.mu * max_trade_duration

        self.theta = np.sqrt(self.mu) * theta / sigma

        profit_taking, stop_loss, max_sharpe = self.optimal_levels()

        self.optimal_profit = sigma * profit_taking / np.sqrt(self.mu)

        self.optimal_stop_loss = sigma * stop_loss / np.sqrt(self.mu)

        self.sharpe = max_sharpe

    def description(self) -> pd.Series:
        """
        Returns the statistics of the model. The optimal thresholds are converted back to the
        terms of the original data.

        :return: (pd.Series) Summary data for model parameters and optimal levels.
        """
        # Calculating the default data values
        data = [self.optimal_profit, self.optimal_stop_loss, self.max_trade_duration / self.mu]
        # Setting the names for the data indexes
        index = ['profit-taking threshold', 'stop-loss level', 'max duration of the trade']

        # Combine data and indexes into the pandas Series
        output = pd.Series(data=data, index=index)

        return output

    def v(self, max_trade_duration: float) -> np.ndarray:
        """
        Calculates the grid of v(t) functions for t in [0,max_trade_duration]
        (p.5 and p.8)

        :param max_trade_duration: (float) Maximum duration of the trade.

        :return: (np.array) Grid of v(t).
        """
        # Setting up the grid of (tau = max_trade_duration - t)
        tau = max_trade_duration - np.arange(0, max_trade_duration, self.delta_grid)

        # Calculating the v(tau) grid values and rearranging into ascending order
        output = ((1 - np.exp(-2 * tau)) / 2)[::-1]

        return output

    @staticmethod
    def upsilon(max_trade_duration: float) -> float:
        """
        Calculates the helper function that correspond to v(0).
        (p.5)

         :param max_trade_duration: (float) Maximum duration of the trade.

         :return: (float) Calculated value of v(0).
        """
        output = (1 - np.exp(-2 * max_trade_duration)) / 2

        return output

    def omega(self, max_trade_duration: float) -> float:
        """
        Calculates helper value for method of heat potentials.
        (p.5)

        :param max_trade_duration: (float) Maximum duration of the trade.

        :return: (float) The result of function calculation.
        """
        upsilon = self.upsilon(max_trade_duration)

        # Calculating the omega value
        output = -np.sqrt(1 - 2 * upsilon) * self.theta

        return output

    def _Pi_upper(self, v: np.ndarray, optimal_profit: float) -> np.ndarray:
        """
        Calculates helper function for representing optimal profit level
        in calculations of the Sharpe ratio. (p.5)

        :param v: (np.array) Grid of v(t) where t in [0,max_trade_duration] with step delta_grid.
        :param optimal_profit: Optimal profit-taking threshold.

        :return: (np.array) Array of values of the helper function with respect to grid elements.
        """

        output = np.sqrt(1 - 2 * v) * (optimal_profit - self.theta)

        return output

    def _Pi_lower(self, v: np.ndarray, optimal_stop_loss: float) -> np.ndarray:
        """
        Calculates helper function for representing optimal stop-loss level
        in calculations of the Sharpe ratio. (p.5)

        :param v: (np.array) Grid of v(t) where t in [0,max_trade_duration] with step delta_grid.
        :param optimal_stop_loss: (float) Optimal stop-loss level.

        :return: (np.array) Array of values of the helper function with respect to grid elements.
        """

        output = np.sqrt(1 - 2 * v) * (optimal_stop_loss - self.theta)

        return output

    def _heat_potential_helper(self, max_trade_duration: float,
                               optimal_profit: float,
                               optimal_stop_loss: float) -> np.ndarray:
        """
        Calculates the values of the helper functions for numerical root calculation.
        (p.6)

        :param max_trade_duration: (float) Maximum duration of the trade.
        :param optimal_profit: (float) Optimal profit-taking threshold.
        :param optimal_stop_loss: (float) Optimal stop-loss level.

        :return: (np.array) List of calculated values of helper functions for every
        element in grid v(t).
        """

        # Setting up a grid excluding the last element to exclude infinite values in the future
        v = self.v(max_trade_duration)[:-1]

        upsilon = self.upsilon(max_trade_duration)

        # Calculating the helper functions
        _Pi_upper = self._Pi_upper(v, optimal_profit)

        _Pi_lower = self._Pi_lower(v, optimal_stop_loss)

        # Calculating the helper values for numerical integral calculation
        e_lower = (2 * optimal_stop_loss / np.log((1 - 2 * v) / (1 - 2 * upsilon))
                   + 2 * (_Pi_lower + self.theta) / np.log(1 - 2 * upsilon))

        e_upper = (2 * optimal_profit / np.log((1 - 2 * v) / (1 - 2 * upsilon))
                   + 2 * (_Pi_upper + self.theta) / np.log(1 - 2 * upsilon))

        f_lower = (4 * optimal_stop_loss ** 2 / (np.log((1 - 2 * v) / (1 - 2 * upsilon))) ** 2
                   - 4 * (v + (_Pi_lower + self.theta) ** 2) / (np.log(1 - 2 * upsilon)) ** 2)

        f_upper = (4 * optimal_profit ** 2 / (np.log((1 - 2 * v) / (1 - 2 * upsilon))) ** 2
                   - 4 * (v + (_Pi_upper + self.theta) ** 2) / (np.log(1 - 2 * upsilon)) ** 2)

        return e_upper, e_lower, f_upper, f_lower

    def _numerical_calculation_helper(self, max_trade_duration: float,
                                      optimal_profit: float,
                                      optimal_stop_loss: float) -> np.ndarray:
        """
        Numerically calculates helping integral functions to solve
        the Volterra equations. Later it uses the acquired values to calculate epsilon
        and phi values necessary for Sharpe ratio calculation.
        (p.8)

        :param max_trade_duration: (float) Maximum duration of the trade.
        :param optimal_profit: (float) Optimal profit-taking threshold.
        :param optimal_stop_loss: (float) Optimal stop-loss level.

        :return: (np.array) List of calculated values of lower and upper epsilon and phi functions
        for every element in grid v(t).
        """

        # Setting up the set of lambda functions that represent the integrated expressions
        # K_1_1_v and K_2_2_v are approximations for the case v=s
        _Pi_upper = lambda v: self._Pi_upper(v, optimal_profit)

        _Pi_lower = lambda v: self._Pi_lower(v, optimal_stop_loss)

        K_1_1 = lambda v, s: ((1 / np.sqrt(2 * np.pi))
                              * (_Pi_lower(v) - _Pi_lower(s)) / (v - s)
                              * np.exp(-(_Pi_lower(v) - _Pi_lower(s)) ** 2
                                       / (2 * (v - s))))

        K_1_1_v = lambda v: ((self.theta - optimal_stop_loss)
                             / np.sqrt((2 * np.pi) * (1 - 2 * v)))

        K_1_2 = lambda v, s: ((1 / np.sqrt(2 * np.pi))
                              * (_Pi_lower(v) - _Pi_upper(s)) / ((v - s) ** 1.5)
                              * np.exp(-(_Pi_lower(v) - _Pi_upper(s)) ** 2
                                       / (2 * (v - s))))

        K_2_1 = lambda v, s: ((1 / np.sqrt(2 * np.pi))
                              * (_Pi_upper(v) - _Pi_lower(s)) / ((v - s) ** 1.5)
                              * np.exp(-(_Pi_upper(v) - _Pi_lower(s)) ** 2
                                       / (2 * (v - s))))

        K_2_2 = lambda v, s: ((1 / np.sqrt(2 * np.pi))
                              * (_Pi_upper(v) - _Pi_upper(s)) / (v - s)
                              * np.exp(-(_Pi_upper(v) - _Pi_upper(s)) ** 2
                                       / (2 * (v - s))))

        K_2_2_v = lambda v: ((self.theta - optimal_profit)
                             / (np.sqrt((2 * np.pi) * (1 - 2 * v))))

        # Setting up the grid values
        v = self.v(max_trade_duration)[:-1]

        # Calculating the helper values for Volterra equations
        e_l, e_u, f_l, f_u = self._heat_potential_helper(max_trade_duration, optimal_profit,
                                                         optimal_stop_loss)

        # Solving the two sets of Volterra equations
        eps_lower, eps_upper = self._numerical_calculation_equations(v, K_1_1, K_1_1_v, K_1_2,
                                                                     K_2_1, K_2_2, K_2_2_v,
                                                                     e_l, e_u)

        phi_lower, phi_upper = self._numerical_calculation_equations(v, K_1_1, K_1_1_v, K_1_2,
                                                                     K_2_1, K_2_2, K_2_2_v,
                                                                     f_l, f_u)

        return eps_lower, eps_upper, phi_lower, phi_upper

    @staticmethod
    def _numerical_calculation_equations(v: np.ndarray,
                                         K_11: Callable[[float], float],
                                         K_11_v: Callable[[float], float],
                                         K_12: Callable[[float], float],
                                         K_21: Callable[[float], float],
                                         K_22: Callable[[float], float],
                                         K_22_v: Callable[[float], float],
                                         f1: np.ndarray,
                                         f2: np.ndarray) -> tuple:
        """
        Numerically calculates general solution for system of Volterra integral equations.
        (p.8)

        :param v: (np.array) Grid of v(t) where t in [0,max_trade_duration] with step delta_grid.
        :param K_11: (function) Function that calculates the K_1_1(v,s).
        :param K_11_v: (function) Function that calculates the approximation of K_1_1(v,v).
        :param K_12: (function) Function that calculates the K_1_2(v,s).
        :param K_21: (function) Function that calculates the K_2_1(v,s).
        :param K_22: (function) Function that calculates the K_2_2(v,s).
        :param K_22_v: (function) Function that calculates the approximation of K_2_2(v,v).

        :return: (tuple) List of calculated solutions for two coupled system of Volterra
        integral equations for every element in grid v(t).
        """

        n = len(v)

        k = n

        # Defining the result arrays
        nu_1 = np.zeros(n)

        nu_2 = np.zeros(n)

        # Defining the first two values of array
        nu_1[0] = f1[0]

        nu_2[0] = -f2[0]

        nu_1[1] = f1[1] / (1 + K_11_v(v[1]) * np.sqrt(v[1]))

        nu_2[1] = -f2[1] / (1 - K_22_v(v[1]) * np.sqrt(v[1]))

        # Calculating the mavle of mutiplication term for both solutions
        nu_1_mult = (1 + K_11_v(v[2:k]) * np.sqrt(v[2:k] - v[1:k - 1])) ** -1

        nu_2_mult = (-1 + K_22_v(v[2:k]) * np.sqrt(v[2:k] - v[1:k - 1])) ** -1

        # Calculating the solutions for the Volterra equations for elements of
        # the v - grid using a trapezoidal rule
        for i in range(2, k):
            sum_1 = sum([((K_11(v[i], v[j]) * nu_1[j] + K_11(v[i], v[j - 1]) * nu_1[j - 1])
                          / (np.sqrt(v[i] - v[j]) + np.sqrt(v[i] - v[j - 1]))
                          + 0.5 * (K_12(v[i], v[j]) * nu_2[j] + K_12(v[i], v[j - 1]) * nu_2[j - 1]))
                         * (v[j] - v[j - 1])
                         for j in range(1, i - 1)])

            nu_1[i] = (nu_1_mult[i - 2]
                       * (f1[i]
                          - K_11(v[i], v[i - 1]) * nu_1[i - 1] * np.sqrt(v[i] - v[i - 1])
                          - 0.5 * K_12(v[i], v[i - 1]) * nu_2[i - 1] * (v[i] - v[i - 1])
                          - sum_1))

            sum_2 = sum([(0.5 * (K_21(v[i], v[j]) * nu_1[j] + K_21(v[i], v[j - 1]) * nu_1[j - 1])
                          + (K_22(v[i], v[j]) * nu_2[j] + K_22(v[i], v[j - 1]) * nu_2[j - 1])
                          / (np.sqrt(v[i] - v[j]) + np.sqrt(v[i] - v[j - 1])))
                         * (v[j] - v[j - 1])
                         for j in range(1, i - 1)])

            nu_2[i] = (nu_2_mult[i - 2]
                       * (f2[i]
                          - 0.5 * K_21(v[i], v[i - 1]) * nu_1[i - 1] * (v[i] - v[i - 1])
                          - K_22(v[i], v[i - 1]) * nu_2[i - 1] * np.sqrt(v[i] - v[i - 1])
                          - sum_2))

        return nu_1, nu_2

    def _sharpe_helper_functions(self, max_trade_duration: float, optimal_profit: float,
                                 optimal_stop_loss: float) -> np.ndarray:
        """
        Numerically calculates the main helper functions E and F for the Sharpe function
        calculation.
        (p.9)

        :param max_trade_duration: (float) Maximum duration of the trade.
        :param optimal_profit: (float) Optimal profit-taking threshold.
        :param optimal_stop_loss: (float) Optimal stop-loss level.

        :return: (np.array) List of calculated values of E and F functions for every
        element in grid v(t).
        """
        # Calculate the core helper values
        eps_lower, eps_upper,\
        phi_lower, phi_upper = self._numerical_calculation_helper(max_trade_duration,
                                                                  optimal_profit,
                                                                  optimal_stop_loss)
        # Setting up the grid
        v = self.v(max_trade_duration)

        n = len(v)

        # Calculating the weights for further calculation
        w_l = np.zeros(n - 1)

        w_u = np.zeros(n - 1)

        omega = self.omega(max_trade_duration)

        _Pi_lower = self._Pi_lower(v[1:-1], optimal_stop_loss)

        _Pi_upper = self._Pi_upper(v[1:-1], optimal_profit)

        w_l[:-1] = ((omega - _Pi_lower)
                    * np.exp(-(omega - _Pi_lower) ** 2
                             / (2 * (v[-1] - v[1:-1])))
                    / (np.sqrt(2 * np.pi) * (v[-1] - v[1:-1]) ** 1.5))

        w_l[-1] = 0

        w_u[:-1] = ((omega - _Pi_upper)
                    * np.exp(-(omega - _Pi_upper) ** 2
                             / (2 * (v[-1] - v[1:-1])))
                    / (np.sqrt(2 * np.pi) * (v[-1] - v[1:-1]) ** 1.5))

        w_u[-1] = 0

        k = n - 1

        # Setting up the vectors for pre-summed E and F values
        E_vect = np.zeros(k)
        F_vect = np.zeros(k)

        # Calculate the E and F functions
        for i in range(1, k):
            E_vect[i - 1] = ((w_l[i - 1] * eps_lower[i] + w_l[i - 2] * eps_lower[i - 1]
                              + w_l[i - 1] * eps_upper[i] +
                              w_l[i - 2] * eps_upper[i - 1]) * (v[i] - v[i - 1]))

        E = 0.5 * sum(E_vect)

        for i in range(1, k):
            F_vect[i - 1] = ((w_l[i - 1] * phi_lower[i] + w_l[i - 2] * phi_lower[i - 1]
                              + w_l[i - 1] * phi_upper[i] +
                              w_l[i - 2] * phi_upper[i - 1]) * (v[i] - v[i - 1]))

        F = 0.5 * sum(F_vect)

        return E, F

    def sharpe_calculation(self, max_trade_duration: float, optimal_profit: float,
                           optimal_stop_loss: float) -> np.ndarray:
        """
        Calculates the Sharpe ratio.
        (p.6 )

        :param max_trade_duration: (float) Maximum duration of the trade.
        :param optimal_profit: (float) Optimal profit-taking threshold.
        :param optimal_stop_loss: (float) Optimal stop-loss level.

        :return: (float) Sharpe ratio.
        """
        # Setting up the helper values
        E, F = self._sharpe_helper_functions(max_trade_duration, optimal_profit, optimal_stop_loss)

        upsilon = self.upsilon(max_trade_duration)

        omega = self.omega(max_trade_duration)

        a = 2 * (omega + self.theta) / np.log(1 - 2 * upsilon)

        summ_term = (4 * (upsilon + np.log(1 - 2 * upsilon) * (omega + self.theta) * E) /
                     (np.log(1 - 2 * upsilon)) ** 2)

        # Calculating the Sharpe ratio
        sharpe_ratio = (E - a) / np.sqrt(F - E ** 2 + summ_term)

        return sharpe_ratio

    def _neg_sharpe_calculation(self, params):
        """
        Calculates the negative Sharpe ratio for the optimization process
        (p.6)

        :param params: (np.array) Optimal profit-taking and stop-loss level.

        :return: (float) Negated Sharpe ratio.
        """

        max_trade_duration = self.max_trade_duration

        optimal_profit, optimal_stop_loss = params

        # Setting up the helper functions
        E, F = self._sharpe_helper_functions(max_trade_duration, optimal_profit, optimal_stop_loss)

        upsilon = self.upsilon(max_trade_duration)

        omega = self.omega(max_trade_duration)

        a = 2 * (omega + self.theta) / np.log(1 - 2 * upsilon)

        summ_term = (4 * (upsilon + np.log(1 - 2 * upsilon) * (omega + self.theta) * E) /\
                     (np.log(1 - 2 * upsilon)) ** 2)

        # Calculate the Sharpe ratio
        sharpe_ratio = (E - a) / np.sqrt(F - E ** 2 + summ_term)

        return -sharpe_ratio

    def optimal_levels(self) -> list:
        """
        Calculates optimal profit-taking and stop-loss levels by maximizing the Sharpe ratio

        :returns: (list) The list that consists of profit-taking level,
        stop-loss level, max Sharpe values.
        """

        # Set up the the initialization points
        stop_loss_guess = self.theta - 6 / np.sqrt(2)

        profit_taking_guess = self.theta + 6 / np.sqrt(2)

        # Setting bounds
        # max duration > 0, profit-taking level > 0, stop-loss < 0
        bounds = ((1e-5, None), (None, 1e-5))

        # Initial guesses for theta, mu, sigma
        initial_guess = np.array((profit_taking_guess, stop_loss_guess))

        result = so.minimize(self._neg_sharpe_calculation, initial_guess, bounds=bounds)

        # Unpacking optimal values
        profit_taking, stop_loss = result.x

        # Undo negation
        max_sharpe = -result.fun

        return profit_taking, stop_loss, max_sharpe
