# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=missing-module-docstring, invalid-name
import warnings
from scipy.optimize import root_scalar, fsolve, minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from arbitragelab.optimal_mean_reversion.ou_model import OrnsteinUhlenbeck
from arbitragelab.util import segment


class ExponentialOrnsteinUhlenbeck(OrnsteinUhlenbeck):
    """
    This class implements the algorithm for solving the optimal stopping problem
    and optimal switching problem for assets with mean-reverting tendencies based on the
    Exponential Ornstein-Uhlenbeck model mentioned in the following publication:
    'Tim Leung and Xin Li Optimal Mean reversion Trading: Mathematical Analysis and
    Practical Applications(November 26, 2015)'
    <https://www.amazon.com/Optimal-Mean-Reversion-Trading-Mathematical/dp/9814725919>`_
    """

    def __init__(self):
        """
        Initializes the module parameters.
        """

        super().__init__()
        self.a_tilde = None

        segment.track('ExponentialOrnsteinUhlenbeck')


    def fit(self, data: pd.DataFrame, data_frequency: str, discount_rate: tuple, transaction_cost: tuple,
            start: str = None, end: str = None, stop_loss: float = None):
        """
        Fits the Exponential Ornstein-Uhlenbeck model to given data and assigns the discount rates,
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
        Fits the Exponential Ornstein-Uhlenbeck model to time series for portfolio prices.

        :param data: (np.array) All given prices of two assets to construct a portfolio from.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        """

        # Using this structure to rewrite the method docstring
        OrnsteinUhlenbeck.fit_to_portfolio(self, data, start, end)

    def fit_to_assets(self, data: np.array = None, start: str = None, end: str = None):
        """
        Creates the optimal portfolio in terms of the Exponential Ornstein-Uhlenbeck model
        from two given time series for asset prices and fits the values of the model's parameters. (p.13)

        :param data: (np.array) All given prices of two assets to construct a portfolio from.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        """

        # Using this structure to rewrite the method docstring
        OrnsteinUhlenbeck.fit_to_assets(self, data, start, end)

    def xou_model_simulation(self, n: int, theta_given: float = None, mu_given: float = None,
                             sigma_given: float = None, delta_t_given: float = None) -> np.array:
        """
        Simulates values of an XOU process with given parameters or parameters fitted to our data.

        :param n: (int) Number of simulated values.
        :param theta_given: (float) Long-term mean.
        :param mu_given: (float) Mean reversion speed.
        :param sigma_given: (float) The amplitude of randomness in the system.
        :param delta_t_given: (float) Delta between observations, calculated in years.
        :return: (np.array) Simulated portfolio prices.
        """

        # Check if the parameters were assigned
        if all(param is not None for param in
               [theta_given, mu_given, sigma_given, delta_t_given]):
            # If parameters were assigned simulate the XOU process with given parameters
            output = np.exp(self.ou_model_simulation(n=n, theta_given=theta_given, mu_given=mu_given,
                                                     sigma_given=sigma_given, delta_t_given=delta_t_given))
        else:
            # If parameters were not assigned simulate the process using the parameters of the fitted model
            output = np.exp(self.ou_model_simulation(n=n))

        return output

    def _check_xou(self) -> bool:
        """
        Performs the check necessary for the existence of the solutions of the optimal problems.

        :return: (bool) Boolean value that corresponds to the check results.
        """

        # Setting bounds for the x variable
        bounds = ((None, None),)

        # Setting the negated value of goal function because scipy.optimize
        # doesn't have maximization function
        func = lambda x: -(self.V_XOU(x) - np.exp(x) - self.c[1])

        # Initial guesses for value of x
        initial_guess = (self.theta - np.sqrt(self.sigma_square))

        # Minimization of the negated function to maximize goal function
        result = minimize(func, initial_guess, bounds=bounds)

        # Testing the condition
        output = -result.fun > 0

        return output

    def V_XOU(self, price: float) -> float:
        """
        Calculates the expected discounted value of liquidation of the position. (p.54)

        :param price: (float) Portfolio value.
        :return: (float) Expected discounted liquidation value.
        """

        # Getting optimal liquidation level
        liquidation_level = self.xou_optimal_liquidation_level()

        # Value of the V function
        if price < liquidation_level:
            output = ((np.exp(liquidation_level) - self.c[0])
                      * self._F(price, self.r[0])
                      / self._F(liquidation_level, self.r[0]))
        else:
            output = np.exp(price) - self.c[0]

        return output

    def _V_XOU_derivative(self, price: float, h: float = 1e-4) -> float:
        """
        Calculates the derivative of the expected discounted value of
        liquidation of the position.

        :param price: (float) Portfolio value.
        :param h: (float) Delta step to use to calculate derivative.
        :return: (float) Value of V derivative function.
        """

        # Numerically calculating the derivative
        output = (self.V_XOU(price + h) - self.V_XOU(price)) / h

        return output

    def xou_optimal_liquidation_level(self) -> float:
        """
        Calculates the optimal liquidation portfolio level. (p.54)

        :return: (float) Optimal liquidation portfolio level.
        """

        # If the liquidation level wasn't calculated before, setting it
        if self.liquidation_level[0] is None:

            # Setting the equation to be solved
            equation = lambda price: (np.exp(price) * self._F(price, self.r[0]) - (np.exp(price) - self.c[0])
                                      * self._F_derivative(price, self.r[0]))

            # Setting the bounds for the root-finding
            bracket = [self.theta - 6 * np.sqrt(self.sigma_square), self.theta + 6 * np.sqrt(self.sigma_square)]

            sol = root_scalar(equation, bracket=bracket)

            output = sol.root

            self.liquidation_level[0] = output

        # If was pre-calculated, using it
        else:

            output = self.liquidation_level[0]

        return output

    def xou_optimal_entry_interval(self) -> tuple:
        """
        Calculates the optimal entry interval for the portfolio price. (p.35)

        :return: (tuple) Optimal entry interval.
        """

        # Checking for the necessary condition
        if not self._check_xou():  # pragma: no cover
            raise Exception("There is no optimal solution for your data")

        # If the entry level wasn't calculated before, set it
        if self.entry_level[0] is None:

            equation1 = lambda price: (self._F(price, self.r[1]) *
                                       (self._V_XOU_derivative(price) - np.exp(price))
                                       - self._F_derivative(price, self.r[1])
                                       * (self.V_XOU(price) - np.exp(price) - self.c[1]))

            equation2 = lambda price: (self._G(price, self.r[1])
                                       * (self._V_XOU_derivative(price) - np.exp(price))
                                       - self._G_derivative(price, self.r[1])
                                       * (self.V_XOU(price) - np.exp(price) - self.c[1]))

            # Set the liquidation level to previously calculated value
            b = self.xou_optimal_liquidation_level()

            bracket = [self.theta - 6 * np.sqrt(self.sigma_square), b]

            # Solving the equations
            sol2 = root_scalar(equation2, bracket=bracket)

            sol1 = root_scalar(equation1, bracket=[self.theta - 100 * np.sqrt(self.sigma_square), sol2.root])

            output = [round(sol1.root, 5), round(sol2.root, 5)]

            self.entry_level[0] = output

        else:

            output = self.entry_level[0]

        return output

    def _fb(self, price: float) -> float:
        """
        Helper function for solving the optimal switching problem regarding
        the entry level.

        :param price: (float) Price of an asset.
        :return: (float) Value of the helper function calculated with respect to given asset price.
        """

        output = ((self.mu * self.theta + 0.5 * self.sigma_square - self.r[0])
                  - self.mu * price
                  - self.r[0] * self.c[1] * np.exp(-price))
        return output

    def _fb_root(self) -> list:
        """
        Calculates the root(s) for _fb function or returns the False if it doesn't exist.

        :return: (list/bool) The root(s) of _fb equation or False flag if the solution doesn't exist.
        """

        a = self.xou_optimal_entry_interval()[0]

        sol = fsolve(self._fb, np.array([a - 6 * np.sqrt(self.sigma_square), -a + 6 * np.sqrt(self.sigma_square)]))

        # Calculating the root
        output = sol

        return output

    def _condition_optimal_switching_a(self) -> bool:
        """
        Checks the condition of optimality of re-entering the market. If a_tilde exists then finds it.

        :return: (bool) Whether the condition is satisfied or not.
        """

        # Setting the initial flag value
        output = True

        try:
            # Checking if the solution exists
            b = list(self._fb_root())
        except ValueError:
            output = False

        if output:
            # Setting up the equation
            equation = lambda x: (self._F(x, self.r[0]) * np.exp(x)
                                  - self._F_derivative(x, self.r[0]) * (np.exp(x) + self.c[1]))
            try:
                # Solving the equation
                self.a_tilde = root_scalar(equation, bracket=b).root
            except ValueError:
                # If the solution doesn't exist set the flag to be False
                output = False

        return output

    def _condition_optimal_switching_inequality(self) -> bool:
        """
        Ensures that inequality that is required for the optimality of re-entering the market is true.

        :return: (bool) Result of the inequality check.
        """

        output = False

        a = self._condition_optimal_switching_a()
        b = self.xou_optimal_liquidation_level()

        if a and ((np.exp(self.a_tilde) + self.c[1])
                  / self._F(self.a_tilde, self.r[0])
                  < (np.exp(b) + self.c[1])
                  / self._F(b, self.r[0])):
            output = True
        return output

    def _check_optimal_switching(self) -> bool:
        """
        Checks if all the conditions are satisfied for re-entering the market to be optimal.

        :return: (bool) Whether the conditions were satisfied or not.
        """

        output = True

        try:
            # Calculating the roots for _fb

            x_b = self._fb_root()
        except ValueError:
            output = False

        # Checking if the two existing roots are distinct and other two conditions
        if not (output and (x_b[0] != x_b[1]) and self._condition_optimal_switching_inequality()):
            # Checking if all conditions are satisfied
            output = False

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

        output = [((np.exp(x[0]) * G(x[0]) - (np.exp(x[0]) + self.c[1]) * G_d(x[0]))
                   / (F_d(x[0]) * G(x[0]) - F(x[0]) * G_d(x[0]))
                   - (np.exp(x[1]) * G(x[1]) - (np.exp(x[1]) - self.c[0]) * G_d(x[1]))
                   / (F_d(x[1]) * G(x[1]) - F(x[1]) * G_d(x[1]))),
                  ((np.exp(x[0]) * F(x[0]) - (np.exp(x[0]) + self.c[1]) * F_d(x[0]))
                   / (F_d(x[0]) * G(x[0]) - F(x[0]) * G_d(x[0]))
                   - (np.exp(x[1]) * F(x[1]) - (np.exp(x[1]) - self.c[0]) * F_d(x[1]))
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
                output = [None, self.xou_optimal_liquidation_level()]

            else:
                # Setting the boundaries
                lower_b = self.xou_optimal_entry_interval()[1]
                upper_b = self.xou_optimal_liquidation_level()
                # Solving the equations
                output = fsolve(self._optimal_switching_equations, [lower_b, upper_b])

                self.entry_level[1] = [self.a_tilde, output[0]]
                self.liquidation_level[1] = output[1]
        else:
            output = np.array([self.entry_level[1][1], self.liquidation_level[1]])

        return output

    def xou_plot_levels(self, data: pd.DataFrame, switching: bool = False) -> Figure:
        """
        Plots the found optimal exit and entry levels on the graph
        alongside with the given data.

        :param data: (np.array/pd.DataFrame) Time series of portfolio prices.
        :param switching: (bool) A flag whether to take stop-loss level into account.
            when showcasing the results.
        :return: (plt.Figure) Figure with optimal exit and entry levels.
        """

        portfolio = data

        if switching:
            # Plotting entry and exit levels calculated by default
            fig = plt.figure()
            plt.plot(portfolio, label='portfolio price', color="#023778")
            plt.axhline(np.exp(self.xou_optimal_liquidation_level()), label="optimal liquidation level",
                        linestyle='-', color='#FE5F55')
            plt.axhline(np.exp(self.xou_optimal_entry_interval()[1]), label="optimal entry level",
                        linestyle='-', color='#9FC490')
            plt.axhline(np.exp(self.optimal_switching_levels()[1]), label="optimal liquidation level for switching",
                        linestyle='--', color='#FE5F55')
            plt.axhline(np.exp(self.optimal_switching_levels()[0]), label="optimal entry level for switching",
                        linestyle='--', color='#C0DFA1')
            plt.legend()
            plt.title('Default optimal levels and optimal switching')

        else:
            fig = plt.figure()
            plt.plot(portfolio, label='portfolio price', color="#023778")
            plt.axhline(np.exp(self.xou_optimal_liquidation_level()), label="optimal liquidation level",
                        linestyle='-', color='#FE5F55')
            plt.axhline(np.exp(self.xou_optimal_entry_interval()[0]), label="optimal entry level",
                        linestyle='-', color='#9FC490')
            plt.axhline(np.exp(self.xou_optimal_entry_interval()[1]), label="optimal entry level",
                        linestyle=':', color='#9FC490')
            plt.legend()
            plt.title('Default optimal levels')

        return fig

    def xou_description(self, switching: bool = False) -> pd.Series:
        """
        Returns all the general parameters of the model, training interval timestamps if provided,
        the goodness of fit, allocated trading costs and discount rates, which stands for the optimal
        ratio between two assets in the created portfolio, default optimal levels calculated.
        If re-entering the market is optimal shows optimal switching levels.

        :param switching: (bool) Flag that signals whether to output switching data.
        :return: (pd.Series) Summary data for all model parameters and optimal levels.
        """

        # Calling _fit_data to create a helper variable
        portfolio = self._fit_data(self.training_period[0], self.training_period[1]).transpose()

        # Calculate the average simulated max log-likelihood
        simulated_mll = np.array(
            [self.optimal_coefficients(self.ou_model_simulation(portfolio.shape[0]))[3] for i in range(100)])

        # Calculating the default data values
        data = [self.training_period, self.theta, self.mu, np.sqrt(self.sigma_square), self.mll - simulated_mll.mean(),
                self.r, self.c, self.B_value,
                [round(np.exp(self.xou_optimal_entry_interval()[0]), 5), round(np.exp(self.xou_optimal_entry_interval()[1]), 5)],
                np.exp(self.optimal_liquidation_level())]
        # Setting the names for the data indexes
        index = ['training period', 'long-term mean', 'speed of reversion', 'volatility', 'fitting error',
                 'discount rates', 'transaction costs', 'beta',
                 'optimal entry level', 'optimal liquidation level']

        # If re-entering the position is optimal - account for additional values
        if switching:
            data.extend([[round(np.exp(self.a_tilde), 5), round(np.exp(self.optimal_switching_levels()[0]), 5)],
                         round(np.exp(self.optimal_switching_levels()[1]), 5)])
            index.extend(['optimal switching entry interval',
                          'optimal switching liquidation level'])

        # Combine data and indexes into the pandas Series
        output = pd.Series(data=data, index=index)

        return output
