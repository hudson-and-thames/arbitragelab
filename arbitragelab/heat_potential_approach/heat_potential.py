# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

# pylint: disable=missing-module-docstring, invalid-name, too-many-instance-attributes
import warnings
import numpy as np
#import matplotlib.pyplot as plt
import scipy.optimize as so
#from arbitragelab.optimal_mean_reversion.ou_model import OrnsteinUhlenbeck as ou


class Heat_potentials():
    """
    """

    def __init__(self):

        self.theta = None
        self.optimal_profit = None
        self.optimal_stop_loss = None
        self.delta_grid = None
        self.sharpe = None
        self.T = None

    def fit(self, ou_params, delta_grid, T=None):
        """
        """
        # theta, mu, sigma = ou.optimal_coefficients(ou_data)

        theta, mu, sigma = ou_params
        self.delta_grid = delta_grid

        if T is not None:
            self.T = mu * T

        self.theta = np.sqrt(mu) * theta / sigma

        profit_taking, stop_loss, max_sharpe = self.optimal_levels()

        self.optimal_profit = sigma * profit_taking / np.sqrt(mu)

        self.optimal_stop_loss = sigma * profit_taking / np.sqrt(mu)

    def description(self):
        """
        """
        # Calculating the default data values
        data = [self.optimal_profit, self.optimal_stop_loss, self.T / mu]
        # Setting the names for the data indexes
        index = ['profit-taking threshold', 'stop-loss level', 'max duration of the trade']

        # Combine data and indexes into the pandas Series
        output = pd.Series(data=data, index=index)

        return output

    def v(self, T: float):
        """
        p.5 upper
        """

        tau = T - np.arange(0, T, self.delta_grid)

        output = ((1 - np.exp(-2 * tau)) / 2)[::-1]

        return output

    def ksi(self, T: float):
        """
        """
        tau = T - np.arange(0, T, 1)

        output = np.exp(-tau)(x - self.theta)

        return output

    def gamma(self, T: float):
        """
        p.5 lower
        """
        output = (1 - np.exp(-2 * T)) / 2

        return output

    def omega(self, T: float):
        """
        p.5 lower
        """
        gamma = self.gamma(T)

        output = -np.sqrt(1 - 2 * gamma) * self.theta

        return output

    def Pi_upper(self, v: np.ndarray, optimal_profit: float):
        """
        p.5 lower
        """

        output = np.sqrt(1 - 2 * v) * (optimal_profit - self.theta)

        return output

    def Pi_lower(self, v: np.ndarray, optimal_stop_loss: float):
        """
        p.5 lower
        """

        output = np.sqrt(1 - 2 * v) * (optimal_stop_loss - self.theta)

        return output

    def heat_potential_helper(self, T: float, optimal_profit: float, optimal_stop_loss: float):
        """
        p.6 middle
        """

        v = self.v(T)[:-1]

        gamma = self.gamma(T)

        Pi_upper = self.Pi_upper(v, optimal_profit)

        Pi_lower = self.Pi_lower(v, optimal_stop_loss)

        e_lower = (2 * optimal_stop_loss / np.log((1 - 2 * v) / (1 - 2 * gamma))
                   + 2 * (Pi_lower + self.theta) / np.log(1 - 2 * gamma))

        e_upper = (2 * optimal_profit / np.log((1 - 2 * v) / (1 - 2 * gamma))
                   + 2 * (Pi_upper + self.theta) / np.log(1 - 2 * gamma))

        f_lower = (4 * optimal_stop_loss ** 2 / (np.log((1 - 2 * v) / (1 - 2 * gamma))) ** 2
                   - 4 * (v + (Pi_lower + self.theta) ** 2) / (np.log(1 - 2 * gamma)) ** 2)

        f_upper = (4 * optimal_profit ** 2 / (np.log((1 - 2 * v) / (1 - 2 * gamma))) ** 2
                   - 4 * (v + (Pi_upper + self.theta) ** 2) / (np.log(1 - 2 * gamma)) ** 2)

        return e_upper, e_lower, f_upper, f_lower

    def numerical_calculation_helper(self, T: float, optimal_profit: float, optimal_stop_loss: float):
        """
        p.8 upper
        """
        Pi_upper = lambda v: self.Pi_upper(v, optimal_profit)

        Pi_lower = lambda v: self.Pi_lower(v, optimal_stop_loss)

        K_1_1 = lambda v, s: ((1 / np.sqrt(2 * np.pi))
                              * (Pi_lower(v) - Pi_lower(s)) / (v - s)
                              * np.exp(-(Pi_lower(v) - Pi_lower(s)) ** 2
                                       / (2 * (v - s))))

        K_1_1_v = lambda v: ((self.theta - optimal_stop_loss)
                             / np.sqrt((2 * np.pi) * (1 - 2 * v)))

        K_1_2 = lambda v, s: ((1 / np.sqrt(2 * np.pi))
                              * (Pi_lower(v) - Pi_upper(s)) / ((v - s) ** 1.5)
                              * np.exp(-(Pi_lower(v) - Pi_upper(s)) ** 2
                                       / (2 * (v - s))))

        K_2_1 = lambda v, s: ((1 / np.sqrt(2 * np.pi))
                              * (Pi_upper(v) - Pi_lower(s)) / ((v - s) ** 1.5)
                              * np.exp(-(Pi_upper(v) - Pi_lower(s)) ** 2
                                       / (2 * (v - s))))

        K_2_2 = lambda v, s: ((1 / np.sqrt(2 * np.pi))
                              * (Pi_upper(v) - Pi_upper(s)) / (v - s)
                              * np.exp(-(Pi_upper(v) - Pi_upper(s)) ** 2
                                       / (2 * (v - s))))

        K_2_2_v = lambda v: ((self.theta - optimal_profit)
                             / (np.sqrt((2 * np.pi) * (1 - 2 * v))))

        v = self.v(T)[:-1]

        e_l, e_u, f_l, f_u = self.heat_potential_helper(T, optimal_profit, optimal_stop_loss)

        eps_lower, eps_upper = self.numerical_calculation_equations(v, K_1_1, K_1_1_v, K_1_2, K_2_1, K_2_2, K_2_2_v,
                                                                    e_l, e_u)

        phi_lower, phi_upper = self.numerical_calculation_equations(v, K_1_1, K_1_1_v, K_1_2, K_2_1, K_2_2, K_2_2_v,
                                                                    f_l, f_u)

        return eps_lower, eps_upper, phi_lower, phi_upper

    def numerical_calculation_equations(self, v, K_11, K_11_v, K_12, K_21, K_22, K_22_v, f1, f2):
        """
        p.8 lower
        """

        n = len(v)

        k = n

        nu_1 = np.zeros(n)

        nu_2 = np.zeros(n)

        nu_1[0] = f1[0]

        nu_2[0] = -f2[0]

        nu_1[1] = f1[1] / (1 + K_11_v(v[1]) * np.sqrt(v[1]))

        nu_2[1] = -f2[1] / (1 - K_22_v(v[1]) * np.sqrt(v[1]))

        nu_1_mult = (1 + K_11_v(v[2:k]) * np.sqrt(v[2:k] - v[1:k - 1])) ** -1

        nu_2_mult = (-1 + K_22_v(v[2:k]) * np.sqrt(v[2:k] - v[1:k - 1])) ** -1

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

    def sharpe_helper_functions(self, T, optimal_profit, optimal_stop_loss):
        """
        p.9 upper
        """
        eps_lower, eps_upper, phi_lower, phi_upper = self.numerical_calculation_helper(T, optimal_profit,
                                                                                       optimal_stop_loss)

        v = self.v(T)

        n = len(v)

        w_l = np.zeros(n - 1)

        w_u = np.zeros(n - 1)

        omega = self.omega(T)

        Pi_lower = self.Pi_lower(v[1:-1], optimal_stop_loss)

        Pi_upper = self.Pi_upper(v[1:-1], optimal_profit)

        w_l[:-1] = ((omega - Pi_lower)
                    * np.exp(-(omega - Pi_lower) ** 2
                             / (2 * (v[-1] - v[1:-1])))
                    / (np.sqrt(2 * np.pi) * (v[-1] - v[1:-1]) ** 1.5))

        w_l[-1] = 0

        w_u[:-1] = ((omega - Pi_upper)
                    * np.exp(-(omega - Pi_upper) ** 2
                             / (2 * (v[-1] - v[1:-1])))
                    / (np.sqrt(2 * np.pi) * (v[-1] - v[1:-1]) ** 1.5))

        w_u[-1] = 0

        k = n - 1

        E = sum([w_l])

        E_vect = np.zeros(k)
        F_vect = np.zeros(k)

        for i in range(1, k):
            E_vect[i - 1] = (w_l[i - 1] * eps_lower[i] + w_l[i - 2] * eps_lower[i - 1] + w_l[i - 1] * eps_upper[i] +
                             w_l[i - 2] * eps_upper[i - 1]) * (v[i] - v[i - 1])

        E = 0.5 * sum(E_vect)

        for i in range(1, k):
            F_vect[i - 1] = (w_l[i - 1] * phi_lower[i] + w_l[i - 2] * phi_lower[i - 1] + w_l[i - 1] * phi_upper[i] +
                             w_l[i - 2] * phi_upper[i - 1]) * (v[i] - v[i - 1])

        F = 0.5 * sum(F_vect)

        return E, F

    def sharpe_calculation(self, T, optimal_profit, optimal_stop_loss):
        """
        p.6 middle
        """
        E, F = self.sharpe_helper_functions(T, optimal_profit, optimal_stop_loss)

        gamma = self.gamma(T)

        omega = self.omega(T)

        a = 2 * (omega + self.theta) / np.log(1 - 2 * gamma)

        summ_term = 4 * (gamma + np.log(1 - 2 * gamma) * (omega + self.theta) * E) / (np.log(1 - 2 * gamma)) ** 2

        sharpe_ratio = (E - a) / np.sqrt(F - E ** 2 + summ_term)

        return sharpe_ratio

    def neg_sharpe_calculation(self, params):
        """
        p.6 middle
        """

        T = self.T

        optimal_profit, optimal_stop_loss = params

        E, F = self.sharpe_helper_functions(T, optimal_profit, optimal_stop_loss)

        gamma = self.gamma(T)

        omega = self.omega(T)

        a = 2 * (omega + self.theta) / np.log(1 - 2 * gamma)

        summ_term = 4 * (gamma + np.log(1 - 2 * gamma) * (omega + self.theta) * E) / (np.log(1 - 2 * gamma)) ** 2

        sharpe_ratio = (E - a) / np.sqrt(F - E ** 2 + summ_term)

        return -sharpe_ratio

    def optimal_levels(self):
        """
        """

        stop_loss_guess = self.theta - 6 / np.sqrt(2)

        profit_taking_guess = self.theta + 6 / np.sqrt(2)

        # Setting bounds
        # max duration > 0, profit-taking level > 0, stop-loss < 0
        bounds = ((1e-5, None), (None, 1e-5))

        # Initial guesses for theta, mu, sigma
        initial_guess = np.array((profit_taking_guess, stop_loss_guess))

        result = so.minimize(self.neg_sharpe_calculation, initial_guess, bounds=bounds)

        # Unpacking optimal values
        profit_taking, stop_loss = result.x

        # Undo negation
        max_sharpe = -result.fun

        return profit_taking, stop_loss, max_sharpe
