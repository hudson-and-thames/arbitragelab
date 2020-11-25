# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

# pylint: disable=missing-module-docstring, invalid-name, too-many-instance-attributes
import warnings
import numpy as np
# from scipy.integrate import quad
# from scipy.optimize import root_scalar
# import matplotlib.pyplot as plt
# import scipy.optimize as so

# import pandas as pd

class Heat_potentials():
    """
    """

    def __init__(self):
        self.theta = None
        self.optimal_profit = None
        self.optimal_stop_loss = None
        self.delta_grid = None

    def v(self, T: float):
        """
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
        """
        output = (1 - np.exp(-2 * T)) / 2

        return output

    def omega(self, T: float):
        """
        """
        gamma = self.gamma(T)

        output = -np.sqrt(1 - 2 * gamma) * self.theta

        return output

    def Pi_upper(self, v: np.ndarray, optimal_profit: float):
        """
        """

        output = np.sqrt(1 - 2 * v) * (optimal_profit - self.theta)

        return output

    def Pi_lower(self, v: np.ndarray, optimal_stop_loss: float):
        """
        """

        output = np.sqrt(1 - 2 * v) * (optimal_stop_loss - self.theta)

        return output

    def heat_potential_helper(self, T: float, optimal_profit: float, optimal_stop_loss: float):
        """
        """

        v = self.v(T)[:-1]

        gamma = self.gamma(T)

        Pi_upper = self.Pi_upper(v, optimal_profit)

        Pi_lower = self.Pi_lower(v, optimal_stop_loss)

        e_lower = (2 * optimal_stop_loss / np.log((1 - 2 * v) / (1 - 2 * gamma))
                   + 2 * (Pi_lower + self.theta) / np.log(1 - 2 * gamma))

        e_upper = (2 * optimal_profit / np.log((1 - 2 * v) / (1 - 2 * gamma))
                   + 2 * (Pi_upper + self.theta) / np.log(1 - 2 * gamma))

        f_lower = (4 * optimal_stop_loss ** 2 / np.log((1 - 2 * v) / (1 - 2 * gamma)) ** 2
                   - 4 * (v + (Pi_lower + self.theta) ** 2) / np.log(1 - 2 * gamma) ** 2)

        f_upper = (4 * optimal_profit ** 2 / np.log((1 - 2 * v) / (1 - 2 * gamma)) ** 2
                   - 4 * (v + (Pi_upper + self.theta) ** 2) / np.log(1 - 2 * gamma) ** 2)

        return e_upper, e_lower, f_upper, f_lower

    def numerical_calculation_helper(self, T: float, optimal_profit: float, optimal_stop_loss: float):
        """
        """
        Pi_upper = lambda v: self.Pi_upper(v, optimal_profit)

        Pi_lower = lambda v: self.Pi_lower(v, optimal_stop_loss)

        K_1_1 = lambda v, s: ((1 / np.sqrt(2 * np.pi)) * (Pi_lower(v) - Pi_lower(s)) / (v - s)
                              * np.exp(-(Pi_lower(v) - Pi_lower(s)) ** 2 / 2 * (v - s)))

        K_1_1_v = lambda v: ((self.theta - optimal_stop_loss) / (np.sqrt((2 * np.pi) * (1 - 2 * v))))

        K_1_2 = lambda v, s: ((1 / np.sqrt(2 * np.pi)) * (Pi_lower(v) - Pi_upper(s)) / (v - s)
                              * np.exp(-(Pi_lower(v) - Pi_upper(s)) ** 2 / 2 * (v - s)))

        K_2_1 = lambda v, s: ((1 / np.sqrt(2 * np.pi)) * (Pi_upper(v) - Pi_lower(s)) / (v - s)
                              * np.exp(-(Pi_upper(v) - Pi_lower(s)) ** 2 / 2 * (v - s)))

        K_2_2 = lambda v, s: ((1 / np.sqrt(2 * np.pi)) * (Pi_upper(v) - Pi_upper(s)) / (v - s)
                              * np.exp(-(Pi_upper(v) - Pi_upper(s)) ** 2 / 2 * (v - s)))

        K_2_2_v = lambda v: ((self.theta - optimal_profit) / (np.sqrt((2 * np.pi) * (1 - 2 * v))))

        e_l, e_u, f_l, f_u = self.heat_potential_helper(T, optimal_profit, optimal_stop_loss)

        v = self.v(T)[:-1]

        eps_lower, eps_upper = self.numerical_calculation_equations(v, K_1_1, K_1_1_v, K_1_2, K_2_1, K_2_2, K_2_2_v,
                                                                    e_l, e_u)

        phi_lower, phi_upper = self.numerical_calculation_equations(v, K_1_1, K_1_1_v, K_1_2, K_2_1, K_2_2, K_2_2_v,
                                                                    f_l, f_u)

        return eps_lower, eps_upper, phi_lower, phi_upper

    def numerical_calculation_equations(self, v, K_11, K_11_v, K_12, K_21, K_22, K_22_v, f1, f2):
        """
        """

        n = len(v)

        nu_1 = np.zeros(n)

        nu_2 = np.zeros(n)

        nu_1[0] = f1[0]

        nu_2[0] = -f2[0]

        nu_1[1] = f1[1] / (1 + K_11_v(v[1]) * np.sqrt(v[1]))

        nu_2[1] = -f2[1] / (1 + K_22_v(v[1]) * np.sqrt(v[1]))

        nu_1_mult = (1 + K_11_v(v[2:]) * np.sqrt(v[2:] - v[1:-1])) ** -1

        nu_2_mult = (1 + K_22_v(v[2:]) * np.sqrt(v[2:] - v[1:-1])) ** -1

        for i in range(2, n):
            sum_1 = sum([(K_11(v[i], v[j]) * nu_1[j] + K_11(v[i], v[j - 1]) * nu_1[j - 1])
                         / (np.sqrt(v[i] - v[j]) + np.sqrt(v[i] - v[j + 1]))
                         + 0.5 * (K_12(v[i], v[j]) * nu_2[j] + K_12(v[i], v[j]) * nu_2[j - 1])
                         for j in range(1, i - 1)])

            nu_1[i] = (nu_1_mult[i - 2]
                       * (f1[i]
                          - K_11(v[i], v[i - 1]) * nu_1[i - 1] * np.sqrt(v[i] - v[i - 1])
                          - 0.5 * K_12(v[i], v[i - 1]) * nu_2[i - 1] * (v[i] - v[i - 1])
                          - sum_1 * (v[i] - v[i - 1])))

            sum_2 = sum([0.5 * (K_21(v[i], v[j]) * nu_1[j] + K_21(v[i], v[j]) * nu_1[j - 1])
                         + (K_22(v[i], v[j]) * nu_2[j] + K_11(v[i], v[j - 1]) * nu_1[j - 1])
                         / (np.sqrt(v[i] - v[j]) + np.sqrt(v[i] - v[j + 1]))
                         for j in range(1, i - 1)])

            nu_2[i] = (nu_2_mult[i - 2]
                       * (f2[i]
                          - 0.5 * K_21(v[i], v[i - 1]) * nu_1[i - 1] * (v[i] - v[i - 1])
                          - K_22(v[i], v[i - 1]) * nu_2[i - 1] * np.sqrt(v[i] - v[i - 1])
                          - sum_2 * (v[i] - v[i - 1])))

        return nu_1, nu_2

    def sharpe_helper_functions(self, T, optimal_profit, optimal_stop_loss):
        """
        """
        eps_lower, eps_upper, phi_lower, phi_upper = self.numerical_calculation_helper(T, optimal_profit,
                                                                                       optimal_stop_loss)

        v = self.v(T)[:-1]

        n = len(v)

        w_l = np.zeros(n - 1)

        w_u = np.zeros(n - 1)

        omega = self.omega(T)

        Pi_lower = self.Pi_lower(v[1:-1], optimal_stop_loss)

        Pi_upper = self.Pi_upper(v[1:-1], optimal_profit)

        w_l[:-1] = (((omega - Pi_lower) * np.exp(-(omega - Pi_lower) ** 2 / (v[-1] - v[1:-1])))
                    / (np.sqrt(2 * np.pi) * (v[-1] - v[1:-1]) ** (3 / 2)))

        w_l[-1] = 0

        w_u[:-1] = (((omega - Pi_upper) * np.exp(-(omega - Pi_upper) ** 2 / (v[-1] - v[1:-1])))
                    / (np.sqrt(2 * np.pi) * (v[-1] - v[1:-1]) ** (3 / 2)))

        w_u[-1] = 0

        E = 0.5 * sum((w_l * eps_lower[1:] + w_l * eps_lower[:-1] + w_l * eps_upper[1:] + w_l * eps_upper[:-1]) * (
                    v[1:] - v[:-1]))

        F = E = 0.5 * sum(
            (w_l * phi_lower[1:] + w_l * phi_lower[:-1] + w_l * phi_upper[1:] + w_l * phi_upper[:-1]) * (
                        v[1:] - v[:-1]))

        return E, F

    def sharpe_calculation(self, T, optimal_profit, optimal_stop_loss):
        """
        """
        E = lambda T: self.sharpe_helper_functions(T, optimal_profit, optimal_stop_loss)[0]

        F = lambda T: self.sharpe_helper_functions(T, optimal_profit, optimal_stop_loss)[1]

        gamma = self.gamma(T)

        omega = self.omega(T)

        a = 2 * (omega + self.theta) / np.log(1 - 2 * gamma)

        summ_term = 4 * (gamma + np.log(1 - 2 * gamma) * (omega - self.theta) * E(T)) / np.log(1 - 2 * gamma) ** 2

        print(E(T))

        sharpe_ratio = (E(T) - a) / np.sqrt(F(T) - E(T) ** 2 + summ_term)

        return sharpe_ratio


