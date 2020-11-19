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

    def v(T=float):
        """
        """

        tau = np.arange(0, T, 1)

        output = (1 - np.exp(-2 * tau)) / 2

        return output

    def ksi(self, T=float):
        """
        """
        tau = np.arange(0, T, 1)

        output = np.exp(-tau)(x - self.theta)

        return output

    def gamma(T=float):
        """
        """
        output = (1 - np.exp(-2 * T)) / 2

    def omega(self, T=float):
        """
        """
        gamma = self.gamma(T)

        output = -np.sqrt(1 - 2 * gamma) * self.theta

        return output

    def Pi_upper(self, v=float, optimal_profit=float):
        """
        """

        output = np.sqrt(1 - 2 * v) * (optimal_profit - self.theta)

    def Pi_lower(v=float, optimal_stop_loss=float):
        """
        """

        output = np.sqrt(1 - 2 * v) * (optimal_stop_loss - self.theta)

    def heat_potential_helper(self, T=float, optimal_profit=float, optimal_stop_loss=float):
        """
        """

        v = self.v(T)

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

    def numerical_calculation_helper(self, T=float, optimal_profit=float, optimal_stop_loss=float):
        """
        """
        Pi_upper = lambda v: self.Pi_upper(v, optimal_profit)

        Pi_lower = lambda v: self.Pi_lower(v, optimal_stop_loss)

        K_1_1 = lambda v, s: ((1 / np.sqrt(2 * np.pi)) * (Pi_lower(v) - Pi_lower(s)) / (v - s)
                              * np.exp(-(Pi_lower(v) - Pi_lower(s)) ** 2 / 2 * (v - s)))

        K_1_2 = lambda v, s: ((1 / np.sqrt(2 * np.pi)) * (Pi_lower(v) - Pi_upper(s)) / (v - s)
                              * np.exp(-(Pi_lower(v) - Pi_upper(s)) ** 2 / 2 * (v - s)))

        K_2_1 = lambda v, s: ((1 / np.sqrt(2 * np.pi)) * (Pi_upper(v) - Pi_lower(s)) / (v - s)
                              * np.exp(-(Pi_upper(v) - Pi_lower(s)) ** 2 / 2 * (v - s)))

        K_2_2 = lambda v, s: ((1 / np.sqrt(2 * np.pi)) * (Pi_upper(v) - Pi_upper(s)) / (v - s)
                              * np.exp(-(Pi_upper(v) - Pi_upper(s)) ** 2 / 2 * (v - s)))

        e_l, e_u, f_l, f_u = self.heat_potential_helper(T, optimal_profit, optimal_stop_loss)

        v = self.v(T)

        eps_lower, eps_upper = self.numerical_calculation_equations(self, v, K_1_1, K_1_2, K_2_1, K_2_2, e_l, e_u)

        phi_lower, phi_upper = self.numerical_calculation_equations(self, v, K_1_1, K_1_2, K_2_1, K_2_2, f_l, f_u)

        return eps_lower, eps_upper, phi_lower, phi_upper

    def numerical_calculation_equations(self, v, K_11, K_12, K_21, K_22, f1, f2):
        """
        """

        nu_1, nu_2 = np.array([])

        nu_1[0] = f1(v[0])

        nu_2[0] = -f2(v[0])

        nu_1[1] = f1(v1) / (1 + K_11(v[1], v[1]) * np.sqrt(v[1]))

        nu_2[1] = -f2(v1) / (1 + K_22(v[1], v[1]) * np.sqrt(v[1]))

        for i in range(2, len(v)):
            sum_1 = sum([(K_11(v[i], v[j]) * nu_1[j] + K_11(v[i], v[j - 1]) * nu_1[j - 1]) / (
                        np.sqrt(i - j) + np.sqrt(i - j + 1))
                         + 0.5(K_12(v[i], v[j]) * nu_2[j] + K_12(v[i], v[j]) * nu_2[j - 1]) for j in range(1, i - 1)])

            nu_1[i] = ((1 + K_11(v[i], v[i]) * np.sqrt(self.delta_grid)) ** -1
                       * (f1(v[i]) - K_11(v[i], v[i - 1]) * nu_1[i - 1] * np.sqrt(self.delta_grid)
                          - 0.5 * K_12(v[i], v[i - 1]) * nu_2[i - 1] * self.delta_grid
                          - sum_1 * self.delta_grid))

            sum_2 = sum([0.5(K_21(v[i], v[j]) * nu_1[j] + K_21(v[i], v[j]) * nu_1[j - 1])
                         + (K_22(v[i], v[j]) * nu_2(j) + K_11(v[i], v[j - 1]) * nu_1[j - 1]) / (
                                     np.sqrt(i - j) + np.sqrt(i - j + 1))
                         for j in range(1, i - 1)])

            nu_2[i] = ((-1 + K_22(v[i], v[i]) * np.sqrt(self.delta_grid)) ** -1
                       * (f2v[i] - 0.5 * K_21(v[i], v[i - 1]) * nu_1[i - 1] * self.delta_grid
                          - K_22(v[i], v[i - 1]) * nu_2[i - 1] * np.sqrt(self.delta_grid)
                          - sum_2 * self.delta_grid))

        return nu_1, nu_2






