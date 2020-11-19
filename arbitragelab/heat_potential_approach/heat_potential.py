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
