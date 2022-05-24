# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Abstract class for pairs copulas implementation
"""

# pylint: disable = invalid-name
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Copula(ABC):
    """
    Copula class houses common functions for each copula subtype.
    """

    def __init__(self):
        """
        Initiate a Copula class.

        This is a helper superclass for all named copulas in this module. There is no need to directly initiate.
        """

        # Name of each types of copula.
        self.archimedean_names = ('Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14')
        self.elliptic_names = ('Gaussian', 'Student')
        self.theta = None
        self.rho = None
        self.nu = None

    def describe(self) -> pd.Series:
        """
        Print the description of the copula's name and parameter as a pd.Series.

        Note: the descriptive name is different from the copula's class name, but its full actual name.
        E.g. The Student copula class has its descriptive name as 'Bivariate Student-t Copula'.

        :return description: (pd.Series) The description of the copula, including its descriptive name, class name,
            and its parameter(s) when applicable.
        """

        description = pd.Series(self._get_param())

        return description

    def get_cop_density(self, u: float, v: float, eps: float = 1e-5) -> float:
        """
        Get the copula density c(u, v).

        Result is analytical. Also the u and v will be remapped into [eps, 1-eps] to avoid edge values that may
        result in infinity or NaN.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :param eps: (float) Optional. The distance to the boundary 0 or 1, such that the value u, v will be mapped
            back. Defaults to 1e-5.
        :return: (float) The probability density (aka copula density).
        """

        # Mapping u, v back to the valid computational interval.
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # Wrapper around individual copula's c method.
        return self.c(u, v)

    def get_cop_eval(self, u: float, v: float, eps: float = 1e-5) -> float:
        """
        Get the evaluation of copula, equivalently the cumulative joint distribution C(u, v).

        Result is analytical. Also the u and v will be remapped into [eps, 1-eps] to avoid edge values that may
        result in infinity or NaN.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :param eps: (float) Optional. The distance to the boundary 0 or 1, such that the value u, v will be mapped
            back. Defaults to 1e-5.
        :return: (float) The evaluation of copula (aka cumulative joint distribution).
        """

        # Mapping u, v back to the valid computational interval.
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # Wrapper around individual copula's C method.
        return self.C(u, v)

    def get_condi_prob(self, u: float, v: float, eps: float = 1e-5) -> float:
        """
        Calculate conditional probability function: P(U<=u | V=v).

        Result is analytical. Also the u and v will be remapped into [eps, 1-eps] to avoid edge values that may
        result in infinity or NaN.

        Note: This probability is symmetric about (u, v).

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :param eps: (float) Optional. The distance to the boundary 0 or 1, such that the value u, v will be mapped
            back. Defaults to 1e-5.
        :return: (float) The conditional probability.
        """

        # Mapping u, v back to the valid computational interval.
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # Wrapper around individual copula's condi_cdf method.
        return self.condi_cdf(u, v)

    def get_log_likelihood_sum(self, u: np.array, v: np.array) -> float:
        """
        Get log-likelihood value sum.

        :param u: (np.array) 1D vector data of X pseudo-observations. Need to be uniformly distributed [0, 1].
        :param v: (np.array) 1D vector data of Y pseudo-observations. Need to be uniformly distributed [0, 1].
        :return: (float) Log-likelihood sum value.
        """
        # Likelihood quantity for each pair of data, stored in a list.
        likelihood_list = [self.c(xi, yi) for (xi, yi) in zip(u, v)]
        # Sum of logarithm of likelihood data.
        log_likelihood_sum = np.sum(np.log(likelihood_list))
        return log_likelihood_sum

    def plot(self, num: int, ax: plt.axes = None, **ax_kwargs) -> plt.axes:
        """
        Plot a given number sampling of the copula.

        :param num: (int) The number of points to sample for plotting.
        :param ax: (plt.axes) Optional. The plotting axes. If not provided it will generate its own. Defaults to None.
        :param ax_kwargs: (dict) Additional axes keyword arguments for plotting.
        :return: (plt.axes) The plot axes.
        """

        samples = self.generate_pairs(num)  # Sample from each copula for plotting

        # Plot the samples on an axes
        ax = ax or plt.gca()
        ax.scatter(samples[:, 0], samples[:, 1], **ax_kwargs)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal', adjustable='box')  # Equal scale in x and y.

        # Modify the title.
        copula_name = self.__class__.__name__  # Each specific copula's name
        if copula_name in self.archimedean_names:
            ax.set_title(r'{} Copula, $\theta={:.3f}$'.format(copula_name, self.theta))
        if copula_name == "Gaussian":
            ax.set_title(r'{} Copula, $\rho={:.3f}$'.format(copula_name, self.rho))
        if copula_name == "Student":
            ax.set_title(r'Student-t Copula, $\rho={:.3f}$, $\nu={:.3f}$'.format(self.rho, self.nu))

        return ax

    @abstractmethod
    def c(self, u: float, v: float) -> float:
        """
        Place holder for calculating copula density.
        """

    @abstractmethod
    def C(self, u: float, v: float) -> float:
        """
        Place holder for calculating copula evaluation.
        """

    @abstractmethod
    def condi_cdf(self, u: float, v: float) -> float:
        """
        Place holder for calculating copula conditional probability.
        """

    @abstractmethod
    def _get_param(self):
        """
        Place holder for getting the parameter(s) of the specific copula.
        """