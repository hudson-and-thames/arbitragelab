# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Abstract class for bivariate copula implementation.
"""

# pylint: disable = invalid-name, too-many-function-args
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss


class Copula(ABC):
    """
    Copula class houses common functions for each copula subtype.
    """

    def __init__(self, copula_name: str):
        """
        Initiate a Copula class.

        This is a helper superclass for all named copulas in this module. There is no need to directly initiate.

        :param copula_name: (str) Copula name.
        """

        # Name of each types of copula
        self.archimedean_names = ('Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14')
        self.elliptic_names = ('Gaussian', 'Student')
        self.theta = None
        self.rho = None
        self.nu = None
        self.copula_name = copula_name

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

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

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

        # Mapping u, v back to the valid computational interval
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # Wrapper around individual copula's c method
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

        # Mapping u, v back to the valid computational interval
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # Wrapper around individual copula's C method
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

        # Mapping u, v back to the valid computational interval
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # Wrapper around individual copula's condi_cdf method
        return self.condi_cdf(u, v)

    def get_log_likelihood_sum(self, u: np.array, v: np.array) -> float:
        """
        Get log-likelihood value sum.

        :param u: (np.array) 1D vector data of X pseudo-observations. Need to be uniformly distributed [0, 1].
        :param v: (np.array) 1D vector data of Y pseudo-observations. Need to be uniformly distributed [0, 1].
        :return: (float) Log-likelihood sum value.
        """

        # Likelihood quantity for each pair of data, stored in a list
        likelihood_list = [self.c(xi, yi) for (xi, yi) in zip(u, v)]

        # Sum of logarithm of likelihood data
        log_likelihood_sum = np.sum(np.log(likelihood_list))

        return log_likelihood_sum

    def c(self, u: float, v: float) -> float:
        """
        Placeholder for calculating copula density.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        """

    def C(self, u: float, v: float) -> float:
        """
        Placeholder for calculating copula evaluation.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        """

    def condi_cdf(self, u: float, v: float) -> float:
        """
        Placeholder for calculating copula conditional probability.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        """

    @abstractmethod
    def sample(self, num: int = None, unif_vec: np.array = None) -> np.array:
        """
        Place holder for sampling from copula.

        :param num: (int) Number of points to generate.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data.
        """

    def fit(self, u: np.array, v: np.array) -> float:
        """
        Fit copula to empirical data (pseudo-observations). Once fit, `self.theta` is updated.

        :param u: (np.array) 1D vector data of X pseudo-observations. Need to be uniformly distributed [0, 1].
        :param v: (np.array) 1D vector data of Y pseudo-observations. Need to be uniformly distributed [0, 1].
        :return: (float) Theta hat estimate for fit copula.
        """

        # Calculate Kendall's tau from data
        tau = ss.kendalltau(u, v)[0]

        # Translate Kendall's tau into theta
        theta_hat = self.theta_hat(tau)
        self.theta = theta_hat

        return theta_hat

    @abstractmethod
    def _get_param(self):
        """
        Placeholder for getting the parameter(s) of the specific copula.
        """

    @staticmethod
    def _3d_surface_plot(x: np.array, y: np.array, z: np.array, bounds: list, title: str, **kwargs) -> plt.axis:
        """
        Helper function to plot 3-d plot.

        :param x: (np.array) X-axis data.
        :param y: (np.array) Y-axis data.
        :param z: (np.array) Z-axis data.
        :param bounds: (list) Min and max values bounds.
        :param title: (str) Plot title.
        :param kwargs: (dict) User-specified params for `ax.plot_surface`.
        :return: (plt.axis) Axis object.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xticks(np.linspace(bounds[0], bounds[1], 6))
        ax.set_yticks(np.linspace(bounds[0], bounds[1], 6))
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.plot_surface(x, y, z, **kwargs)
        plt.title(title)
        plt.show()

        return ax

    @staticmethod
    def _2d_contour_plot(x: np.array, y: np.array, z: np.array, bounds: float, title: str,
                         levels: list, **kwargs) -> plt.axis:
        """
        Helper function to plot 2-d contour plot.

        :param x: (np.array) X-axis data.
        :param y: (np.array) Y-axis data.
        :param z: (np.array) Z-axis data.
        :param bounds: (list) Min and max values bounds.
        :param title: (str) Plot title.
        :param levels: (list) List of float values that determine the number and levels of lines in a contour plot.
        :param kwargs: (dict) User-specified params for `plt.contour`.
        :return: (plt.axis) Axis object.
        """

        plt.figure()
        contour_plot = plt.contour(x, y, z, levels, colors='k', linewidths=1., linestyles=None, **kwargs)
        plt.clabel(contour_plot, fontsize=8, inline=1)
        plt.xlim(bounds)
        plt.ylim(bounds)
        plt.title(title)
        plt.show()

        return contour_plot

    def plot_cdf(self, plot_type: str = '3d', grid_size: int = 50, levels: list = None, **kwargs) -> plt.axis:
        """
        Plot either '3d' or 'contour' plot of copula CDF.

        :param plot_type: (str) Either '3d' or 'contour'(2D) plot.
        :param grid_size: (int) Mesh grid granularity.
        :param kwargs: (dict) User-specified params for 'ax.plot_surface'/'plt.contour'.
        :param levels: (list) List of float values that determine the number and levels of lines in a contour plot.
            If not provided, these are calculated automatically.
        :return: (plt.axis) Axis object.
        """

        title = "Copula CDF"

        bounds = [0 + 1e-2, 1 - 1e-2]
        u_grid, v_grid = np.meshgrid(
            np.linspace(bounds[0], bounds[1], grid_size),
            np.linspace(bounds[0], bounds[1], grid_size))

        z = np.array(
            [self.C(u, v) for u, v in zip(np.ravel(u_grid), np.ravel(v_grid))])

        z = z.reshape(u_grid.shape)

        if plot_type == "3d":
            ax = self._3d_surface_plot(u_grid, v_grid, z, [0, 1], title, **kwargs)

        elif plot_type == "contour":
            # Calculate levels if not given by user
            if not levels:
                min_ = np.nanpercentile(z, 5)
                max_ = np.nanpercentile(z, 95)
                levels = np.round(np.linspace(min_, max_, num=5), 3)
            ax = self._2d_contour_plot(u_grid, v_grid, z, [0, 1], title, levels, **kwargs)

        else:
            raise ValueError('Only contour and 3d plot options are available.')

        return ax

    def plot_scatter(self, num_points: int = 100) -> plt.axis:
        """
        Plot copula scatter plot of generated pseudo-observations.

        :param num_points: (int) Number of samples to generate.
        :return: (plt.axis) Axis object.
        """

        samples = self.sample(num=num_points)
        ax = sns.kdeplot(x=samples[:, 0], y=samples[:, 1], shade=True)
        ax.set_title('Scatter/heat plot for generated copula samples.')

        return ax

    def plot_pdf(self, plot_type: str = '3d', grid_size: int = 50, levels: list = None, **kwargs) -> plt.axis:
        """
        Plot either '3d' or 'contour' plot of copula PDF.

        :param plot_type: (str) Either '3d' or 'contour'(2D) plot.
        :param grid_size: (int) Mesh grid granularity.
        :param levels: (list) List of float values that determine the number and levels of lines in a contour plot.
            If not provided, these are calculated automatically.
        :return: (plt.axis) Axis object.
        """

        title = " Copula PDF"

        if plot_type == "3d":
            bounds = [0 + 1e-1 / 2, 1 - 1e-1 / 2]
        else:  # plot_type == "contour"
            bounds = [0 + 1e-2, 1 - 1e-2]

        u_grid, v_grid = np.meshgrid(
            np.linspace(bounds[0], bounds[1], grid_size),
            np.linspace(bounds[0], bounds[1], grid_size))

        z = np.array(
            [self.c(u, v) for u, v in zip(np.ravel(u_grid), np.ravel(v_grid))])

        z = z.reshape(u_grid.shape)

        if plot_type == "3d":
            ax = self._3d_surface_plot(u_grid, v_grid, z, [0, 1], title, **kwargs)

        elif plot_type == "contour":
            # Calculate levels if not given by user
            if not levels:
                min_ = np.nanpercentile(z, 5)
                max_ = np.nanpercentile(z, 95)
                levels = np.round(np.linspace(min_, max_, num=5), 3)
            ax = self._2d_contour_plot(u_grid, v_grid, z, [0, 1], title, levels, **kwargs)

        else:
            raise ValueError('Only contour and 3d plot options are available.')

        return ax
