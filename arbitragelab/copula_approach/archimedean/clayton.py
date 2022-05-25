# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module that houses all copula classes and the parent copula class.

Also include a Switcher class to create copula by its name and parameters,
to emulate a switch functionality.
"""

# pylint: disable = invalid-name, too-many-lines
from abc import ABC, abstractmethod
from typing import Callable
from scipy.optimize import brentq
from scipy.special import gamma as gm
from scipy.integrate import dblquad, quad
import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arbitragelab.copula_approach.base import Copula

from arbitragelab.util import segment


class Clayton(Copula):
    """
    Clayton copula.
    """

    def __init__(self, theta: float = None, threshold: float = 1e-10, ):
        r"""
        Initiate a Clayton copula object.

        :param theta: (float) Range in [-1, +inf) \ {0}, measurement of copula dependency.
        :param threshold: (float) Optional. Below this threshold, a percentile will be rounded to the threshold.
        """

        super().__init__()
        # Lower than this amount will be rounded to threshold
        self.threshold = threshold
        self.theta = theta  # Default input

        segment.track('ClaytonCopula')

    def sample(self, num: int = None, unif_vec: np.array = None) -> np.array:
        r"""
        Generate pairs according to P.D.F., stored in a 2D np.array.

        User may choose to side-load independent uniformly distributed data in [0, 1].

        Note: Large theta might suffer from accuracy issues.

        :param num: (int) Number of points to generate.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data.
            Default uses numpy pseudo-random generators.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """

        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec.")

        theta = self.theta  # Use the default input

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Frank copulas from the unif pairs
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta)

        return sample_pairs

    @staticmethod
    def _generate_one_pair(u1: float, v2: float, theta: float) -> tuple:
        r"""
        Generate one pair of vectors from Clayton copula.

        :param v1: (float) I.I.D. uniform random variable in [0,1].
        :param v2: (float) I.I.D. uniform random variable in [0,1].
        :param theta: (float) Range in [1, +inf), measurement of copula dependency.
        :return: (tuple) The sampled pair in [0, 1]x[0, 1].
        """

        u2 = np.power(u1 ** (-theta) * (v2 ** (-theta / (1 + theta)) - 1) + 1,
                      -1 / theta)

        return u1, u2

    def _get_param(self) -> dict:
        """
        Get the name and parameter(s) for this copula instance.

        :return: (dict) Name and parameters for this copula.
        """

        descriptive_name = 'Bivariate Clayton Copula'
        class_name = 'Clayton'
        theta = self.theta
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'theta': theta}

        return info_dict

    def c(self, u: float, v: float) -> float:
        """
        Calculate probability density of the bivariate copula: P(U=u, V=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The probability density (aka copula density).
        """

        theta = self.theta
        u_part = u ** (-1 - theta)
        v_part = v ** (-1 - theta)
        pdf = ((1 + theta) * u_part * v_part
               * (-1 + u_part * u + v_part * v) ** (-2 - 1 / theta))

        return pdf

    def C(self, u: float, v: float) -> float:
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The cumulative density.
        """

        theta = self.theta
        cdf = np.max(u ** (-1 * theta) + v ** (-1 * theta) - 1,
                     0) ** (-1 / theta)

        return cdf

    def condi_cdf(self, u: float, v: float) -> float:
        """
        Calculate conditional probability function: P(U<=u | V=v).

        Result is analytical.

        Note: This probability is symmetric about (u, v).

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The conditional probability.
        """

        theta = self.theta
        unt = u ** (-1 * theta)
        vnt = v ** (-1 * theta)
        t_power = 1 / theta + 1
        result = vnt / v / np.power(unt + vnt - 1, t_power)

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

        return 2 * tau / (1 - tau)
