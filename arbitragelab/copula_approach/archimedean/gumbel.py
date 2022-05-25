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


class Gumbel(Copula):
    """
    Gumbel Copula.
    """

    # TODO: remebmer theta and threshold have been swapped! That's why tests break down.

    def __init__(self, theta: float = None, threshold: float = 1e-10):
        r"""
        Initiate a Gumbel copula object.

        :param theta: (float) Range in [1, +inf), measurement of copula dependency.
        :param threshold: (float) Optional. Below this threshold, a percentile will be rounded to the threshold.

        """

        super().__init__()
        # Lower than this amount will be rounded to threshold.
        self.threshold = threshold
        self.theta = theta  # Gumbel copula parameter.

        segment.track('GumbelCopula')

    def sample(self, num: int = None, unif_vec: np.array = None) -> np.array:
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.

        User may choose to side-load independent uniformly distributed data in [0, 1].

        :param num: (int) Number of points to generate.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data.
            Default uses numpy pseudo-random generators.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """

        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec.")

        theta = self.theta  # Use the default input

        # Distribution of C(U1, U2). To be used for numerically solving the inverse.
        def _Kc(w: float, theta: float):
            return w * (1 - np.log(w) / theta)

        # Generate pairs of indep uniform dist vectors.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Gumbel copulas from the independent uniform pairs.
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta,
                                                        Kc=_Kc)

        return sample_pairs

    def _generate_one_pair(self, v1: float, v2: float, theta: float, Kc: Callable[[float, float], float]) -> tuple:
        """
        Generate one pair of vectors from Gumbel copula.

        v1, v2 are i.i.d. random numbers uniformly distributed in [0, 1].

        :param v1: (float) I.I.D. uniform random variable in [0, 1].
        :param v2: (float) I.I.D. uniform random variable in [0, 1].
        :param theta: (float) Range in [1, +inf), measurement of copula dependency.
        :param Kc: (func) Conditional probability function, for numerical inverse.
        :return: (tuple) The sampled pair in [0, 1]x[0, 1].
        """
        # Numerically root finding for w1, where Kc(w1) = v2.
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1, theta) - v2, self.threshold, 1)
        else:
            w = 1e10  # Below the threshold, gives a large number as root.
        u1 = np.exp(v1 ** (1 / theta) * np.log(w))
        u2 = np.exp((1 - v1) ** (1 / theta) * np.log(w))

        return u1, u2

    def _get_param(self):
        """
        Get the name and parameter(s) for this copula instance.

        :return: (dict) Name and parameters for this copula.
        """

        descriptive_name = 'Bivariate Gumbel Copula'
        class_name = 'Gumbel'
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
        # Prepare parameters.
        u_part = (-np.log(u)) ** theta
        v_part = (-np.log(v)) ** theta
        expo = (u_part + v_part) ** (1 / theta)

        # Assembling for P.D.F.
        pdf = 1 / (u * v) \
              * (np.exp(-expo)
                 * u_part / (-np.log(u)) * v_part / (-np.log(v))
                 * (theta + expo - 1)
                 * (u_part + v_part) ** (1 / theta - 2))

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
        # Prepare parameters.
        expo = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta)

        # Assembling for P.D.F.
        cdf = np.exp(-expo)

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
        expo = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** ((1 - theta) / theta)
        result = self.C(u, v) * expo * (-np.log(v)) ** (theta - 1) / v

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

        return 1 / (1 - tau)
