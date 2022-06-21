# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module that houses N13 copula class.
"""

# pylint: disable = invalid-name, too-many-lines
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from arbitragelab.copula_approach.base import Copula
from arbitragelab.util import segment


class N13(Copula):
    """
    N13 Copula (Nelsen 13).
    """

    def __init__(self, theta: float = None, threshold: float = 1e-10):
        r"""
        Initiate an N13 copula object.

        :param theta: (float) Range in [0, +inf), measurement of copula dependency.
        :param threshold: (float) Optional. Below this threshold, a percentile will be rounded to the threshold.
        """

        super().__init__('N13')
        # Lower than this amount will be rounded to threshold
        self.threshold = threshold
        self.theta = theta  # Default input

        segment.track('N13Copula')

    def sample(self, num: int = None, unif_vec: np.array = None) -> np.array:
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.

        User may choose to side-load independent uniformly distributed data in [0, 1].

        :param num: (int) Number of points to generate.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data.
            Default uses numpy pseudo-random generators.
        :return: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """

        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec.")

        theta = self.theta  # Use the default input

        def _Kc(w: float, theta: float):
            return w + 1 / theta * (
                    w - w * np.power((1 - np.log(w)), 1 - theta) - w * np.log(w))

        # Generate pairs of indep uniform dist vectors. Use numpy to generate
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute N13 copulas from the i.i.d. unif pairs
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta,
                                                        Kc=_Kc)

        return sample_pairs

    def _generate_one_pair(self, v1: float, v2: float, theta: float, Kc: Callable[[float, float], float]) -> tuple:
        """
        Generate one pair of vectors from N13 copula.

        :param v1: (float) I.I.D. uniform random variable in [0,1].
        :param v2: (float) I.I.D. uniform random variable in [0,1].
        :param theta: (float) Range in [1, +inf), measurement of copula dependency.
        :param Kc: (func) Conditional probability function, for numerical inverse.
        :return: (tuple) The sampled pair in [0, 1]x[0, 1].
        """

        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1, theta) - v2,
                       self.threshold, 1 - self.threshold)
        else:
            w = self.threshold  # Below the threshold, gives threshold as the root
        u1 = np.exp(
            1 - (v1 * ((1 - np.log(w)) ** theta - 1) + 1) ** (1 / theta))

        u2 = np.exp(
            1 - ((1 - v1) * ((1 - np.log(w)) ** theta - 1) + 1) ** (1 / theta))

        return u1, u2

    def _get_param(self) -> dict:
        """
        Get the name and parameter(s) for this copula instance.

        :return: (dict) Name and parameters for this copula.
        """

        descriptive_name = 'Bivariate Nelsen 13 Copula'
        class_name = 'N13'
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
        u_part = (1 - np.log(u)) ** theta
        v_part = (1 - np.log(v)) ** theta
        Cuv = self.C(u, v)

        numerator = (Cuv * u_part * v_part
                     * (-1 + theta + (-1 + u_part + v_part) ** (1 / theta))
                     * (-1 + u_part + v_part) ** (1 / theta))

        denominator = u * v * (1 - np.log(u)) * (1 - np.log(v)) * (-1 + u_part + v_part) ** 2

        pdf = numerator / denominator

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
        u_part = (1 - np.log(u)) ** theta
        v_part = (1 - np.log(v)) ** theta
        cdf = np.exp(
            1 - (-1 + u_part + v_part) ** (1 / theta))

        return cdf

    def condi_cdf(self, u, v) -> float:
        """
        Calculate conditional probability function: P(U<=u | V=v).

        Result is analytical.

        Note: This probability is symmetric about (u, v).

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The conditional probability.
        """

        theta = self.theta
        u_part = (1 - np.log(u)) ** theta
        v_part = (1 - np.log(v)) ** theta
        Cuv = self.C(u, v)

        numerator = Cuv * (-1 + u_part + v_part) ** (1 / theta) * v_part
        denominator = v * (-1 + u_part + v_part) * (1 - np.log(v))

        result = numerator / denominator

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

        # Calculate tau(theta) = 1 + 4*intg_0^1[phi(t)/d(phi(t)) dt]
        def kendall_tau(theta):
            # phi(t)/d(phi(t)), phi is the generator function for this copula
            pddp = lambda x: -((x - x * (1 - np.log(x)) ** (1 - theta) - x * np.log(x)) / theta)
            result = quad(pddp, 0, 1, full_output=1)[0]
            return 1 + 4 * result

        # Numerically find the root.
        result = brentq(lambda theta: kendall_tau(theta) - tau, 1e-7, 100)

        return result
