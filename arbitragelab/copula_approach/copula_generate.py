# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module that houses all copula classes and the parent copula class.

Also include a Switcher class to create copula by its name and parameters,
to emulate a switch functionality.
"""
# pylint: disable = invalid-name, too-many-lines
from typing import Callable
from scipy.optimize import brentq
from scipy.special import gamma as gm
from scipy.integrate import quad
import scipy.stats as ss
import numpy as np
import pandas as pd


class Copula:
    """
    Copula class houses common functions for each coplula subtype.
    """

    def __init__(self):
        """
        Initiate a Copula class.

        This is a helper superclass for all named copulas in this module. There is no need to directly initiate.
        """

    def describe(self, if_print: bool = True) -> pd.Series:
        r"""
        Print the description of the copula's name and parameter as a pd.Series.

        Note: the descriptive name is different from the copula's class name, but its full actual name.
        E.g. The Student copula class has its descriptive name as 'Bivariate Student-t Copula'.

        :param if_print: (bool) Whether prints the description. Defaults to True.
        :return description: (pd.Series) The description of the copula, including its descriptive name, class name,
            and its parameter(s) when applicable.
        """

        description = pd.Series(self._get_param())
        if if_print:
            print(description)

        return description


class Gumbel(Copula):
    """
    Gumbel Copula.
    """

    def __init__(self, threshold: float = 1e-10, theta: float = None):
        r"""
        Initiate a Gumbel copula object.

        :param threshold: (float) Optional. Below this threshold, a percentile will be rounded to the threshold.
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        """

        super().__init__()
        # Lower than this amount will be rounded to threshold.
        self.threshold = threshold
        self.theta = theta  # Gumbel copula parameter.

    def generate_pairs(self, num: int = None, unif_vec: np.array = None) -> np.array:
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
        :param theta: (float) Range in [1, +inf), measurement of correlation.
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
        :return: (float) The culumative density.
        """

        theta = self.theta
        # Prepare parameters.
        expo = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta)

        # Assembling for P.D.F.
        cdf = np.exp(-expo)

        return cdf

    def condi_cdf(self, u: float, v: float) -> float:
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).

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


class Frank(Copula):
    """
    Frank Copula.
    """

    def __init__(self, threshold: float = 1e-10, theta: float = None):
        r"""
        Initiate a Frank copula object.

        :param threshold: (float) Optional. Below this threshold, a percentile will be rounded to the threshold.
        :param theta: (float) All reals except for 0, measurement of correlation.
        """

        super().__init__()
        # Lower than this amount will be rounded to threshold
        self.threshold = threshold
        self.theta = theta  # Default input

    def generate_pairs(self, num: int = None, unif_vec: np.array = None) -> np.array:
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.

        User may choose to side-load independent uniformly distributed data in [0, 1]

        :param num: (int) Number of points to generate.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data.
            Default uses numpy pseudo-random generators.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """

        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec.")

        theta = self.theta  # Use the default input.

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Frank copulas from the unif pairs.
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta)

        return sample_pairs

    @staticmethod
    def _generate_one_pair(u1: float, v2: float, theta: float) -> tuple:
        """
        Generate one pair of vectors from Frank copula.

        :param u1: (float) I.I.D. uniform random variable in [0,1].
        :param v2: (float) I.I.D. uniform random variable in [0,1].
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :return: (tuple) The sampled pair in [0, 1]x[0, 1].
        """

        u2 = -1 / theta * np.log(1 + (v2 * (1 - np.exp(-theta))) /
                                 (v2 * (np.exp(-theta * u1) - 1)
                                  - np.exp(-theta * u1)))

        return u1, u2

    def _get_param(self) -> dict:
        """
        Get the name and parameter(s) for this copula instance.

        :return: (dict) Name and parameters for this copula.
        """

        descriptive_name = 'Bivariate Frank Copula'
        class_name = 'Frank'
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
        et = np.exp(theta)
        eut = np.exp(u * theta)
        evt = np.exp(v * theta)
        pdf = (et * eut * evt * (et - 1) * theta /
               (et + eut * evt - et * eut - et * evt)**2)

        return pdf

    def C(self, u: float, v: float) -> float:
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The culumative density.
        """

        theta = self.theta
        cdf = -1 / theta * np.log(
            1 + (np.exp(-1 * theta * u) - 1) * (np.exp(-1 * theta * v) - 1)
            / (np.exp(-1 * theta) - 1))

        return cdf

    def condi_cdf(self, u: float, v: float) -> float:
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).

        Result is analytical.

        Note: This probability is symmetric about (u, v).

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The conditional probability.
        """

        theta = self.theta
        enut = np.exp(-u * theta)
        envt = np.exp(-v * theta)
        ent = np.exp(-1 * theta)
        result = (envt * (enut - 1)
                  / ((ent - 1) + (enut - 1) * (envt - 1)))

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

        def debye1(theta: float) -> float:
            """
            Debye function D_1(theta).
            """

            result = quad(lambda x: x / theta / (np.exp(x) - 1), 0, theta)

            return result[0]

        def kendall_tau(theta: float) -> float:
            """
            Kendall Tau calculation function.
            """

            return 1 - 4 / theta + 4 * debye1(theta) / theta

        # Numerically find the root.
        result = brentq(lambda theta: kendall_tau(theta) - tau, -100, 100)

        return result


class Clayton(Copula):
    """
    Clayton copula.
    """

    def __init__(self, threshold: float = 1e-10, theta: float = None):
        r"""
        Initiate a Clayton copula object.

        :param threshold: (float) Optional. Below this threshold, a percentile will be rounded to the threshold.
        :param theta: (float) Range in [-1, +inf) \ {0}, measurement of correlation.
        """

        super().__init__()
        # Lower than this amount will be rounded to threshold
        self.threshold = threshold
        self.theta = theta  # Default input

    def generate_pairs(self, num: int = None, unif_vec: np.array = None) -> np.array:
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
        :param theta: (float) Range in [1, +inf), measurement of correlation.
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
        :return: (float) The culumative density.
        """

        theta = self.theta
        cdf = np.max(u ** (-1 * theta) + v ** (-1 * theta) - 1,
                     0) ** (-1 / theta)

        return cdf

    def condi_cdf(self, u: float, v: float) -> float:
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).

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


class Joe(Copula):
    """
    Joe Copula.
    """

    def __init__(self, theta: float = None, threshold: float = 1e-10):
        r"""
        Initiate a Joe copula object.

        :param threshold: (float) Optional. Below this threshold, a percentile will be rounded to the threshold.
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        """

        super().__init__()
        self.theta = theta  # Default input
        # Lower than this amount will be rounded to threshold
        self.threshold = threshold

    def generate_pairs(self, num: int = None, unif_vec: np.array = None) -> np.array:
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.

        User may choose to side-load independent uniformly distributed data in [0, 1].

        :param num: (int) Number of points to generate.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data.
            Default uses numpy pseudo-random generators.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """

        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")

        theta = self.theta  # Use the default input

        def _Kc(w: float, theta: float):
            return w - 1 / theta * (
                (np.log(1 - (1 - w)**theta)) * (1 - (1 - w)**theta)
                / ((1 - w)**(theta - 1)))

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Joe copulas from the unif i.i.d. pairs.
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta,
                                                        Kc=_Kc)

        return sample_pairs

    def _generate_one_pair(self, v1: float, v2: float, theta: float, Kc: Callable[[float, float], float]) -> tuple:
        """
        Generate one pair of vectors from Joe copula.

        :param v1: (float) I.I.D. uniform random variable in [0,1].
        :param v2: (float) I.I.D. uniform random variable in [0,1].
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :param Kc: (func) conditional probability function, for numerical inverse.
        :return: (tuple) The sampled pair in [0, 1]x[0, 1].
        """

        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1, theta) - v2,
                       self.threshold, 1 - self.threshold)
        else:
            w = self.threshold  # Below the threshold, gives the threshold
        u1 = 1 - (1 - (1 - (1 - w)**theta)**v1)**(1 / theta)

        u2 = 1 - (1 - (1 - (1 - w)**theta)**(1 - v1))**(1 / theta)

        return u1, u2

    def _get_param(self) -> dict:
        """
        Get the name and parameter(s) for this copula instance.

        :return: (dict) Name and parameters for this copula.
        """

        descriptive_name = 'Bivariate Joe Copula'
        class_name = 'Joe'
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
        u_part = (1 - u)**theta
        v_part = (1 - v)**theta
        pdf = (u_part / (1 - u) * v_part / (1 - v)
               * (u_part + v_part - u_part * v_part)**(1 / theta - 2)
               * (theta - (u_part - 1) * (v_part - 1)))

        return pdf

    def C(self, u: float, v: float) -> float:
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The culumative density.
        """

        theta = self.theta
        u_part = (1 - u) ** theta
        v_part = (1 - v) ** theta
        cdf = 1 - ((u_part + v_part - u_part * v_part)
                   ** (1 / theta))

        return cdf

    def condi_cdf(self, u: float, v: float) -> float:
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).

        Result is analytical.

        Note: This probability is symmetric about (u, v).

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The conditional probability.
        """

        theta = self.theta
        u_part = (1 - u) ** theta
        v_part = (1 - v) ** theta
        result = -(-1 + u_part) * (u_part + v_part - u_part * v_part)**(-1 + 1/theta) \
            * v_part / (1 - v)

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
            # phi(t)/d(phi(t)), phi is the generator function for this copula.
            pddp = lambda x: (1 - (1 - x)**theta) * (1 - x)**(1 - theta) * np.log(1 - (1 - x)**theta) / theta
            result = quad(pddp, 0, 1, full_output=1)[0]
            return 1 + 4*result

        # Numerically find the root.
        result = brentq(lambda theta: kendall_tau(theta) - tau, 1, 100)

        return result


class N13(Copula):
    """
    N13 Copula (Nelsen 13).
    """

    def __init__(self, threshold: float = 1e-10, theta: float = None):
        r"""
        Initiate an N13 copula object.

        :param threshold: (float) Optional. Below this threshold, a percentile will be rounded to the threshold.
        :param theta: (float) Range in [0, +inf), measurement of correlation.
        """

        super().__init__()
        # Lower than this amount will be rounded to threshold
        self.threshold = threshold
        self.theta = theta  # Default input

    def generate_pairs(self, num: int = None, unif_vec: np.array = None) -> np.array:
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

        def _Kc(w: float, theta: float):
            return w + 1 / theta * (
                w - w * np.power((1 - np.log(w)), 1 - theta) - w * np.log(w))

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
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
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :param Kc: (func) Conditional probability function, for numerical inverse.
        :return: (tuple) The sampled pair in [0, 1]x[0, 1].
        """

        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1, theta) - v2,
                       self.threshold, 1 - self.threshold)
        else:
            w = self.threshold  # Below the threshold, gives threshold as the root.
        u1 = np.exp(
            1 - (v1 * ((1 - np.log(w))**theta - 1) + 1)**(1 / theta))

        u2 = np.exp(
            1 - ((1 - v1) * ((1 - np.log(w))**theta - 1) + 1)**(1 / theta))

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
        u_part = (1 - np.log(u))**theta
        v_part = (1 - np.log(v))**theta
        Cuv = self.C(u, v)

        numerator = (Cuv * u_part * v_part
                     * (-1 + theta + (-1 + u_part + v_part)**(1/theta))
                     * (-1 + u_part + v_part)**(1/theta))

        denominator = u * v * (1 - np.log(u)) * (1 - np.log(v)) * (-1 + u_part + v_part)**2

        pdf = numerator / denominator

        return pdf

    def C(self, u: float, v: float) -> float:
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The culumative density.
        """

        theta = self.theta
        u_part = (1 - np.log(u))**theta
        v_part = (1 - np.log(v))**theta
        cdf = np.exp(
            1 - (-1 + u_part + v_part)**(1/theta))

        return cdf

    def condi_cdf(self, u, v) -> float:
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).

        Result is analytical.

        Note: This probability is symmetric about (u, v).

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The conditional probability.
        """

        theta = self.theta
        u_part = (1 - np.log(u))**theta
        v_part = (1 - np.log(v))**theta
        Cuv = self.C(u, v)

        numerator = Cuv * (-1 + u_part + v_part)**(1/theta) * v_part
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
            # phi(t)/d(phi(t)), phi is the generator function for this copula.
            pddp = lambda x: -((x - x * (1 - np.log(x))**(1 - theta) - x * np.log(x)) / theta)
            result = quad(pddp, 0, 1, full_output=1)[0]
            return 1 + 4*result

        # Numerically find the root.
        result = brentq(lambda theta: kendall_tau(theta) - tau, 1e-7, 100)

        return result


class N14(Copula):
    """
    N14 Copula (Nelsen 14).
    """

    def __init__(self, threshold: float = 1e-10, theta: float = None):
        r"""
        Initiate an N14 copula object.

        :param threshold: (float) Optional. Below this threshold, a percentile will be rounded to the threshold.
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        """

        super().__init__()
        # Lower than this amount will be rounded to threshold.
        self.threshold = threshold
        self.theta = theta  # Default input.

    def generate_pairs(self, num: int = None, unif_vec: np.array = None) -> np.array:
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.

        User may choose to side-load independent uniformly distributed data in [0, 1].

        :param num: (int) Number of points to generate.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data.
            Default uses numpy pseudo-random generators.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """

        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")

        theta = self.theta  # Use the default input.

        def _Kc(w: float, theta: float):
            return -w * (-2 + w**(1/theta))

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Gumbel copulas from the unif pairs.
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta,
                                                        Kc=_Kc)

        return sample_pairs

    def _generate_one_pair(self, v1: float, v2: float, theta: float, Kc: Callable[[float, float], float]) -> tuple:
        """
        Generate one pair of vectors from N14 copula.

        :param v1: (float) I.I.D. uniform random variable in [0,1].
        :param v2: (float) I.I.D. uniform random variable in [0,1].
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :param Kc: (func) Conditional probability function, for numerical inverse.
        :return: (tuple) The sampled pair in [0, 1]x[0, 1].
        """

        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1, theta) - v2,
                       self.threshold, 1 - self.threshold)
        else:
            w = self.threshold  # Below the threshold, gives threshold as the root.
        u1 = (1 + (v1 * (w**(-1 / theta) - 1)**theta) ** (1 / theta))**(-theta)
        u2 = (1 + ((1 - v1) * (w**(-1 / theta) - 1)**theta)**(1 / theta))**(-theta)

        return u1, u2

    def _get_param(self) -> dict:
        """
        Get the name and parameter(s) for this copula instance.

        :return: (dict) Name and parameters for this copula.
        """

        descriptive_name = 'Bivariate Nelsen 14 Copula'
        class_name = 'N14'
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
        u_ker = -1 + np.power(u, 1/theta)
        v_ker = -1 + np.power(v, 1/theta)
        u_part = (-1 + np.power(u, -1/theta))**theta
        v_part = (-1 + np.power(v, -1/theta))**theta
        cdf_ker = 1 + (u_part + v_part)**(1/theta)

        numerator = (u_part * v_part * (cdf_ker -1)
                     * (-1 + theta + 2 * theta * (cdf_ker -1)))

        denominator = ((u_part + v_part)**2 * cdf_ker**(2 + theta)
                       * u * v * u_ker * v_ker * theta)

        pdf = numerator / denominator

        return pdf

    def C(self, u: float, v: float) -> float:
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The culumative density.
        """

        theta = self.theta
        u_part = (-1 + np.power(u, -1/theta))**theta
        v_part = (-1 + np.power(v, -1/theta))**theta
        cdf = (1 + (u_part + v_part)**(1/theta))**(-1 * theta)

        return cdf

    def condi_cdf(self, u: float, v: float) -> float:
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).

        Result is analytical.

        Note: This probability is symmetric about (u, v).

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The conditional probability.
        """
        theta = self.theta
        v_ker = -1 + np.power(v, -1/theta)
        u_part = (-1 + np.power(u, -1/theta))**theta
        v_part = (-1 + np.power(v, -1/theta))**theta
        cdf_ker = 1 + (u_part + v_part)**(1/theta)

        numerator = v_part * (cdf_ker - 1)
        denominator = v**(1 + 1/theta) * v_ker * (u_part + v_part) * cdf_ker**(1 + theta)

        result = numerator / denominator

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

        # N14 has a closed form solution
        result = (1 + tau) / (2 - 2 * tau)

        return result


class Gaussian(Copula):
    """
    Bivariate Gaussian Copula.
    """

    def __init__(self, cov: np.array = None):
        r"""
        Initiate a Gaussian copula object.

        :param cov: (np.array) Covariance matrix (NOT correlation matrix), measurement of covariation. The class will
        calculate correlation internally once the covariance matrix is given.
        """

        super().__init__()
        self.cov = cov  # Covariance matrix
        # Correlation
        self.rho = cov[0][1] / (np.sqrt(cov[0][0]) * np.sqrt(cov[1][1]))

    def generate_pairs(self, num: int = None) -> np.array:
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.

        User may choose to side-load independent uniformly distributed data in [0, 1].

        :param num: (int) Number of points to generate.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """

        cov = self.cov

        gaussian_pairs = self._generate_corr_gaussian(num, cov)
        sample_pairs = ss.norm.cdf(gaussian_pairs)

        return sample_pairs

    @staticmethod
    def _generate_corr_gaussian(num: int, cov: np.array) -> np.array:
        """
        Sample from a bivariate Gaussian dist.

        :param num: (int) Number of samples.
        :param cov: (np.array) Covariance matrix.
        :return: (np.array) The bivariate gaussian sample, shape = (num, 2).
        """

        # Generate bivariate normal with mean 0 and intended covariance.
        rand_generator = np.random.default_rng()
        result = rand_generator.multivariate_normal(mean=[0, 0],
                                                    cov=cov,
                                                    size=num)

        return result

    def _get_param(self) -> dict:
        """
        Get the name and parameter(s) for this copula instance.

        :return: (dict) Name and parameters for this copula.
        """

        descriptive_name = 'Bivariate Gaussian Copula'
        class_name = 'Gaussian'
        cov = self.cov
        rho = self.rho
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'cov': cov,
                     'rho': rho}

        return info_dict

    def c(self, u: int, v: int) -> float:
        """
        Calculate probability density of the bivariate copula: P(U=u, V=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The probability density (aka copula density).
        """

        rho = self.rho
        inv_u = ss.norm.ppf(u)
        inv_v = ss.norm.ppf(v)

        exp_ker = (rho * (-2 * inv_u * inv_v + inv_u**2 * rho + inv_v**2 * rho)
                   / (2 * (rho**2 - 1)))

        pdf = np.exp(exp_ker) / np.sqrt(1 - rho**2)

        return pdf

    def C(self, u: int, v: int) -> float:
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The culumative density.
        """

        corr = [[1, self.rho], [self.rho, 1]]  # Correlation matrix.
        inv_cdf_u = ss.norm.ppf(u)  # Inverse cdf of standard normal.
        inv_cdf_v = ss.norm.ppf(v)
        mvn_dist = ss.multivariate_normal(mean=[0, 0], cov=corr)  # Joint cdf of multivar normal.
        cdf = mvn_dist.cdf((inv_cdf_u, inv_cdf_v))

        return cdf

    def condi_cdf(self, u, v) -> float:
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).

        Result is analytical.

        Note: This probability is symmetric about (u, v).

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The conditional probability.
        """

        rho = self.rho
        inv_cdf_u = ss.norm.ppf(u)
        inv_cdf_v = ss.norm.ppf(v)
        sqrt_det_corr = np.sqrt(1 - rho * rho)
        result = ss.norm.cdf((inv_cdf_u - rho * inv_cdf_v)
                             / sqrt_det_corr)

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

        return np.sin(tau * np.pi / 2)


class Student(Copula):
    """
    Bivariate Student-t Copula, need degree of freedom nu.
    """

    def __init__(self, nu: float = None, cov: np.array = None):
        r"""
        Initiate a Student copula object.

        :param nu: (float) Degrees of freedom.
        :param cov: (np.array) Covariance matrix (NOT correlation matrix), measurement of covariation. The class will
        calculate correlation internally once the covariance matrix is given.
        """

        super().__init__()
        self.cov = cov  # Covariance matrix.
        self.nu = nu  # Degree of freedom.
        # Correlation from covariance matrix.
        self.rho = cov[0][1] / (np.sqrt(cov[0][0]) * np.sqrt(cov[1][1]))

    def generate_pairs(self, num: int = None) -> np.array:
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.

        User may choose to side-load independent uniformly distributed data in [0, 1].

        :param num: (int) Number of points to generate.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """

        cov = self.cov
        nu = self.nu

        student_pairs = self._generate_corr_student(num, cov, nu)
        t_dist = ss.t(df=nu)
        copula_pairs = t_dist.cdf(student_pairs)

        return copula_pairs

    @staticmethod
    def _generate_corr_student(num: int, cov: np.ndarray, nu: float) -> tuple:
        """
        Sample from a bivariate Student-t dist.

        :param num: (int) Number of samples.
        :param cov: (np.array) 2 by 2 covariance matrix.
        :param nu: (float) Degree of freedom.
        :return: (tuple) The sampled pair in [0, 1]x[0, 1].
        """

        # Sample from bivariate Normal with cov=cov.
        rand_generator = np.random.default_rng()
        normal = rand_generator.multivariate_normal(mean=[0, 0], cov=cov, size=num)
        # Sample from Chi-square with df=nu.
        chisq = rand_generator.chisquare(df=nu, size=num)
        result = np.zeros((num, 2))
        for row_idx, row in enumerate(result):
            row[0] = normal[row_idx][0] / np.sqrt(chisq[row_idx] / nu)
            row[1] = normal[row_idx][1] / np.sqrt(chisq[row_idx] / nu)

        return result

    def _get_param(self) -> dict:
        """
        Get the name and parameter(s) for this copula instance.

        :return: (dict) Name and parameters for this copula.
        """

        descriptive_name = r'Bivariate Student-t Copula'
        class_name = r'Student'
        cov = self.cov
        rho = self.rho
        nu = self.nu
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'cov': cov,
                     'rho': rho,
                     'nu (degrees of freedom)': nu}

        return info_dict

    def c(self, u: float, v: float) -> float:
        """
        Calculate probability density of the bivariate copula: P(U=u, V=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The probability density (aka copula density).
        """

        nu = self.nu
        rho = self.rho
        corr = [[1, rho],
                [rho, 1]]
        t_dist = ss.t(df=nu)
        y1 = t_dist.ppf(u)
        y2 = t_dist.ppf(v)

        numerator = self._bv_t_dist(x=(y1, y2),
                                    mu=(0, 0),
                                    cov=corr,
                                    df=nu)
        denominator = t_dist.pdf(y1) * t_dist.pdf(y2)

        pdf = numerator / denominator

        return pdf

    def condi_cdf(self, u: float, v: float) -> float:
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).

        Result is analytical.

        Note: This probability is symmetric about (u, v).

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The conditional probability.
        """

        rho = self.rho
        nu = self.nu
        t_dist = ss.t(nu)
        t_dist_nup1 = ss.t(nu + 1)
        inv_cdf_u = t_dist.ppf(u)
        inv_cdf_v = t_dist.ppf(v)
        numerator = (inv_cdf_u - rho * inv_cdf_v) * np.sqrt(nu + 1)
        denominator = np.sqrt((1 - rho**2) * (inv_cdf_v**2 + nu))

        result = t_dist_nup1.cdf(numerator / denominator)

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

        return np.sin(tau * np.pi / 2)

    @staticmethod
    def _bv_t_dist(x: np.array, mu: np.array, cov: np.array, df: float) -> float:
        """
        Bivariate Student-t probability density.

        :param x: (np.array) A pair of values, shape=(2, ).
        :param mu: (np.array) Mean for the distribution, shape=(2, ).
        :param cov: (np.array) Covariance matrix, shape=(2, 2).
        :param df: (float) Degree of freedom.
        :return: (float) The probability density.
        """
        x1 = x[0] - mu[0]
        x2 = x[1] - mu[1]
        c11 = cov[0][0]
        c12 = cov[0][1]
        c21 = cov[1][0]
        c22 = cov[1][1]
        det_cov = c11 * c22 - c12 * c21
        # Pseudo code: (x.transpose)(cov.inverse)(x)/ Det(cov)
        xT_covinv_x = (-2 * c12 * x1 * x2 + c11 * (x1 ** 2 + x2 ** 2)) / det_cov

        numerator = gm((2 + df) / 2)
        denominator = (gm(df / 2) * df * np.pi * np.sqrt(det_cov)
                       * np.power(1 + xT_covinv_x / df, (2 + df) / 2))

        result = numerator / denominator

        return result


class Switcher:
    """
    Switch class to emulate switch functionality.

    A helper class that creates a copula by its string name.
    """

    def __init__(self):
        """
        Initiate a Switcher object.
        """

        self.theta = None
        self.cov = None
        self.nu = None

    def choose_copula(self, **kwargs: dict) -> Callable[[], object]:
        """
        Choosing a method to instantiate a copula.

        User needs to input copula's name and necessary parameters as kwargs.

        :param kwargs: (dict) Keyword arguments to generate a copula by its name.
            copula_name: (str) Name of the copula.
            theta: (float) A measurement of correlation.
            cov: (np.array) Covariance matrix, only useful for Gaussian and Student-t.
            nu: (float) Degree of freedom, only useful for Student-t.
        :return method: (func) The method that creates the wanted copula. Eventually the returned object
            is a copula.
        """

        # Taking parameters from kwargs.
        copula_name = kwargs.get('copula_name')
        self.theta = kwargs.get('theta', None)
        self.cov = kwargs.get('cov', None)
        self.nu = kwargs.get('nu', None)

        # Create copula from string names, by using class attributes/methods.
        method_name = '_create_' + str(copula_name).lower()
        method = getattr(self, method_name)

        return method()

    def _create_gumbel(self) -> Gumbel:
        """
        Create Gumbel copula.
        """

        my_copula = Gumbel(theta=self.theta)

        return my_copula

    def _create_frank(self)-> Frank:
        """
        Create Frank copula.
        """

        my_copula = Frank(theta=self.theta)

        return my_copula

    def _create_clayton(self) -> Clayton:
        """
        Create Clayton copula.
        """

        my_copula = Clayton(theta=self.theta)

        return my_copula

    def _create_joe(self) -> Joe:
        """
        Create Joe copula.
        """

        my_copula = Joe(theta=self.theta)

        return my_copula

    def _create_n13(self) -> N13:
        """
        Create N13 copula.
        """

        my_copula = N13(theta=self.theta)

        return my_copula

    def _create_n14(self) -> N14:
        """
        Create N14 copula.
        """

        my_copula = N14(theta=self.theta)

        return my_copula

    def _create_gaussian(self) -> Gaussian:
        """
        Create Gaussian copula.
        """

        my_copula = Gaussian(cov=self.cov)

        return my_copula

    def _create_student(self) -> Student:
        """
        Create Student copula.
        """

        my_copula = Student(nu=self.nu, cov=self.cov)

        return my_copula
