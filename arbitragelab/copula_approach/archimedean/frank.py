"""
Module that houses Frank copula class.
"""

# pylint: disable = invalid-name, too-many-lines
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from arbitragelab.copula_approach.base import Copula


class Frank(Copula):
    """
    Frank Copula.
    """

    def __init__(self, theta: float = None, threshold: float = 1e-10):
        r"""
        Initiate a Frank copula object.

        :param theta: (float) All reals except for 0, measurement of copula dependency.
        :param threshold: (float) Optional. Below this threshold, a percentile will be rounded to the threshold.
        """

        super().__init__('Frank')
        # Lower than this amount will be rounded to threshold
        self.threshold = threshold
        self.theta = theta  # Default input

    def sample(self, num: int = None, unif_vec: np.array = None) -> np.array:
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.

        User may choose to side-load independent uniformly distributed data in [0, 1]

        :param num: (int) Number of points to generate.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data.
            Default uses numpy pseudo-random generators.
        :return: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """

        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec.")

        theta = self.theta  # Use the default input

        # Generate pairs of indep uniform dist vectors. Use numpy to generate
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
        """
        Generate one pair of vectors from Frank copula.

        :param u1: (float) I.I.D. uniform random variable in [0,1].
        :param v2: (float) I.I.D. uniform random variable in [0,1].
        :param theta: (float) All reals except for 0, measurement of copula dependency.
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
               (et + eut * evt - et * eut - et * evt) ** 2)

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
        cdf = -1 / theta * np.log(
            1 + (np.exp(-1 * theta * u) - 1) * (np.exp(-1 * theta * v) - 1)
            / (np.exp(-1 * theta) - 1))

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

        # Numerically find the root
        result = brentq(lambda theta: kendall_tau(theta) - tau, -100, 100)

        return result
