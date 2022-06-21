# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Gaussian copula implementation.
"""

# pylint: disable = invalid-name, too-many-lines, arguments-differ
import numpy as np
import scipy.stats as ss
from sklearn.covariance import EmpiricalCovariance

from arbitragelab.copula_approach.base import Copula
from arbitragelab.util import segment


class GaussianCopula(Copula):
    """
    Bivariate Gaussian Copula.
    """

    def __init__(self, cov: np.array = None):
        r"""
        Initiate a Gaussian copula object.

        :param cov: (np.array) Covariance matrix (NOT correlation matrix), measurement of covariance. The class will
            calculate correlation internally once the covariance matrix is given.
        """

        super().__init__('Gaussian')

        self.cov = None
        self.rho = None

        if cov is not None:
            self.cov = cov  # Covariance matrix
            # Correlation
            self.rho = cov[0][1] / (np.sqrt(cov[0][0]) * np.sqrt(cov[1][1]))

        segment.track('GaussianCopula')

    def sample(self, num: int = None) -> np.array:
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.

        User may choose to side-load independent uniformly distributed data in [0, 1].

        :param num: (int) Number of points to generate.
        :return: (np.array) Shape=(num, 2) array, sampled data for this copula.
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

        # Generate bivariate normal with mean 0 and intended covariance
        rand_generator = np.random.default_rng()
        result = rand_generator.multivariate_normal(mean=[0, 0], cov=cov, size=num)

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

    def fit(self, u: np.array, v: np.array) -> float:
        """
        Fit gaussian-copula to empirical data (pseudo-observations) and find cov/rho params. Once fit, `self.rho`, `self.cov` is updated.

        :param u: (np.array) 1D vector data of X pseudo-observations. Need to be uniformly distributed [0, 1].
        :param v: (np.array) 1D vector data of Y pseudo-observations. Need to be uniformly distributed [0, 1].
        :return: (float) Rho(correlation) parameter value.
        """

        super().fit(u, v)
        # 1. Calculate covariance matrix using sklearn
        # Correct matrix dimension for fitting in sklearn
        unif_data = np.array([u, v]).reshape(2, -1).T
        value_data = ss.norm.ppf(unif_data)  # Change from quantile to value

        # Getting empirical covariance matrix
        cov_hat = EmpiricalCovariance().fit(value_data).covariance_
        self.cov = cov_hat
        self.rho = cov_hat[0][1] / (np.sqrt(cov_hat[0][0]) * np.sqrt(cov_hat[1][1]))

        return self.rho

    def c(self, u: float, v: float) -> float:
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

        exp_ker = (rho * (-2 * inv_u * inv_v + inv_u ** 2 * rho + inv_v ** 2 * rho)
                   / (2 * (rho ** 2 - 1)))

        pdf = np.exp(exp_ker) / np.sqrt(1 - rho ** 2)

        return pdf

    def C(self, u: float, v: float) -> float:
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The cumulative density.
        """

        corr = [[1, self.rho], [self.rho, 1]]  # Correlation matrix
        inv_cdf_u = ss.norm.ppf(u)  # Inverse cdf of standard normal
        inv_cdf_v = ss.norm.ppf(v)
        mvn_dist = ss.multivariate_normal(mean=[0, 0], cov=corr)  # Joint cdf of multivariate normal
        cdf = mvn_dist.cdf((inv_cdf_u, inv_cdf_v))

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
