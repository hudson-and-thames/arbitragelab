# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module that uses copula for trading strategy based method described in the following article.

`B Sabino da Silva, F., Ziegelman, F. and Caldeira, J., 2017. Mixed Copula Pairs Trading Strategy on the S&P 500.
Flávio and Caldeira, João, Mixed Copula Pairs Trading Strategy on the S&P, 500.
<https://www.researchgate.net/profile/Fernando_Sabino_Da_Silva/publication/315878098_Mixed_Copula_Pairs_Trading_Strategy_on_the_SP_500/links/5c6f080b92851c695036785f/Mixed-Copula-Pairs-Trading-Strategy-on-the-S-P-500.pdf>`__

Note: Algorithm for fitting mixed copula was adapted from

`Cai, Z. and Wang, X., 2014. Selection of mixed copula model via penalized likelihood. Journal of the American
Statistical Association, 109(506), pp.788-801.
<https://www.tandfonline.com/doi/pdf/10.1080/01621459.2013.873366?casa_token=sey8HrojSgYAAAAA:TEMBX8wLYdGFGyM78UXSYm6hXl1Qp_K6wiLgRJf6kPcqW4dYT8z3oA3I_odrAL48DNr3OSoqkQsEmQ>`__
"""

# pylint: disable = invalid-name, too-many-locals
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from arbitragelab.copula_approach.base import Copula


class MixedCopula(Copula, ABC):
    """
    Class template for mixed copulas.
    """

    def __init__(self):
        """
        Initiate the MixedCopula class.
        """
        super().__init__()
        self.weights = None
        self.copulas = None

    def describe(self) -> pd.Series:
        """
        Describe the components and coefficients of the mixed copula.

        The description includes descriptive name, class name, the copula dependency parameter for each mixed copula
        component and their weights.

        :return: (pd.Series) The description of the specific mixed copula.
        """

        description = pd.Series(self._get_param())

        return description

    @abstractmethod
    def _get_param(self):
        """
        Get the parameters of the mixed copula.
        """

    def get_cop_density(self, u: float, v: float, eps: float = 1e-5) -> float:
        """
        Calculate probability density of the bivariate copula: P(U=u, V=v).

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

        # linear combo w.r.t. weights for each copula in the mix
        pdf = np.sum([self.weights[i] * cop.c(u, v) for i, cop in enumerate(self.copulas)])

        return pdf

    def get_cop_eval(self, u: float, v: float, eps: float = 1e-4) -> float:
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).

        Result is analytical except for Student-t copula. Also at the u and v will be remapped into [eps, 1-eps] to
        avoid edge values that may result in infinity or NaN.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :param eps: (float) Optional. The distance to the boundary 0 or 1, such that the value u, v will be mapped
            back. Defaults to 1e-4.
        :return: (float) The cumulative density.
        """

        # Mapping u, v back to the valid computational interval.
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # linear combo w.r.t. weights for each copula in the mix
        cdf = np.sum([self.weights[i] * cop.C(u, v) for i, cop in enumerate(self.copulas)])

        return cdf

    def get_condi_prob(self, u: float, v: float, eps: float = 1e-5) -> float:
        """
        Calculate conditional probability function: P(U<=u | V=v).

        Result is analytical. Also at the u and v will be remapped into [eps, 1-eps] to avoid edge values that may
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

        # linear combo w.r.t. weights for each copula in the mix
        result = np.sum([self.weights[i] * cop.condi_cdf(u, v) for i, cop in enumerate(self.copulas)])

        return result

    def sample(self, num_points: int) -> np.array:
        """
        Generate pairs according to P.D.F., stored in a 2D np.array of shape (num, 2).

        :param num_points: (int) Number of points to generate.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """

        # Generate a list of identities in {0, 1, 2} with given probability to determine which copula each
        # observation comes from. e.g. cop_identities[100, 2] means the 100th observation comes from copula 2
        cop_identities = np.random.choice([0, 1, 2], num_points, p=self.weights)

        # Generate random pairs from the copula given by cop_identities
        sample_pairs = np.zeros(shape=(num_points, 2))
        for i, cop_id in enumerate(cop_identities):
            sample_pairs[i] = self.copulas[cop_id].sample(num=1).flatten()

        return sample_pairs

    @staticmethod
    def _away_from_0(x: float, lower_limit: float = -1e-5, upper_limit: float = 1e-5) -> float:
        """
        Keep the parameter x away from 0 but still retain the sign.

        0 is remapped to the upper_limit.

        :param x: (float) The number to be remapped.
        :param lower_limit: (float) The lower limit to be considered a close enough to 0.
        :param upper_limit: (float) The upper limit to be considered a close enough to 0.
        :return: (float) The remapped parameter.
        """

        small_pos_bool = (0 <= x < upper_limit)  # Whether it is a small positive number
        small_neg_bool = (lower_limit < x < 0)  # Whether it is a small negative number
        small_bool = small_pos_bool or small_neg_bool  # Whether it is a small number
        # If not small, then return the param
        # If small, then return the corresponding limit.
        remapped_param = (x * int(not small_bool)
                          + upper_limit * int(small_pos_bool) + lower_limit * int(small_neg_bool))

        return remapped_param
