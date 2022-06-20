# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Abstract class for pairs copulas implementation.

Also include a Switcher class to create copula by its name and parameters,
to emulate a switch functionality.
"""

# pylint: disable = invalid-name, too-many-function-args
from typing import Callable

from arbitragelab.copula_approach.archimedean import (Gumbel, Frank, Clayton, Joe, N13, N14)
from arbitragelab.copula_approach.elliptical import (GaussianCopula, StudentCopula)
from arbitragelab.util import segment


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

        segment.track('SwitcherCopula')

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

    def _create_frank(self) -> Frank:
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

    def _create_gaussian(self) -> GaussianCopula:
        """
        Create Gaussian copula.
        """

        my_copula = GaussianCopula(cov=self.cov)

        return my_copula

    def _create_student(self) -> StudentCopula:
        """
        Create Student copula.
        """

        my_copula = StudentCopula(nu=self.nu, cov=self.cov)

        return my_copula
