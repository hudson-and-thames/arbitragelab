# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This module implements the Feature Expansion class.
"""

import numpy as np

# pylint: disable=W0102

class FeatureExpander:
    """
    Higher order term Feature Expander implementation.
    """

    def __init__(self, methods=[], n_orders=1):
        """

        :param methods: (list) Possible expansion methods [chebyshev, legendre, laguerre, power].
        :param n_orders: (int) Number of orders.
        """
        self.methods = methods
        self.n_orders = n_orders
        self.dataset = None

    @staticmethod
    def _chebyshev(series, degree):
        """

        :param series: (pd.Series)
        :param degree: (int)
        """

        return np.polynomial.chebyshev.chebvander(series, degree)

    @staticmethod
    def _legendre(series, degree):
        """

        :param series: (pd.Series)
        :param degree: (int)
        """

        return np.polynomial.legendre.legvander(series, degree)

    @staticmethod
    def _laguerre(series, degree):
        """

        :param series: (pd.Series)
        :param degree: (int)
        """

        return np.polynomial.laguerre.lagvander(series, degree)

    @staticmethod
    def _power(series, degree):
        """

        :param series: (pd.Series)
        :param degree: (int)
        """

        return np.polynomial.polynomial.polyvander(series, degree)

    def fit(self, frame):
        """


        :param frame: (np.array) dataset
        """
        self.dataset = frame
        return self

    def transform(self) -> list:
        """
        Transform data to polynomial features

        :return: List of lists of the expanded values.
        """
        new_dataset = []

        for row in self.dataset.values:
            expanded_row = list(row)
            for degree in range(1, self.n_orders):
                for meth in self.methods:
                    expanded_row.extend(
                        np.ravel(getattr(self, '_' + meth)(row, degree)))

            new_dataset.append(np.ravel(expanded_row).tolist())

        return new_dataset
