# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This module implements the Feature Expansion class.
"""

import itertools
import numpy as np
import pandas as pd

# pylint: disable=W0102

class FeatureExpander:
    """
    Higher order term Feature Expander implementation.
    """

    def __init__(self, methods=[], n_orders=1):
        """

        :param methods: (list) Possible expansion methods [chebyshev, legendre,
            laguerre, power, product].
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

    @staticmethod
    def _product(series, degree):
        """
        
        :param series: (pd.Series)
        :param degree: (int)        
        """

        comb_range = range(len(series[0]))

        combinations = [list(comb) for comb in itertools.combinations(comb_range, degree)]

        vectorized_x = pd.DataFrame(series)

        return [np.prod(vectorized_x.iloc[:, comb], axis=1) for comb in combinations ]

    def fit(self, frame):
        """


        :param frame: (np.array) dataset
        """
        self.dataset = frame
        return self

    def transform(self) -> pd.DataFrame:
        """
        Transform data to polynomial features

        :return: Original DataFrame with the expanded values appended to it.
        """
        new_dataset = []

        for row in self.dataset:
            expanded_row = list(row)
            for meth in self.methods:
                if meth != "product":
                    expanded_row.extend( np.ravel( getattr(self, '_' + meth)(row, self.n_orders) ) )

            new_dataset.append( np.ravel(expanded_row).tolist() )

        new_dataset_df = pd.DataFrame(new_dataset)

        if "product" in self.methods:
            prod_result = getattr(self, '_product')(self.dataset, self.n_orders)
            prod_result_df = pd.DataFrame(prod_result).T

            return pd.concat([new_dataset_df, prod_result_df], axis=1)

        return new_dataset_df
