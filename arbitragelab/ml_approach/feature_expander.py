# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the Feature Expansion class.
"""

import itertools
import numpy as np
import pandas as pd

from arbitragelab.util import devadarsh

# This silencer is related to dangerous-default-value in __init__
# pylint: disable=W0102

class FeatureExpander:
    """
    Higher-order term Feature Expander implementation. The implementation consists
    of two major parts. The first part consists of using a collection of orthogonal
    polynomials' coefficients, ordered from lowest order term to highest. The implemented
    series are ['chebyshev', 'legendre', 'laguerre', 'power'] polynomials. The second part is a combinatorial
    version of feature crossing, which involves the generation of feature collections
    of the n order and multiplying them together. This can be used by adding ['product']
    in the 'methods' parameter in the constructor.
    """

    def __init__(self, methods: list = [], n_orders: int = 1):
        """
        Initializes main variables.

        :param methods: (list) Possible expansion methods are ['chebyshev', 'legendre',
            'laguerre', 'power', 'product'].
        :param n_orders: (int) Number of orders.
        """

        self.methods = methods
        self.n_orders = n_orders
        self.dataset = None

        devadarsh.track('FeatureExpander')

    @staticmethod
    def _chebyshev(series: pd.Series, degree: int):
        """
        Retrieves the chebyshev polynomial coefficients of a specific
        degree.

        :param series: (pd.Series) Series to use.
        :param degree: (int) Degree to use.
        :return: (np.array) Resulting polynomial.
        """

        return np.polynomial.chebyshev.chebvander(series, degree)

    @staticmethod
    def _legendre(series: pd.Series, degree: int):
        """
        Retrieves the legendre polynomial coefficients of a specific
        degree.

        :param series: (pd.Series) Series to use.
        :param degree: (int) Degree to use.
        :return: (np.array) Resulting polynomial.
        """

        return np.polynomial.legendre.legvander(series, degree)

    @staticmethod
    def _laguerre(series: pd.Series, degree: int):
        """
        Retrieves the laguerre polynomial coefficients of a specific
        degree.

        :param series: (pd.Series) Series to use.
        :param degree: (int) Degree to use.
        :return: (np.array) Resulting polynomial.
        """

        return np.polynomial.laguerre.lagvander(series, degree)

    @staticmethod
    def _power(series: pd.Series, degree: int):
        """
        Retrieves the power polynomial coefficients of a specific
        degree.

        :param series: (pd.Series) Series to use.
        :param degree: (int) Degree to use.
        :return: (np.array) Resulting polynomial.
        """

        return np.polynomial.polynomial.polyvander(series, degree)

    @staticmethod
    def _product(series: pd.Series, degree: int):
        """
        Implements the feature crossing method of feature expansion,
        which involves the generation of feature groups and appending
        the resulting product to the original series.

        :param series: (pd.Series) Series to use.
        :param degree: (int) Degree to use.
        :return: (pd.DataFrame) Resulting polynomial.
        """

        # Get feature count.
        comb_range = range(len(series[0]))

        # Generate N degree combinations in relation to feature count.
        combinations = [list(comb) for comb in itertools.combinations(comb_range, degree)]

        vectorized_x = pd.DataFrame(series)

        # N-wise product for C combinations.
        return [np.prod(vectorized_x.iloc[:, comb], axis=1) for comb in combinations]

    def fit(self, frame: pd.DataFrame):
        """
        Stores the dataset inside the class object.

        :param frame: (pd.DataFrame) Dataset to store.
        """

        self.dataset = frame

        return self

    def transform(self) -> pd.DataFrame:
        """
        Returns the original dataframe with features requested from
        the 'methods' parameter in the constructor.

        :return: (pd.DataFrame) Original DataFrame with the expanded values appended to it.
        """

        new_dataset = []

        for row in self.dataset:
            expanded_row = list(row)
            for meth in self.methods:
                if meth != "product":
                    # Dynamically call the needed method using 'getattr'.
                    math_return = getattr(self, '_' + meth)(row, self.n_orders)
                    # Ravel result and concatenate it to expanded_row.
                    expanded_row.extend(np.ravel(math_return))

            new_dataset.append(np.ravel(expanded_row).tolist())

        new_dataset_df = pd.DataFrame(new_dataset)

        if "product" in self.methods:
            # Dynamically call the '_product' method using 'getattr' method.
            prod_result = getattr(self, '_product')(self.dataset, self.n_orders)
            # Transpose the result to make it compatible with original structure.
            prod_result_df = pd.DataFrame(prod_result).T

            # Return concatenated dataset parts.
            return pd.concat([new_dataset_df, prod_result_df], axis=1)

        return new_dataset_df
