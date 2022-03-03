# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests spread construction of Hedge Ratios - used in Pairs Selection module:
"""
# pylint: disable=protected-access

import os
import unittest
import pandas as pd
import numpy as np

from arbitragelab.hedge_ratios import construct_spread


class TestSpreadConstruction(unittest.TestCase):
    """
    Tests construct_spread class.
    """

    def setUp(self):
        """
        Loads price universe and instantiates the pairs selection class.
        """

        np.random.seed(0)

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/sp100_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.data.dropna(inplace=True)

    def test_spread_construction(self):
        """
        Verifies spread construction function.
        """

        hedge_ratios = pd.Series({'A': 1, 'AVB': 0.832406370860649})
        spread = construct_spread(self.data[['AVB', 'A']], hedge_ratios=hedge_ratios)
        inverted_spread = construct_spread(self.data[['AVB', 'A']], hedge_ratios=hedge_ratios, dependent_variable='A')
        self.assertAlmostEqual(spread.mean(), -81.853, delta=1e-4)
        self.assertEqual((spread - inverted_spread).sum(), 0)
