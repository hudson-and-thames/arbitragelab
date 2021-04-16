# Copyright 2021, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests the pearson distance approach from the Distance Approach module of ArbitrageLab.
"""
import unittest
import os

import numpy as np
import pandas as pd
import matplotlib

from arbitragelab.distance_approach.pearson_distance_approach import PearsonStrategy


class TestPearsonStrategy(unittest.TestCase):
    """
    Test the Pearson Strategy module.
    """

    def setUp(self):
        """
        Sets the file path for the data.
        """

        # Using saved ETF price series for testing and trading
        project_path = os.path.dirname(__file__)
        data_path = project_path + "/test_data/stock_prices.csv"
        data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

        # Datasets for pairs formation and trading steps of the strategy
        self.train_data = data[:100]
        self.test_data = data[100:]

    def test_form_portfolio(self):
        """
        Tests the generation of portfolios from the PearsonStrategy class.
        """
        # Basic Strategy
        strategy = PearsonStrategy()

        # Performing the pairs formation step
        strategy.form_pairs(self.train_data, method = 'standard', num_top=5, skip_top=0)

        # Testing min and max values of series used for dataset normalization
        self.assertAlmostEqual(strategy.min_normalize.mean(), 63.502009, delta=1e-5)
        self.assertAlmostEqual(strategy.max_normalize.mean(), 72.119729, delta=1e-5)

        # Testing values of historical volatility for portfolios
        self.assertAlmostEqual(np.mean(list(strategy.train_std.values())), 0.056361, delta=1e-5)

        # Testing values of train portfolio which was created to get the number of zero crossings
        self.assertAlmostEqual(strategy_industry.train_portfolio.mean().mean(), 0.011405, delta=1e-5)

        # Testing the number of zero crossings for the pairs
        self.assertAlmostEqual(np.mean(list(strategy_industry.num_crossing.values())), 16.4, delta=1e-5)

        # Testing the list of created pairs for both of the cases
        expected_pairs = [('EFA', 'VGK'), ('EPP', 'VPL'), ('EWQ', 'VGK'),
                          ('EFA', 'EWQ'), ('EPP', 'SPY')]
        expected_pairs_industry = [('EFA', 'EWQ'), ('EFA', 'EWU'), ('SPY', 'VPL'),
                                   ('EEM', 'EWU'), ('DIA', 'SPY')]
        expected_pairs_zero_crossing = [('EPP', 'SPY'), ('DIA', 'EWJ'), ('EEM', 'EWJ'),
                                        ('EEM', 'EFA'), ('EWU', 'VPL')]
        expected_pairs_variance = [('IEF', 'TIP'), ('EWU', 'FXI'), ('SPY', 'VGK'),
                                   ('EWJ', 'EWQ'), ('EEM', 'EWQ')]

        self.assertCountEqual(strategy.pairs, expected_pairs)
        self.assertCountEqual(strategy_industry.pairs, expected_pairs_industry)
        self.assertCountEqual(strategy_zero_crossing.pairs, expected_pairs_zero_crossing)
        self.assertCountEqual(strategy_variance.pairs, expected_pairs_variance)




