# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests the basic distance approach from the Distance Approach module of ArbitrageLab.
"""
import unittest
import os

import numpy as np
import pandas as pd
import matplotlib

from arbitragelab.distance_approach.basic_distance_approach import DistanceStrategy


class TestDistanceStrategy(unittest.TestCase):
    """
    Test the Distance Strategy module.
    """

    def setUp(self):
        """
        Sets the file path for the data.
        """

        # Using saved ETF price series for testing and trading
        project_path = os.path.dirname(__file__)
        data_path = project_path + "/test_data/stock_prices.csv"
        data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

        # Generate a random industry dictionary for testing
        industry_dict = {}
        for index, ticker in enumerate(data.columns):
            if index < len(data.columns) * (1 / 3):
                industry_dict[ticker] = 'Information Technology'
            elif index < len(data.columns) * (2 / 3):
                industry_dict[ticker] = 'Financials'
            else:
                industry_dict[ticker] = 'Industrials'

        # Datasets for pairs formation and trading steps of the strategy
        self.train_data = data[:100]
        self.test_data = data[100:]
        self.industry_dict = industry_dict

    def test_form_pairs(self):
        """
        Tests the generation of pairs from the DistanceStrategy class.
        """
        # Basic Strategy
        strategy = DistanceStrategy()

        # Industry based Strategy
        strategy_industry = DistanceStrategy()

        # Zero-crossing based Strategy
        strategy_zero_crossing = DistanceStrategy()

        # Variance based Strategy
        strategy_variance = DistanceStrategy()

        # Performing the pairs formation step
        strategy.form_pairs(self.train_data, method = 'standard', num_top=5, skip_top=0)
        strategy_industry.form_pairs(self.train_data, method='industry',
                                     industry_dict=self.industry_dict, num_top=5, skip_top=0)
        strategy_zero_crossing.form_pairs(self.train_data, method='zero_crossing', num_top=5, skip_top=0)
        strategy_variance.form_pairs(self.train_data, method='variance', num_top=5, skip_top=0)

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

    def test_trade_pairs(self):
        """
        Tests the generation of trading signals from the DistanceStrategy class.
        """

        strategy = DistanceStrategy()

        # Performing the pairs formation step
        strategy.form_pairs(self.train_data, num_top=5, skip_top=0)

        # Performing the signals generation step
        strategy.trade_pairs(self.test_data, divergence=2)

        # Testing normalized data and portfolio price series values
        self.assertAlmostEqual(strategy.normalized_data.mean().mean(), 0.357394, delta=1e-5)
        self.assertAlmostEqual(strategy.portfolios.mean().mean(), -0.083095, delta=1e-5)

        # Testing values of trading signals
        self.assertAlmostEqual(strategy.trading_signals.sum().mean(), -887.0, delta=1e-5)

    def test_trade_pairs_numpy(self):
        """
        Tests the generation of trading signals using np.array as input from the DistanceStrategy class.
        """

        strategy = DistanceStrategy()

        # Alternative data input format
        np_train_data = self.train_data.to_numpy()
        names_train_data = list(self.train_data.columns)
        strategy.form_pairs(np_train_data, num_top=5, skip_top=0, list_names=names_train_data)

        # Alternative data input format
        np_test_data = self.test_data.to_numpy()
        strategy.trade_pairs(np_test_data, divergence=2)

        # Testing normalized data and portfolio price series values
        self.assertAlmostEqual(strategy.normalized_data.mean().mean(), 0.357394, delta=1e-5)
        self.assertAlmostEqual(strategy.portfolios.mean().mean(), -0.083095, delta=1e-5)

        # Testing values of trading signals
        self.assertAlmostEqual(strategy.trading_signals.sum().mean(), -887.0, delta=1e-5)

    def test_get_signals(self):
        """
        Tests the output of generated trading signals.
        """

        strategy = DistanceStrategy()

        strategy.form_pairs(self.train_data, num_top=5, skip_top=0)

        strategy.trade_pairs(self.test_data, divergence=2)

        pd.testing.assert_frame_equal(strategy.trading_signals, strategy.get_signals())

    def test_get_portfolios(self):
        """
        Tests the output of portfolios series.
        """

        strategy = DistanceStrategy()

        strategy.form_pairs(self.train_data, num_top=5, skip_top=0)

        strategy.trade_pairs(self.test_data, divergence=2)

        pd.testing.assert_frame_equal(strategy.portfolios, strategy.get_portfolios())

    def test_get_scaling_parameters(self):
        """
        Tests the output of scaling parameters to use.
        """

        strategy = DistanceStrategy()

        strategy.form_pairs(self.train_data, num_top=5, skip_top=0)

        strategy.trade_pairs(self.test_data, divergence=2)

        pd.testing.assert_series_equal(strategy.min_normalize.rename('min_value'),
                                       strategy.get_scaling_parameters()['min_value'])
        pd.testing.assert_series_equal(strategy.max_normalize.rename('max_value'),
                                       strategy.get_scaling_parameters()['max_value'])

    def test_plot_portfolio(self):
        """
        Tests the plotting of a portfolio value and trading signals.
        """

        strategy = DistanceStrategy()

        strategy.form_pairs(self.train_data, num_top=5, skip_top=0)

        strategy.trade_pairs(self.test_data, divergence=2)

        self.assertTrue(isinstance(strategy.plot_portfolio(1), matplotlib.figure.Figure))

    def test_plot_pair(self):
        """
        Tests the plotting of a pair price series and trading signals.
        """

        strategy = DistanceStrategy()

        strategy.form_pairs(self.train_data, num_top=5, skip_top=0)

        strategy.trade_pairs(self.test_data, divergence=2)

        self.assertTrue(isinstance(strategy.plot_pair(1), matplotlib.figure.Figure))

    def test_get_pairs(self):
        """
        Tests the output of pairs.
        """

        strategy = DistanceStrategy()

        strategy.form_pairs(self.train_data, num_top=5, skip_top=0)

        strategy.trade_pairs(self.test_data, divergence=2)

        expected_pairs = [('EFA', 'VGK'), ('EPP', 'VPL'), ('EWQ', 'VGK'),
                          ('EFA', 'EWQ'), ('EPP', 'SPY')]

        self.assertCountEqual(strategy.get_pairs(), expected_pairs)

    def test_get_num_crossing(self):
        """
        Tests the number of crossings.
        """

        strategy = DistanceStrategy()

        strategy.form_pairs(self.train_data, num_top=5, skip_top=0)

        expected_number_crossings = [22, 20, 17, 16, 31]

        self.assertEqual(list(strategy.get_num_crossing().values()), expected_number_crossings)

    def test_exceptions(self):
        """
        Tests exceptions from methods of the class.
        """

        strategy = DistanceStrategy()

        # When trying to generate trading signals without creating pairs first
        with self.assertRaises(Exception):
            strategy.trade_pairs(self.test_data, divergence=2)

        # When trying to get pairs with inappropriate method
        with self.assertRaises(Exception):
            strategy.selection_method(method='wrong input',num_top=5,skip_top=0)
