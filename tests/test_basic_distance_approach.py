# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
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

        # Datasets for pairs formation and trading steps of the strategy
        self.train_data = data[:100]
        self.test_data = data[100:]

    def test_form_pairs(self):
        """
        Tests the generation of pairs from the DistanceStrategy class.
        """

        strategy = DistanceStrategy()

        # Performing the pairs formation step
        strategy.form_pairs(self.train_data, num_top=5, skip_top=0)

        # Testing min and max values of series used for dataset normalization
        self.assertAlmostEqual(strategy.min_normalize.mean(), 63.502009, delta=1e-5)
        self.assertAlmostEqual(strategy.max_normalize.mean(), 72.119729, delta=1e-5)

        # Testing values of historical volatility for portfolios
        self.assertAlmostEqual(np.mean(list(strategy.train_std.values())), 0.056361, delta=1e-5)

        # Testing the list of created pairs
        expected_pairs = [('EFA', 'VGK'), ('EPP', 'VPL'), ('EWQ', 'VGK'),
                          ('EFA', 'EWQ'), ('EPP', 'SPY')]

        self.assertCountEqual(strategy.pairs, expected_pairs)

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

    def test_exceptions(self):
        """
        Tests exceptions from methods of the class.
        """

        strategy = DistanceStrategy()

        # When trying to generate trading signals without creating pairs first
        with self.assertRaises(Exception):
            strategy.trade_pairs(self.test_data, divergence=2)
