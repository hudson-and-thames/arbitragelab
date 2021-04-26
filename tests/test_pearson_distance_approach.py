# Copyright 2021, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests the pearson distance approach from the Distance Approach module of ArbitrageLab.
"""
import unittest
import os

import pandas as pd
import numpy as np

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
        risk_free = pd.Series(data=np.array([0.01]*2141))
        risk_free.index = data.index
        risk_free.rename("risk_free")

        # Datasets for portfolio formation and trading steps of the strategy
        self.train_data = data[:252 * 5 - 1]  # 5 years
        self.test_data = data[252 * 5 - 1:252 * 6 - 1]  # 1year
        self.risk_free_train = risk_free[:252 * 5 - 1]
        self.risk_free_test = risk_free[252 * 5 - 1:252 * 6 - 1]

    def test_form_portfolio(self):
        """
        Tests the generation of portfolios from the PearsonStrategy class.
        """
        # Three different strategies
        strategy_basic = PearsonStrategy()
        strategy_risk_free = PearsonStrategy()
        strategy_corr_weight = PearsonStrategy()

        # Performing the portfolio formation step
        strategy_basic.form_portfolio(self.train_data, long_pct=0.05, short_pct=0.05)
        strategy_risk_free.form_portfolio(self.train_data, self.risk_free_train)
        strategy_corr_weight.form_portfolio(self.train_data, weight="correlation")

        # Testing the last month data of the formation period
        self.assertAlmostEqual(strategy_basic.last_month.mean(), 1.017491, delta=1e-5)
        self.assertAlmostEqual(strategy_risk_free.last_month.mean(), 1.017491, delta=1e-5)
        self.assertAlmostEqual(strategy_corr_weight.last_month.mean(), 1.017491, delta=1e-5)

        # Testing the long and short percentage for the strategy_basic
        self.assertEqual(strategy_basic.long_pct, 0.05)
        self.assertEqual(strategy_basic.short_pct, 0.05)

        # Testing the monthly return for the formation period
        self.assertAlmostEqual(strategy_basic.monthly_return.mean().mean(), 0.999993, delta=1e-5)
        self.assertAlmostEqual(strategy_risk_free.monthly_return.mean().mean(), 0.999993, delta=1e-5)
        self.assertAlmostEqual(strategy_corr_weight.monthly_return.mean().mean(), 0.999993, delta=1e-5)

        # Testing the risk free rate for the formation period
        self.assertAlmostEqual(strategy_basic.risk_free.mean(), 0.0, delta=1e-5)
        self.assertAlmostEqual(strategy_risk_free.risk_free.mean(), 0.01, delta=1e-5)
        self.assertAlmostEqual(strategy_corr_weight.risk_free.mean(), 0.0, delta=1e-5)

        # Testing the beta value for the stocks
        self.assertAlmostEqual(sum(strategy_basic.beta_dict.values()), 23.038961, delta=1e-5)
        self.assertAlmostEqual(sum(strategy_risk_free.beta_dict.values()), 23.038961, delta=1e-5)
        self.assertAlmostEqual(sum(strategy_corr_weight.beta_dict.values()), 34.371443, delta=1e-5)

        # Testing the pairs
        expected_pairs = ['EPP', 'EFA', 'XLB', 'EWG', 'VGK', 'VPL', 'EWU', 'EWQ', 'FXI', 'SPY', 'DIA', 'XLK', 'XLE',
                          'XLF', 'EWJ', 'XLU', 'CSJ', 'TIP', 'LQD', 'BND', 'IEF', 'TLT']
        self.assertCountEqual(strategy_basic.pairs_dict['EEM'], expected_pairs)
        self.assertCountEqual(strategy_risk_free.pairs_dict['EEM'], expected_pairs)
        self.assertCountEqual(strategy_corr_weight.pairs_dict['EEM'], expected_pairs)

    def test_trade_portfolio(self):
        """
        Tests the generation of trading signals in the test phase
        """

        # Basic Strategy
        strategy_no_test = PearsonStrategy()
        strategy_test = PearsonStrategy()

        # Performing the portfolio formation step
        strategy_no_test.form_portfolio(self.train_data)
        strategy_test.form_portfolio(self.train_data, self.risk_free_train)

        # Generating trading signal
        strategy_no_test.trade_portfolio()
        strategy_test.trade_portfolio(self.test_data, self.risk_free_test)

        # Testing trading signals
        self.assertAlmostEqual(strategy_no_test.trading_signal.mean(), -0.043478, delta=1e-5)
        self.assertAlmostEqual(strategy_test.trading_signal.mean().mean(), -0.043478, delta=1e-5)

        # Testing monthly return and risk free rate in test period
        self.assertAlmostEqual(strategy_test.test_monthly_return.mean().mean(), 1.007467, delta=1e-5)
        self.assertAlmostEqual(strategy_test.risk_free.mean().mean(), 0.01, delta=1e-5)

    def test_get_trading_signal(self):

        strategy = PearsonStrategy()

        strategy.form_portfolio(self.train_data)

        strategy.trade_portfolio(self.test_data)

        pd.testing.assert_frame_equal(strategy.trading_signal, strategy.get_trading_signal())

    def test_get_beta_dict(self):

        strategy = PearsonStrategy()

        strategy.form_portfolio(self.train_data)

        self.assertAlmostEqual(sum(strategy.get_beta_dict().values()), sum(strategy.beta_dict.values()), delta=1e-5)

    def test_get_pairs_dict(self):

        strategy = PearsonStrategy()

        strategy.form_portfolio(self.train_data)

        expected_pairs = ['EPP', 'EFA', 'XLB', 'EWG', 'VGK', 'VPL', 'EWU', 'EWQ', 'FXI', 'SPY', 'DIA', 'XLK', 'XLE',
                          'XLF', 'EWJ', 'XLU', 'CSJ', 'TIP', 'LQD', 'BND', 'IEF', 'TLT']

        self.assertCountEqual(strategy.get_pairs_dict()['EEM'], expected_pairs)
