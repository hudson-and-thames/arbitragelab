# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests class of the Spread Trading module:
trading/cointegration_approach/z_score.py
"""

import os
import unittest
from collections import deque

import pandas as pd

from arbitragelab.trading.z_score import BollingerBandsTradingRule


class TestBollingerBandsTradingRule(unittest.TestCase):
    """
    Test BollingerBandsTradingRule functions.
    """

    def setUp(self):
        """
        Creates spread to use the bollinger bands trading rule on.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'

        price_data = pd.read_csv(data_path, parse_dates=True, index_col="Date")[['EEM', 'EWG']]
        hedge_ratios = {'EEM': 1.0, 'EWG': 1.5766259695851286}

        weighted_prices = price_data * hedge_ratios
        non_dependent_variables = [x for x in weighted_prices.columns if x != 'EEM']

        self.spread_series = weighted_prices['EEM'] - weighted_prices[non_dependent_variables].sum(axis=1)

    def test_strategy_normal_use(self):
        """
        Tests the normal use of the strategy, feeding spread value by value.
        """

        strategy = BollingerBandsTradingRule(10, 10, entry_z_score=2.5, exit_z_score_delta=3)

        # Add initial spread value
        strategy.update_spread_value(self.spread_series[0])

        # Run over next
        for ind in range(1, len(self.spread_series[1:])):
            strategy.update_spread_value(self.spread_series[ind])
            trade, side = strategy.check_entry_signal()

            if trade:
                strategy.add_trade(start_timestamp=self.spread_series.index[ind], side_prediction=side)
            strategy.update_trades(update_timestamp=self.spread_series.index[ind])

        self.assertEqual(len(strategy.open_trades), 0)
        self.assertEqual(len(strategy.closed_trades), 3)

        self.assertEqual(list(strategy.closed_trades.keys())[0].to_datetime64(),
                         pd.Timestamp('2008-06-25 00:00+00:00').to_datetime64())
        self.assertEqual(list(strategy.closed_trades.keys())[1].to_datetime64(),
                         pd.Timestamp('2011-02-09 00:00+00:00').to_datetime64())
        self.assertEqual(list(strategy.closed_trades.keys())[2].to_datetime64(),
                         pd.Timestamp('2013-01-25 00:00+00:00').to_datetime64())

    def test_get_z_score(self):
        """
        Tests the use of the get_z_score method.
        """

        # Create a deque of spread values
        spread_slice = deque(maxlen=5)
        for element in self.spread_series[:5]:
            spread_slice.append(element)

        z_score = BollingerBandsTradingRule.get_z_score(spread_slice, 5, 5)

        self.assertAlmostEqual(z_score, 0.56609, delta=1e-5)

    def test_check_entry_signal_zero_std(self):
        """
        Tests the generation of a negative signal if std of spread is zero.
        """

        strategy = BollingerBandsTradingRule(5, 5, entry_z_score=2.5, exit_z_score_delta=3)

        # Feed same values
        for _ in range(5):
            strategy.update_spread_value(0.5)
        signal, _ = strategy.check_entry_signal()

        self.assertTrue(not signal)
