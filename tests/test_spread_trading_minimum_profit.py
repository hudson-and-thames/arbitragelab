# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests class of the Spread Trading module:
spread_trading/minimum_profit.py
"""

import os
import unittest

import pandas as pd
import numpy as np

from arbitragelab.spread_trading import MinimumProfitTradingRule


class TestMinimumProfitTradingRule(unittest.TestCase):
    """
    Test MinimumProfitTradingRule functions.
    """

    def setUp(self):
        """
        Creates spread and variables to use the minimum profit trading rule on.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'

        price_data = pd.read_csv(data_path, parse_dates=True, index_col="Date")[['EEM', 'EWG']]
        beta = -1.6235743

        self.spread_series = price_data['EEM'] + beta * price_data['EWG']
        self.shares = np.array([10, 15])
        self.optimal_levels = np.array([-5.63296941, -4.77296941, -3.91296941])

    def test_strategy_normal_use(self):
        """
        Tests the normal use of the strategy, feeding spread value by value.
        """

        strategy = MinimumProfitTradingRule(self.shares, self.optimal_levels)

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
        self.assertEqual(len(strategy.closed_trades), 30)

        self.assertEqual(list(strategy.closed_trades.keys())[0].to_datetime64(),
                         pd.Timestamp('2008-01-03 00:00+00:00').to_datetime64())
        self.assertEqual(list(strategy.closed_trades.keys())[10].to_datetime64(),
                         pd.Timestamp('2008-09-15 00:00+00:00').to_datetime64())
        self.assertEqual(list(strategy.closed_trades.keys())[20].to_datetime64(),
                         pd.Timestamp('2008-12-19 00:00+00:00').to_datetime64())
