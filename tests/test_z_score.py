# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests class of the Spread Trading module:
spread_trading/z_score.py
"""

import os
import unittest

import pandas as pd

from arbitragelab.spread_trading import BollingerBandsTradingRule


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
            strategy.update_trades(update_timestamp=self.spread_series.index[ind], update_value=self.spread_series[ind])

        self.assertEqual(len(strategy.open_trades), 3)
        self.assertEqual(len(strategy.closed_trades), 1)
