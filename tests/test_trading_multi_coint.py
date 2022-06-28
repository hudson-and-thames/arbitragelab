# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests class of the Spread Trading module:
trading/cointegration_approach/multi_coint.py
"""

import os
import unittest

import pandas as pd

from arbitragelab.trading.multi_coint import MultivariateCointegrationTradingRule


class TestMultivariateCointegrationTradingRule(unittest.TestCase):
    """
    Test MultivariateCointegrationTradingRule functions.
    """

    def setUp(self):
        """
        Creates spread and variables to use the multivariate cointegration profit trading rule on.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'

        self.price_values = pd.read_csv(data_path, parse_dates=True, index_col="Date")[['EEM', 'EWG', 'IEF']].iloc[:200]

        self.coint_vec = pd.Series({'EEM': 0.778721,
                                    'EWG': 4.545739,
                                    'IEF': -6.459130})

    def test_strategy_normal_use(self):
        """
        Tests the normal use of the strategy, feeding spread value by value.
        """

        strategy = MultivariateCointegrationTradingRule(self.coint_vec)

        # Add initial spread value
        strategy.update_price_values(self.price_values.iloc[0])

        # Run over next
        for ind in range(1, len(self.price_values[1:])):

            time = self.price_values.index[ind]
            value = self.price_values.iloc[ind]

            strategy.update_price_values(value)

            # Getting signal
            pos_shares, neg_shares, pos_notional, neg_notional = strategy.get_signal()

            strategy.add_trade(start_timestamp=time, pos_shares=pos_shares, neg_shares=neg_shares)

            strategy.update_trades(update_timestamp=time)

        self.assertEqual(len(strategy.open_trades), 0)
        self.assertEqual(len(strategy.closed_trades), 198)

        self.assertEqual(pos_shares['EEM'], 51067.0)
        self.assertEqual(pos_shares['EWG'], 420773.0)
        self.assertEqual(neg_shares['IEF'], -115381.0)

        self.assertAlmostEqual(pos_notional['EEM'], 1462558.848, 2)
        self.assertAlmostEqual(pos_notional['EWG'], 8537484.555, 2)
        self.assertAlmostEqual(neg_notional['IEF'], -10000071.058, 2)

        self.assertEqual(list(strategy.closed_trades.keys())[0].to_datetime64(),
                         pd.Timestamp('2008-01-03 00:00+00:00').to_datetime64())
        self.assertEqual(list(strategy.closed_trades.keys())[50].to_datetime64(),
                         pd.Timestamp('2008-03-17 00:00+00:00').to_datetime64())
        self.assertEqual(list(strategy.closed_trades.keys())[100].to_datetime64(),
                         pd.Timestamp('2008-05-28 00:00+00:00').to_datetime64())


    def test_get_signal_warning(self):
        """
        Tests the warning being raised when trying to get signal without providing data.
        """

        strategy = MultivariateCointegrationTradingRule(self.coint_vec)

        # No data given, warning need to be raised
        with self.assertWarns(Warning):
            strategy.get_signal()

    def test_no_positions_to_close(self):
        """
        Tests the situation when no positions should be closed.
        """

        strategy = MultivariateCointegrationTradingRule(self.coint_vec)

        # Add initial spread value
        strategy.update_price_values(self.price_values.iloc[0])
        strategy.update_price_values(self.price_values.iloc[1])
        time = self.price_values.index[1]

        # Getting signal
        pos_shares, neg_shares, _, _ = strategy.get_signal()
        strategy.add_trade(start_timestamp=time, pos_shares=pos_shares, neg_shares=neg_shares)

        strategy.update_trades(update_timestamp=time)

        # Nothing to close the second time
        strategy.update_trades(update_timestamp=time)

        self.assertEqual(len(strategy.open_trades), 0)
        self.assertEqual(len(strategy.closed_trades), 1)

        self.assertEqual(list(strategy.closed_trades.keys())[0].to_datetime64(),
                         pd.Timestamp('2008-01-03 00:00+00:00').to_datetime64())
