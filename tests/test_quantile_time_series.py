# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests AUTO ARIMA prediction functions.
"""

import unittest
import numpy as np
import pandas as pd

from arbitragelab.time_series_approach import QuantileTimeSeriesTradingStrategy


class TestQuantileTimeSeries(unittest.TestCase):
    """
    Tests Auto ARIMA predictions.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """

        spread_data = [-0.2, 0.3, 0.5, 1.7, 1.0, 0.0, -5, -6, -9,
                       -7, -2, 1, 1.1, 1.2, 1.3, 1.4, 1.8, 3, 0.2]

        forecast_data = [-0.21, 0.35, 0.55, 1.6, 1.0, 0.0, -5.5, -6,
                         -9.1, -7.1, -2.1, 1, 1.1, 1.3, 1.5, 1.9, 2,
                         0.2, 5]
        self.spread_series = pd.Series(spread_data)
        self.forecast_series = pd.Series(forecast_data)

    def test_time_series_strategy(self):
        """
        Tests get_trend_order function.
        """

        trading_strategy = QuantileTimeSeriesTradingStrategy()
        trading_strategy.fit_thresholds(self.spread_series)
        trading_strategy.plot_thresholds()
        self.assertAlmostEqual(trading_strategy.short_diff_threshold, -4, delta=1e-2)
        self.assertAlmostEqual(trading_strategy.long_diff_threshold, 2.9, delta=1e-2)

        # Test predictions
        for pred, actual in zip(self.forecast_series.shift(-1), self.spread_series):
            trading_strategy.get_allocation(pred-actual, exit_threshold=0)

        self.assertEqual(trading_strategy.positions[5], -1)
        self.assertEqual(trading_strategy.positions[9], 1)
        self.assertEqual(trading_strategy.positions[18], 0)
        self.assertAlmostEqual(np.mean(trading_strategy.positions), 0.21, delta=1e-2)
