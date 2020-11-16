# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Tests the Kalman Filter Strategy from the Other Approaches module.
"""

import unittest
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from arbitragelab.other_approaches import KalmanFilterStrategy


class TestKalmanFilter(unittest.TestCase):
    """
    Test Kalman Filter Strategy functions.
    """

    def setUp(self):
        """
        Creates pairs data set from TIP and IEF tickers.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.kalman_filter = KalmanFilterStrategy()

    def test_kalman_filter_update(self):
        """
        Tests the update method of Kalman filter.
        """

        price_data_subset = self.data[['TIP', 'IEF']]
        linear_reg = LinearRegression(fit_intercept=True)  # Fit regression on the whole dataset
        linear_reg.fit(price_data_subset['TIP'].values.reshape(-1, 1), price_data_subset['IEF'])

        for _, row in price_data_subset.iterrows():
            self.kalman_filter.update(row['TIP'], row['IEF'])

        self.assertAlmostEqual(linear_reg.coef_[0], np.mean(self.kalman_filter.hedge_ratios), delta=0.02)
        self.assertAlmostEqual(np.mean(self.kalman_filter.spread_series), 0.089165, delta=1e-5)
        self.assertAlmostEqual(np.mean(self.kalman_filter.spread_std_series), 1.109202, delta=1e-5)

    def test_kalman_filter_trading_signals(self):
        """
        Tests the generation of trading signals from Kalman filter module.
        """

        price_data_subset = self.data[['TIP', 'IEF']]
        linear_reg = LinearRegression(fit_intercept=True)  # Fit regression on the whole dataset
        linear_reg.fit(price_data_subset['TIP'].values.reshape(-1, 1), price_data_subset['IEF'])

        for _, row in price_data_subset.iterrows():
            self.kalman_filter.update(row['TIP'], row['IEF'])

        signals = self.kalman_filter.trading_signals(entry_std_score=1, exit_std_score=1)

        self.assertAlmostEqual(signals['errors'].mean(), np.mean(self.kalman_filter.spread_series), delta=1e-5)
        self.assertAlmostEqual(signals['target_quantity'].sum(), -5, delta=0.1)
        self.assertAlmostEqual(abs(signals['target_quantity']).sum(), 27, delta=0.1)
