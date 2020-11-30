# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Tests AUTO ARIMA prediction functions.
"""

import warnings
import unittest
import os
import numpy as np
import pandas as pd

from arbitragelab.auto_arima import get_trend_order, AutoARIMAForecast


class TestAutoARIMA(unittest.TestCase):
    """
    Tests Auto ARIMA predictions.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """

        np.random.seed(0)
        project_path = os.path.dirname(__file__)
        path = project_path + '/test_data/stock_prices.csv'

        stock_prices = pd.read_csv(path, index_col=0, parse_dates=[0])
        self.non_stationary_series = stock_prices['XLF'].iloc[200:300]  # Non-stationary part
        returns = np.random.normal(0, 1, size=self.non_stationary_series.shape[0])
        self.stationary_series = pd.Series(index=self.non_stationary_series.index, data=returns)

    def test_trend_order(self):
        """
        Tests get_trend_order function.
        """

        stationary_trend_order = get_trend_order(self.stationary_series)
        non_stationary_trend_order = get_trend_order(self.non_stationary_series)
        self.assertEqual(stationary_trend_order, 0)
        self.assertEqual(non_stationary_trend_order, 1)

    def test_auto_arima(self):
        """
        Test Auto ARIMA prediction function.
        """

        y_train = self.non_stationary_series.iloc[:70]
        y_test = self.non_stationary_series.iloc[70:]

        auto_arima_model = AutoARIMAForecast(start_p=3, start_q=3, max_p=10, max_q=10)

        with warnings.catch_warnings():  # Silencing specific Statsmodels ConvergenceWarning
            warnings.filterwarnings('ignore', r'Maximum Likelihood optimization failed to converge.')

            auto_arima_model.get_best_arima_model(y_train, verbose=False)

        recursive_arima_prediction = auto_arima_model.predict(y=y_test, retrain_freq=1, train_window=None)
        non_recursive_arima_prediction = auto_arima_model.predict(y=y_test, retrain_freq=1, train_window=30)

        self.assertAlmostEqual(recursive_arima_prediction.mean(), 6.72, delta=1e-2)
        self.assertAlmostEqual(recursive_arima_prediction.iloc[10], 8.04, delta=1e-2)
        self.assertAlmostEqual((recursive_arima_prediction - non_recursive_arima_prediction).mean(), 0.08, delta=1e-2)
        self.assertAlmostEqual(recursive_arima_prediction.iloc[1], 7.42, delta=1e-2)
        self.assertAlmostEqual(non_recursive_arima_prediction.iloc[1], 5.08, delta=1e-2)
