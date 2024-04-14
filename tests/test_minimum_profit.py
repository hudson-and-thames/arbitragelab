"""
Tests function of Minimum Profit Condition Optimization module:
cointegration_approach/minimum_profit.py
"""

import os
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from arbitragelab.cointegration_approach.minimum_profit import MinimumProfit


class TestMinimumProfit(unittest.TestCase):
    """
    Test Minimum Profit Condition Optimization module: minimum profit optimization.
    """

    def setUp(self):
        """
        Set up the data and parameters.

        Data: ANZ-ADB daily data (1/1/2001 - 8/30/2002)
        Data: XLF-XLK daily data (1/1/2018 - 11/17/2020)
        """

        np.random.seed(50)
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/ANZ-ADB.csv'
        no_coint_path = project_path + '/test_data/XLF-XLK.csv'

        # Normal data
        self.data = pd.read_csv(data_path, parse_dates=['Date'])
        self.data.set_index("Date", inplace=True)

        # Data with missing columns or extra columns
        self.faulty_data = deepcopy(self.data)
        self.faulty_data = self.faulty_data.assign(AAPL=np.random.standard_normal(len(self.faulty_data)))
        self.faulty_data2 = self.data[['ANZ']]

        # Data that are not cointegrated
        self.no_coint_data = pd.read_csv(no_coint_path, parse_dates=['Date'])
        self.no_coint_data.set_index("Date", inplace=True)

    def test_set_train_dataset(self):
        """
        Unit tests for set_train_dataset.
        """

        optimizer = MinimumProfit()
        optimizer.set_train_dataset(self.data)

        self.assertEqual(optimizer.price_df.shape[1], 2)

        # Test a dataframe with 3 columns
        self.assertRaises(Exception, optimizer.set_train_dataset, self.faulty_data)

        # Test a dataframe with 1 column
        self.assertRaises(Exception, optimizer.set_train_dataset, self.faulty_data2)

    def test_fit(self):
        """
        Unit tests for cointegration coefficient calculation.
        """

        optimizer = MinimumProfit()
        optimizer.set_train_dataset(self.data[self.data.index < pd.Timestamp(2002, 1, 1)])

        beta_eg, epsilon_t_eg, ar_coeff_eg, ar_resid_eg = optimizer.fit(sig_level="90%", use_johansen=False)
        beta_jo, epsilon_t_jo, ar_coeff_jo, ar_resid_jo = optimizer.fit(sig_level="90%", use_johansen=True)

        # Check the AR(1) coefficient and cointegration coefficient
        self.assertAlmostEqual(beta_eg, -1.8378837809650117)
        self.assertAlmostEqual(beta_jo, -1.8647763422880634)
        self.assertAlmostEqual(ar_coeff_eg, 0.8933437089287942)
        self.assertAlmostEqual(ar_coeff_jo, 0.8924885761351791)

        # Check if the cointegration error and residual error follows the following relationship:
        # sigma_epsilon = \sqrt{1 - phi^2} sigma_a
        epsilon_ratio_eg = ar_resid_eg.std() / epsilon_t_eg.std()
        ar_ratio_eg = np.sqrt(1. - ar_coeff_eg ** 2)

        epsilon_ratio_jo = ar_resid_jo.std() / epsilon_t_jo.std()
        ar_ratio_jo = np.sqrt(1. - ar_coeff_jo ** 2)

        error_eg = abs(epsilon_ratio_eg - ar_ratio_eg) / ar_ratio_eg
        error_jo = abs(epsilon_ratio_jo - ar_ratio_jo) / ar_ratio_jo

        self.assertTrue(error_eg < 0.02)
        self.assertTrue(error_jo < 0.02)

    def test_fit_warning(self):
        """
        Unit tests for warnings triggered when the series pair is not cointegrated.
        """

        optimizer = MinimumProfit()
        optimizer.set_train_dataset(self.no_coint_data[self.no_coint_data.index < pd.Timestamp(2020, 1, 1)])

        with self.assertWarnsRegex(Warning, 'ADF'):
            _, _, _, _ = optimizer.fit(sig_level="95%", use_johansen=False)
        with self.assertWarnsRegex(Warning, 'eigen'):
            _, _, _, _ = optimizer.fit(sig_level="90%", use_johansen=True)
        with self.assertWarnsRegex(Warning, 'trace'):
            _, _, _, _ = optimizer.fit(sig_level="99%", use_johansen=True)
        with self.assertRaises(ValueError):
            _, _, _, _ = optimizer.fit(sig_level="91%", use_johansen=True)

    def test_optimize(self):
        """
        Unit test for the optimization procedure.

        Use specified parameters here instead of fit from data.
        """

        # Initialize an instance of the optimizer
        optimizer = MinimumProfit()

        # Parameters
        ar_coeff = -0.2
        sigma_a = 0.1
        horizon = 1000

        # Results on paper
        upper_bounds = 0.09
        mtps = 13.369806195172025

        # Only do two tests as this process is quite time consuming
        ar_resid = np.random.normal(0, sigma_a, 1000)
        sigma_epsilon = sigma_a / np.sqrt(1 - ar_coeff ** 2)
        epsilon_t = pd.Series(np.random.normal(0, sigma_epsilon, 1000))
        optimal_ub, _, _, optimal_mtp, _ = optimizer.optimize(ar_coeff, epsilon_t, ar_resid, horizon)
        self.assertAlmostEqual(optimal_ub, upper_bounds)
        self.assertAlmostEqual(optimal_mtp, mtps)

    def test_trade_signal(self):
        """
        Unit tests for trade signal generation.
        """

        optimizer = MinimumProfit()
        optimizer.set_train_dataset(self.data[self.data.index < pd.Timestamp(2002, 1, 1)])

        # Fit the data
        beta_eg, epsilon_t_eg, _, _ = optimizer.fit(use_johansen=False)

        # Optimize the upper bound. The result has been pre-runned to save build time.
        # optimal_ub, _, _, optimal_mtp, _ = optimizer.optimize(ar_coeff_eg, epsilon_t_eg, ar_resid_eg, len(train_df))
        optimal_ub = 0.37
        optimal_mtp = 3.1276935677285995

        # Exception check
        with self.assertRaises(Exception):
            _, _, _ = optimizer.trade_signal(optimal_ub, optimal_ub - 0.05,
                                             beta_eg, epsilon_t_eg)

        # Generate trade_signal
        num_of_shares, cond_values = optimizer.get_optimal_levels(optimal_ub, optimal_mtp,
                                                                  beta_eg, epsilon_t_eg)

        # Check if the number of shares is calculated correctly
        self.assertEqual(num_of_shares[0], 9)
        self.assertEqual(num_of_shares[1], 16)

        # Check if the condition values are calculated correctly
        self.assertAlmostEqual(cond_values[0], 3.97987712)
        self.assertAlmostEqual(cond_values[1], 4.34987712)
        self.assertAlmostEqual(cond_values[2], 4.71987712)

        # Test exception for optimal levels
        self.assertRaises(Exception, optimizer.get_optimal_levels,
                          optimal_ub + 1, optimal_ub, beta_eg, epsilon_t_eg)

    def test_construct_spread(self):
        """
        Unit tests for construct_spread.
        """

        # Asset price series
        series = self.data
        beta = -1.62

        # Construct spread
        spread = MinimumProfit.construct_spread(series, beta)

        self.assertEqual(spread.size, 421)
        self.assertAlmostEqual(spread[5], 5.416668, 1e-5)
