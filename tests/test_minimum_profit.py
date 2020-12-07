# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

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
        :return:
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

    def test_init(self):
        """
        Unit tests for constructor.
        """

        optimizer = MinimumProfit(self.data)
        self.assertEqual(optimizer.price_df.shape[1], 2)

        # Test a dataframe with 3 columns
        self.assertRaises(Exception, MinimumProfit, self.faulty_data)

        # Test a dataframe with 1 column
        self.assertRaises(Exception, MinimumProfit, self.faulty_data2)

    def test_fit(self):
        """
        Unit tests for cointegration coefficient calculation.
        """

        optimizer = MinimumProfit(self.data)

        _, _ = optimizer.train_test_split(date_cutoff=pd.Timestamp(2002, 1, 1))
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

        optimizer = MinimumProfit(self.no_coint_data)

        _, _ = optimizer.train_test_split(date_cutoff=pd.Timestamp(2020, 1, 1))
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
        Unit test for optimization procedure.

        Use specified parameters here instead of fit from data.
        """

        # Use an empty Dataframe to initialize an instance of optimizer
        empty_df = pd.DataFrame(columns=['Share S1', 'Share S2'])
        optimizer = MinimumProfit(empty_df)

        # Parameters
        ar_coeff = -0.2
        sigma_a = 0.1
        horizon = 1000

        # Results on paper
        upper_bounds = 0.09
        mtps = 13.369806195172025

        # Only do two tests as this process is quite time consuming.
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

        optimizer = MinimumProfit(self.data)

        # Split data into training and test set
        _, _ = optimizer.train_test_split(date_cutoff=pd.Timestamp(2002, 1, 1))

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
        trade_signals, num_of_shares, cond_values = optimizer.trade_signal(optimal_ub, optimal_mtp,
                                                                           beta_eg, epsilon_t_eg)

        # Check if trade signals are correctly stored in the dataframe
        self.assertTrue("otc_U" in trade_signals.columns)
        self.assertTrue("otc_L" in trade_signals.columns)
        self.assertTrue("ctc_U" in trade_signals.columns)
        self.assertTrue("ctc_L" in trade_signals.columns)

        # Check if the number of shares is calculated correctly
        self.assertEqual(num_of_shares[0], 9)
        self.assertEqual(num_of_shares[1], 16)

        # Check if the condition values are calculated correctly
        self.assertAlmostEqual(cond_values[0], 3.97987712)
        self.assertAlmostEqual(cond_values[1], 4.34987712)
        self.assertAlmostEqual(cond_values[2], 4.71987712)

        # Finally check in-sample/out-of-sample switch
        trade_signals_is, _, _ = optimizer.trade_signal(optimal_ub, optimal_mtp,
                                                        beta_eg, epsilon_t_eg, insample=True)
        self.assertEqual(len(trade_signals), 168)
        self.assertEqual(len(trade_signals_is), 253)

    def test_split_dataset(self):
        """
        Unit tests for cointegration coefficient calculation.
        """

        optimizer = MinimumProfit(self.data)

        # Cutoff by date
        train_date, test_date = optimizer.train_test_split(date_cutoff=pd.Timestamp(2002, 1, 1))

        # Cutoff by number
        with self.assertWarns(Warning):
            # Expected warning here that date cutoff input is not used
            train_number, test_number = optimizer.train_test_split(date_cutoff=None, num_cutoff=253)

        # No cutoff, should result in same dataset being returned twice
        train_same, test_same = optimizer.train_test_split(date_cutoff=None)

        # Test output dataframe shapes
        self.assertTupleEqual(train_date.shape, (253, 2))
        self.assertTupleEqual(test_date.shape, (168, 2))

        # Test outputs are the same
        pd.testing.assert_frame_equal(train_date, train_number)
        pd.testing.assert_frame_equal(test_date, test_number)

        # Test no cutoff returns same dataframe
        pd.testing.assert_frame_equal(test_same, self.data)
        pd.testing.assert_frame_equal(train_same, self.data)

    def test_split_dataset_errors(self):
        """
        Unit tests for cointegration coefficient calculation.
        """

        # Test for warning when the Index is not of type pd.DatetimeIndex
        bad_data = self.data.copy()
        bad_data.index = np.zeros(len(bad_data.index))

        optimizer = MinimumProfit(bad_data)

        self.assertRaises(AssertionError, optimizer.train_test_split, pd.Timestamp(2002, 1, 1))

        # Test for warning when the date cutoff point is out of range
        self.assertRaises(AssertionError, optimizer.train_test_split, pd.Timestamp(2021, 1, 1))
