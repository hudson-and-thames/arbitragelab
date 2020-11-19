# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

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
from arbitragelab.util.split_dataset import train_test_split


class TestMinimumProfit(unittest.TestCase):
    """
    Test Minimum Profit Condition Optimization module.
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

        train_df, _ = train_test_split(optimizer.price_df, date_cutoff=pd.Timestamp(2002, 1, 1))
        beta_eg, epsilon_t_eg, ar_coeff_eg, ar_resid_eg = optimizer.fit(train_df, use_johansen=False)
        beta_jo, epsilon_t_jo, ar_coeff_jo, ar_resid_jo = optimizer.fit(train_df, use_johansen=True)

        # Check the AR(1) coefficient and cointegration coefficient
        self.assertAlmostEqual(beta_eg, -1.8378837809650117)
        self.assertAlmostEqual(beta_jo, -1.8647763422880634)
        self.assertAlmostEqual(ar_coeff_eg, 0.8933542389605265)
        self.assertAlmostEqual(ar_coeff_jo, 0.892487910270181)

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

        train_df, _ = train_test_split(optimizer.price_df, date_cutoff=pd.Timestamp(2020, 1, 1))
        with self.assertWarnsRegex(Warning, 'ADF'):
            beta_eg, _, _, _ = optimizer.fit(train_df, use_johansen=False)
        with self.assertWarnsRegex(Warning, 'eigen'):
            beta_jo, _, _, _ = optimizer.fit(train_df, use_johansen=True)
        with self.assertWarnsRegex(Warning, 'trace'):
            beta_jo, _, _, _ = optimizer.fit(train_df, use_johansen=True)

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
        sigma_a = 0.5
        horizon = 1000

        # Results on paper
        upper_bounds = 0.47
        mtps = 66.92441657550803

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
        train_df, trade_df = train_test_split(optimizer.price_df,
                                              date_cutoff=pd.Timestamp(2002, 1, 1))

        # Fit the data
        beta_eg, epsilon_t_eg, ar_coeff_eg, ar_resid_eg = optimizer.fit(train_df,
                                                                        use_johansen=False)

        # Optimize the upper bound
        trade_days = len(trade_df)
        optimal_ub, _, _, optimal_mtp, optimal_num_of_trades = optimizer.optimize(ar_coeff_eg,
                                                                                  epsilon_t_eg,
                                                                                  ar_resid_eg,
                                                                                  trade_days)

        # Generate trade_signal
        trade_signals, num_of_shares = optimizer.trade_signal(trade_df, optimal_ub, optimal_mtp,
                                                              beta_eg, epsilon_t_eg)

        # Check if trade signals are correctly stored in the dataframe
        self.assertTrue("otc_U" in trade_signals.columns)
        self.assertTrue("otc_L" in trade_signals.columns)
        self.assertTrue("ctc_U" in trade_signals.columns)
        self.assertTrue("ctc_L" in trade_signals.columns)

        # Check if the number of shares is calculated correctly
        self.assertEqual(num_of_shares[0], 8)
        self.assertEqual(num_of_shares[1], 13)













