# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests function of Multivariate Cointegration module:
cointegration_approach/multi_coint.py
"""

import os
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from arbitragelab.cointegration_approach.multi_coint import MultivariateCointegration


class TestMultivariateCointegration(unittest.TestCase):
    """
    Test Multivariate Cointegration module.
    """

    def setUp(self):
        """
        Set up the data and parameters.

        Data: AEX, DAX, CAC40, and FTSE100 data from Jan 1st 1996 to Dec 31st 2006.
        """

        # Read data.
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/multi_coint.csv'
        self.data = pd.read_csv(data_path, parse_dates=['Date'])
        self.data.set_index("Date", inplace=True)

        # Split the data into in-sample and out-of-sample subset.
        trade_date = pd.Timestamp(2001, 11, 6)

        # Train data will end on 2001, Nov 5th.
        self.train_data = self.data.loc[:trade_date].iloc[:-1]

        # Test data will start on 2001, Nov 6th.
        self.trade_data = self.data.loc[trade_date:]

    def test_missing_impute_ffill(self):
        """
        Test the calculation of log prices with forward-fill missing data imputation.
        """

        # Test missing data forward fill imputation.
        ffill_test = MultivariateCointegration(self.data, None)
        # Test if the price dataframe has been properly read.
        self.assertEqual(len(ffill_test.asset_df), 2836)

        nan_date1 = pd.Timestamp(1997, 10, 3)
        nan_date2 = pd.Timestamp(1999, 4, 30)
        nan_date3 = pd.Timestamp(1999, 5, 31)
        nan_date4 = pd.Timestamp(2000, 6, 12)

        # Do missing data imputation.
        ffill_result = ffill_test.missing_impute(ffill_test.asset_df, nan_method='ffill')

        # See if the forward filling algorithm worked properly.
        self.assertAlmostEqual(ffill_result.loc[nan_date1]['DAX'], 4266.17)
        self.assertAlmostEqual(ffill_result.loc[nan_date2]['AEX'], 573.52)
        self.assertAlmostEqual(ffill_result.loc[nan_date3]['FTSE'], 6226.22)
        self.assertAlmostEqual(ffill_result.loc[nan_date4]['CAC'], 6549.05)

    def test_missing_impute_spline(self):
        """
        Test the calculation of log prices with cubic spline missing data imputation.
        """

        # Test missing data cubic spline imputation.
        spline_test = MultivariateCointegration(self.data, None)

        # Do missing data imputation.
        spline_result = spline_test.missing_impute(spline_test.asset_df, nan_method='spline', order=3)

        nan_date1 = pd.Timestamp(1997, 10, 3)
        nan_date2 = pd.Timestamp(1999, 4, 30)
        nan_date3 = pd.Timestamp(1999, 5, 31)
        nan_date4 = pd.Timestamp(2000, 6, 12)

        # See if the forward filling algorithm worked properly.
        self.assertAlmostEqual(spline_result.loc[nan_date1]['DAX'], 4266.639227962006)
        self.assertAlmostEqual(spline_result.loc[nan_date2]['AEX'], 572.7939563445334)
        self.assertAlmostEqual(spline_result.loc[nan_date3]['FTSE'], 6231.123973566768)
        self.assertAlmostEqual(spline_result.loc[nan_date4]['CAC'], 6483.287118123764)

    def test_missing_impute_error(self):
        """
        Test the calculation of log prices with erroneous input.
        """

        # Test missing data but with wrong parameters.
        error_test = MultivariateCointegration(self.data, self.trade_data)

        # Raise ValueError.
        self.assertRaises(ValueError, error_test.missing_impute, error_test.asset_df, 'ignore')

    def test_calc_log_price_result(self):
        """
        Test the results of log price calculation.
        """

        # Use ffill to do missing value imputation as it is faster.
        log_price_test = MultivariateCointegration(self.data, None)

        # Do missing data imputation and log price calculation.
        log_price = log_price_test.calc_log_price(log_price_test.asset_df, nan_method='ffill')

        result_test_sample = log_price.tail(1)

        self.assertAlmostEqual(result_test_sample['AEX'].values[0], 6.205244395469226)
        self.assertAlmostEqual(result_test_sample['DAX'].values[0], 8.794358152425072)
        self.assertAlmostEqual(result_test_sample['FTSE'].values[0], 8.73565540233506)
        self.assertAlmostEqual(result_test_sample['CAC'].values[0], 8.620067418819382)

    def test_calc_price_diff_result(self):
        """
        Test the results of price difference calculation.
        """

        # Use ffill to do missing value imputation as it is faster.
        price_diff_test = MultivariateCointegration(self.train_data, self.trade_data)

        # Do missing value imputation and price difference calculation.
        # Use out-of-sample dataframe to test the class property.
        price_diff = price_diff_test.calc_price_diff(price_diff_test.trade_df)

        # Test length to see if dropna() worked properly.
        self.assertEqual(price_diff.shape[0], 1326)

        # Test the last value.
        price_diff_tail = price_diff.tail(1)

        self.assertAlmostEqual(price_diff_tail['AEX'].values[0], -1.84)
        self.assertAlmostEqual(price_diff_tail['DAX'].values[0], -14.89)
        self.assertAlmostEqual(price_diff_tail['FTSE'].values[0], -20.14)
        self.assertAlmostEqual(price_diff_tail['CAC'].values[0], 8.4)

    def test_fit_sig_level_error(self):
        """
        Test the exception generated by inputting a wrong significance level.
        """

        # Initialize the trading signal generator.
        fit_test = MultivariateCointegration(self.data, None)

        # Provide a wrong parameter.
        self.assertRaises(ValueError, fit_test.fit, fit_test.asset_df, sig_level='91%')

    def test_fit_no_rolling_window(self):
        """
        Test the cointegration vector fitting procedure with all available data.
        """

        # Initialize the trading signal generator.
        roll_window = MultivariateCointegration(self.data, None)

        # Calculating log price with ffill imputation.
        log_price = roll_window.calc_log_price(roll_window.asset_df, nan_method='ffill')

        # Use all data, no rolling window.
        no_rw_coint_vec = roll_window.fit(log_price, rolling_window_size=None)

        self.assertAlmostEqual(no_rw_coint_vec['AEX'], 3.905186528497843)
        self.assertAlmostEqual(no_rw_coint_vec['DAX'], 13.59504820164842)
        self.assertAlmostEqual(no_rw_coint_vec['FTSE'], -21.908546878682)
        self.assertAlmostEqual(no_rw_coint_vec['CAC'], -4.642046206684594)

    def test_fit_rolling_window(self):
        """
        Test the cointegration vector fitting procedure with rolling window of 1,500 days.
        """

        # Initialize the trading signal generator.
        roll_window = MultivariateCointegration(self.data, None)

        # Calculating log price with ffill imputation.
        log_price = roll_window.calc_log_price(roll_window.asset_df, nan_method='ffill')

        # Use all data, rolling window with 1500 days.
        rw_coint_vec = roll_window.fit(log_price, rolling_window_size=1500)

        self.assertAlmostEqual(rw_coint_vec['AEX'], 4.096651742116751)
        self.assertAlmostEqual(rw_coint_vec['DAX'], -19.41801830323376)
        self.assertAlmostEqual(rw_coint_vec['FTSE'], 45.98640603614171)
        self.assertAlmostEqual(rw_coint_vec['CAC'], -13.476049682084337)

    def test_fit_not_coint(self):
        """
        Test the cointegration vector fitting procedure with price series that are not cointegrated.
        """
        np.random.seed(0)
        # Simulate an asset price with AR(1) dynamics.
        asset1 = sm.tsa.ArmaProcess(ar=np.array([1, -0.95])).generate_sample(500) + 1
        asset1 = 100 + np.cumsum(asset1)

        # Simulate another asset price with different AR(1) dynamics.
        asset2 = sm.tsa.ArmaProcess(ar=np.array([1, -0.75])).generate_sample(500) + 0.3
        asset2 = 100 + np.cumsum(asset2)

        # Simulate an asset price with AR(2) dynamics.
        asset3 = sm.tsa.ArmaProcess(ar=np.array([1, -0.55, -0.15])).generate_sample(500) + 0.05
        asset3 = 100 + np.cumsum(asset3)

        # Fit and catch the warning.
        no_coint_data = pd.DataFrame(np.vstack((asset1, asset2, asset3)).T)
        no_coint = MultivariateCointegration(no_coint_data, None)
        with self.assertWarnsRegex(Warning, 'trace'):
            no_coint.fit(no_coint.asset_df, sig_level="99%", rolling_window_size=None)
        with self.assertWarnsRegex(Warning, 'eigen'):
            no_coint.fit(no_coint.asset_df, sig_level="99%", rolling_window_size=None)

    def test_num_of_shares(self):
        """
        Test the calculation of number of shares to trade.
        """

        # Find the cointegration vector, calculate the trading signal, i.e. the number of shares.
        num_of_shares_test = MultivariateCointegration(self.train_data, self.trade_data)

        # Calculate log price.
        log_price = num_of_shares_test.calc_log_price(num_of_shares_test.asset_df, nan_method='ffill')

        # Find the cointegration vector.
        num_of_shares_test.fit(log_price, rolling_window_size=None)

        # Default trading position is 10,000,000 currency.
        pos_shares, neg_shares = num_of_shares_test.num_of_shares(log_price, num_of_shares_test.trade_df.iloc[0],
                                                                  nlags=30)

        # Check if the share numbers are correct.
        self.assertEqual(pos_shares.values[0], 20740)
        self.assertEqual(neg_shares.values[0], -613)
        self.assertEqual(neg_shares.values[1], -1242)
        self.assertEqual(neg_shares.values[2], -145)

        # Verify if the positions are dollar-neutral.
        last_price = num_of_shares_test.trade_df.iloc[0]
        neg_pos = pd.concat([last_price, neg_shares], axis=1).dropna()
        neg_pos_dollar_value = neg_pos.iloc[:, 0] @ neg_pos.iloc[:, 1]

        pos_pos = pd.concat([last_price, pos_shares], axis=1).dropna()
        pos_pos_dollar_value = pos_pos.iloc[:, 0] @ pos_pos.iloc[:, 1]

        # Rounding error will always cause positions not exactly dollar neutral.
        # As long as it is close to neutral it should be passing the test.
        self.assertTrue(abs(neg_pos_dollar_value + pos_pos_dollar_value) / 1.e7 < 1.e-3)

    def test_trade_signal(self):
        """
        Test trading signal generation.
        """

        # Initialize two trading signal generator. Short one do not trigger a cointegration vector update.
        trade_signal_test = MultivariateCointegration(self.train_data, self.trade_data)
        trade_signal_short_test = MultivariateCointegration(self.train_data, self.trade_data.iloc[:21, :])

        # Generate trading signals
        signals, coint_vec_time_evo, returns = trade_signal_test.trading_signal(30, rolling_window_size=None)
        _, coint_vec_time_evo_short, _ = trade_signal_short_test.trading_signal(15, rolling_window_size=None)

        # Check the shape of signal and cointegration vector evolution dataframe
        self.assertTupleEqual(signals.shape, (1326, 4))
        self.assertTupleEqual(coint_vec_time_evo.shape, (1326, 4))

        # Check the value of the calculation results. Check head and tail.
        self.assertIsNone(np.testing.assert_allclose(signals.iloc[0].values,
                                                     np.array([20740., -613., -1242., -145.])))
        self.assertIsNone(np.testing.assert_allclose(signals.iloc[-1].values,
                                                     np.array([-4468., -1176., 1321., 316.])))
        target_head = np.array([30.582711178446427, -12.11624038306599, -27.220035836898937, -2.716097980580956])
        self.assertIsNone(np.testing.assert_allclose(coint_vec_time_evo.iloc[0].values, target_head, rtol=1e-5))

        target_tail = np.array([3.884615498623572, 13.602601047839087, -21.879100926485236, -4.644199584857195])
        self.assertIsNone(np.testing.assert_allclose(coint_vec_time_evo.iloc[-1].values, target_tail, rtol=1e-5))

        # If cointegration vector is not updated, then the cointegration vector should be the same for each data point.
        self.assertTrue(np.isclose(coint_vec_time_evo_short['AEX'], coint_vec_time_evo_short['AEX'].mean()).all())
        self.assertTrue(np.isclose(coint_vec_time_evo_short['FTSE'], coint_vec_time_evo_short['FTSE'].mean()).all())
        self.assertTrue(np.isclose(coint_vec_time_evo_short['CAC'], coint_vec_time_evo_short['CAC'].mean()).all())
        self.assertTrue(np.isclose(coint_vec_time_evo_short['DAX'], coint_vec_time_evo_short['DAX'].mean()).all())

        # Check the final cumulative returns.
        cum_return_df = (1 + returns).cumprod() - 1
        self.assertAlmostEqual(cum_return_df.iloc[-1].values[0], 0.20506352278552242)
