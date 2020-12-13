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
from matplotlib.dates import num2date
from matplotlib.figure import Figure

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

        # Read data
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/multi_coint.csv'
        self.data = pd.read_csv(data_path, parse_dates=['Date'])
        self.data.set_index("Date", inplace=True)

        # Split the data into in-sample and out-of-sample subset
        trade_date = pd.Timestamp(2001, 11, 6)

        # Train data will end on 2001, Nov 5th
        self.train_data = self.data.loc[:trade_date].iloc[:-1]

        # Test data will start on 2001, Nov 6th
        self.trade_data = self.data.loc[trade_date:]

    def test_missing_impute_ffill(self):
        """
        Test the calculation of log prices with forward-fill missing data imputation.
        """

        # Test missing data forward fill imputation
        ffill_test = MultivariateCointegration(self.train_data, self.trade_data)

        # Test if the price dataframe has been properly read
        self.assertEqual(len(ffill_test.asset_df), 1509)
        self.assertEqual(len(ffill_test.trade_df), 1327)

        nan_date1 = pd.Timestamp(1997, 10, 3)
        nan_date2 = pd.Timestamp(1999, 4, 30)
        nan_date3 = pd.Timestamp(1999, 5, 31)
        nan_date4 = pd.Timestamp(2000, 6, 12)

        # Do missing data imputation
        ffill_test.fillna_inplace(nan_method='ffill')

        # See if the forward filling algorithm worked properly
        self.assertAlmostEqual(ffill_test.asset_df.loc[nan_date1]['DAX'], 4266.17)
        self.assertAlmostEqual(ffill_test.asset_df.loc[nan_date2]['AEX'], 573.52)
        self.assertAlmostEqual(ffill_test.asset_df.loc[nan_date3]['FTSE'], 6226.22)
        self.assertAlmostEqual(ffill_test.asset_df.loc[nan_date4]['CAC'], 6549.05)

    def test_missing_impute_spline(self):
        """
        Test the calculation of log prices with cubic spline missing data imputation.
        """

        # Test missing data cubic spline imputation
        spline_test = MultivariateCointegration(self.train_data, self.trade_data)

        # Do missing data imputation
        spline_test.fillna_inplace(nan_method='spline', order=3)

        nan_date1 = pd.Timestamp(1997, 10, 3)
        nan_date2 = pd.Timestamp(1999, 4, 30)
        nan_date3 = pd.Timestamp(1999, 5, 31)
        nan_date4 = pd.Timestamp(2000, 6, 12)

        # See if the forward filling algorithm worked properly
        self.assertAlmostEqual(spline_test.asset_df.loc[nan_date1]['DAX'], 4266.639227962006)
        self.assertAlmostEqual(spline_test.asset_df.loc[nan_date2]['AEX'], 572.7939563445334)
        self.assertAlmostEqual(spline_test.asset_df.loc[nan_date3]['FTSE'], 6231.123973566768)
        self.assertAlmostEqual(spline_test.asset_df.loc[nan_date4]['CAC'], 6483.287118123764)

    def test_missing_impute_error(self):
        """
        Test the calculation of log prices with erroneous input.
        """

        # Test missing data but with wrong parameters
        error_test = MultivariateCointegration(self.train_data, self.trade_data)

        # Raise ValueError
        self.assertRaises(ValueError, error_test.fillna_inplace, nan_method='ignore')

    def test_calc_log_price_result(self):
        """
        Test the results of log price calculation.
        """

        # Use ffill to do missing value imputation as it is faster
        log_price_test = MultivariateCointegration(self.train_data, self.trade_data)

        # Do missing data imputation and log price calculation
        log_price_test.fillna_inplace(nan_method='ffill')
        log_price = log_price_test.calc_log_price(log_price_test.asset_df)

        result_test_sample = log_price.tail(1)

        self.assertAlmostEqual(result_test_sample['AEX'].values[0], 6.179954539922313)
        self.assertAlmostEqual(result_test_sample['DAX'].values[0], 8.466975108255552)
        self.assertAlmostEqual(result_test_sample['FTSE'].values[0], 8.5581662145311)
        self.assertAlmostEqual(result_test_sample['CAC'].values[0], 8.408569579869317)

    def test_calc_price_diff_result(self):
        """
        Test the results of price difference calculation.
        """

        # Use ffill to do missing value imputation as it is faster
        price_diff_test = MultivariateCointegration(self.train_data, self.trade_data)

        # Do missing value imputation and price difference calculation
        price_diff_test.fillna_inplace(nan_method='ffill')

        # Use out-of-sample dataframe to test the class property
        price_diff = price_diff_test.calc_price_diff(price_diff_test.trade_df)

        # Test length to see if dropna() worked properly
        self.assertEqual(price_diff.shape[0], 1326)

        # Test the last value
        price_diff_tail = price_diff.tail(1)

        self.assertAlmostEqual(price_diff_tail['AEX'].values[0], -1.84)
        self.assertAlmostEqual(price_diff_tail['DAX'].values[0], -14.89)
        self.assertAlmostEqual(price_diff_tail['FTSE'].values[0], -20.14)
        self.assertAlmostEqual(price_diff_tail['CAC'].values[0], 8.4)

    def test_fit_sig_level_error(self):
        """
        Test the exception generated by inputting a wrong significance level.
        """

        # Initialize the trading signal generator
        fit_test = MultivariateCointegration(self.train_data, self.trade_data)

        # Provide a wrong parameter
        self.assertRaises(ValueError, fit_test.fit, fit_test.asset_df, sig_level='91%')

    def test_fit_no_rolling_window(self):
        """
        Test the cointegration vector fitting procedure with all available data.
        """

        # Initialize the trading signal generator
        roll_window = MultivariateCointegration(self.train_data, self.trade_data)

        # Calculating log price with ffill imputation
        roll_window.fillna_inplace(nan_method='ffill')
        log_price = roll_window.calc_log_price(roll_window.asset_df)

        # Use all data, no rolling window
        no_rw_coint_vec = roll_window.fit(log_price, rolling_window_size=None)

        self.assertAlmostEqual(no_rw_coint_vec['AEX'], 30.582711178446427)
        self.assertAlmostEqual(no_rw_coint_vec['DAX'], -12.11624038306599)
        self.assertAlmostEqual(no_rw_coint_vec['FTSE'], -27.220035836898937)
        self.assertAlmostEqual(no_rw_coint_vec['CAC'], -2.716097980580956)

    def test_fit_rolling_window(self):
        """
        Test the cointegration vector fitting procedure with rolling window of 1,500 days.
        """

        # Initialize the trading signal generator
        roll_window = MultivariateCointegration(self.train_data, self.trade_data)

        # Calculating log price with ffill imputation
        roll_window.fillna_inplace(nan_method='ffill')
        log_price = roll_window.calc_log_price(roll_window.asset_df)

        # Use all data, rolling window with 1500 days
        rw_coint_vec = roll_window.fit(log_price, rolling_window_size=1000)

        self.assertAlmostEqual(rw_coint_vec['AEX'], 47.7612404443356)
        self.assertAlmostEqual(rw_coint_vec['DAX'], -12.279690370837093)
        self.assertAlmostEqual(rw_coint_vec['FTSE'], -32.68813027305016)
        self.assertAlmostEqual(rw_coint_vec['CAC'], -11.082349345405024)

    def test_fit_not_coint(self):
        """
        Test the cointegration vector fitting procedure with price series that are not cointegrated.
        """
        np.random.seed(0)
        # Simulate an asset price with AR(1) dynamics
        asset1 = sm.tsa.ArmaProcess(ar=np.array([1, -0.95])).generate_sample(500) + 1
        asset1 = 100 + np.cumsum(asset1)

        # Simulate another asset price with different AR(1) dynamics
        asset2 = sm.tsa.ArmaProcess(ar=np.array([1, -0.75])).generate_sample(500) + 0.3
        asset2 = 100 + np.cumsum(asset2)

        # Simulate an asset price with AR(2) dynamics
        asset3 = sm.tsa.ArmaProcess(ar=np.array([1, -0.55, -0.15])).generate_sample(500) + 0.05
        asset3 = 100 + np.cumsum(asset3)

        # Fit and catch the warning
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

        # Find the cointegration vector, calculate the trading signal, i.e. the number of shares
        num_of_shares_test = MultivariateCointegration(self.train_data, self.trade_data)

        # Calculate log price
        num_of_shares_test.fillna_inplace(nan_method='ffill')
        log_price = num_of_shares_test.calc_log_price(num_of_shares_test.asset_df)

        # Find the cointegration vector
        num_of_shares_test.fit(log_price, rolling_window_size=None)

        # Default trading position is 10,000,000 currency
        pos_shares, neg_shares, pos_ntn, neg_ntn = num_of_shares_test.num_of_shares(log_price,
                                                                                    num_of_shares_test.trade_df.iloc[0],
                                                                                    nlags=30)

        # Check if the share numbers are correct
        self.assertEqual(pos_shares.values[0], 20740)
        self.assertEqual(neg_shares.values[0], -613)
        self.assertEqual(neg_shares.values[1], -1242)
        self.assertEqual(neg_shares.values[2], -145)

        # Check if the portfolio notional are correct
        self.assertAlmostEqual(pos_ntn.values[0], 10000205.8)
        self.assertAlmostEqual(neg_ntn.values[0], -2885789.45)
        self.assertAlmostEqual(neg_ntn.values[1], -6475862.52)
        self.assertAlmostEqual(neg_ntn.values[2], -646945.05)

        # Verify if the positions are dollar-neutral
        last_price = num_of_shares_test.trade_df.iloc[0]
        neg_pos = pd.concat([last_price, neg_shares], axis=1).dropna()
        neg_pos_dollar_value = neg_pos.iloc[:, 0] @ neg_pos.iloc[:, 1]

        pos_pos = pd.concat([last_price, pos_shares], axis=1).dropna()
        pos_pos_dollar_value = pos_pos.iloc[:, 0] @ pos_pos.iloc[:, 1]

        # Rounding error will always cause positions not exactly dollar neutral
        # As long as it is close to neutral it should be passing the test
        self.assertTrue(abs(neg_pos_dollar_value + pos_pos_dollar_value) / 1.e7 < 1.e-3)

    # pylint: disable=too-many-locals, too-many-statements
    def test_trade_signal(self):
        """
        Test trading signal generation.

        Trade summary and plot functions will be tested here as well in order to cut off build time.
        """

        # Initialize two trading signal generator. Short one do not trigger a cointegration vector update
        trade_signal_test = MultivariateCointegration(self.train_data, self.trade_data)
        trade_signal_short_test = MultivariateCointegration(self.train_data, self.trade_data.iloc[:21, :])

        # Impute NaN values
        trade_signal_test.fillna_inplace(nan_method='ffill')
        trade_signal_short_test.fillna_inplace(nan_method='ffill')

        # Generate trading signals
        signals, signals_ntn, coint_vec_time_evo, returns = trade_signal_test.trading_signal(30,
                                                                                             rolling_window_size=None)
        _, _, coint_vec_time_evo_short, _ = trade_signal_short_test.trading_signal(15, rolling_window_size=None)

        # Check the shape of signal and cointegration vector evolution dataframe
        self.assertTupleEqual(signals.shape, (1326, 4))
        self.assertTupleEqual(coint_vec_time_evo.shape, (1326, 4))

        # Check the value of the calculation results. Check head and tail
        self.assertIsNone(np.testing.assert_allclose(signals.iloc[0].values,
                                                     np.array([20740., -613., -1242., -145.])))
        self.assertIsNone(np.testing.assert_allclose(signals.iloc[-1].values,
                                                     np.array([-4468., -1176., 1321., 316.])))
        self.assertIsNone(np.testing.assert_allclose(signals_ntn.iloc[0].values,
                                                     np.array([10000205.8, -2885789.45, -6475862.52, -646945.05])))
        self.assertIsNone(np.testing.assert_allclose(signals_ntn.iloc[-1].values,
                                                     np.array([-2221400.24, -7775488.56,  8244294.95, 1748541.76])))
        target_head = np.array([30.582711178446427, -12.11624038306599, -27.220035836898937, -2.716097980580956])
        self.assertIsNone(np.testing.assert_allclose(coint_vec_time_evo.iloc[0].values, target_head, rtol=1e-5))

        target_tail = np.array([3.884615498623572, 13.602601047839087, -21.879100926485236, -4.644199584857195])
        self.assertIsNone(np.testing.assert_allclose(coint_vec_time_evo.iloc[-1].values, target_tail, rtol=1e-5))

        # If cointegration vector is not updated, then the cointegration vector should be the same for each data point
        self.assertTrue(np.isclose(coint_vec_time_evo_short['AEX'], coint_vec_time_evo_short['AEX'].mean()).all())
        self.assertTrue(np.isclose(coint_vec_time_evo_short['FTSE'], coint_vec_time_evo_short['FTSE'].mean()).all())
        self.assertTrue(np.isclose(coint_vec_time_evo_short['CAC'], coint_vec_time_evo_short['CAC'].mean()).all())
        self.assertTrue(np.isclose(coint_vec_time_evo_short['DAX'], coint_vec_time_evo_short['DAX'].mean()).all())

        # Check the final cumulative returns
        cum_return_df = (1 + returns).cumprod() - 1
        self.assertAlmostEqual(cum_return_df.iloc[-1].values[0], 0.30152507496544234)

        # Check summary function
        summary = trade_signal_test.summary(returns)
        self.assertListEqual(list(summary.index), ["Cumulative Return",
                                                   "Returns Mean",
                                                   "Returns Standard Deviation",
                                                   "Returns Skewness",
                                                   "Returns Kurtosis",
                                                   "Max Return",
                                                   "Min Return",
                                                   "Sharpe ratio",
                                                   "Sortino ratio",
                                                   "Percentage of Up Days",
                                                   "Percentage of Down Days"])

        # Check plot functions
        # Plot the raw number of shares figure
        fig_full = trade_signal_test.plot_all(signals, signals_ntn, coint_vec_time_evo, returns,
                                              use_weights=False)
        # Plot the portfolio weights figure
        fig_full_weight = trade_signal_test.plot_all(signals, signals_ntn, coint_vec_time_evo, returns,
                                                     use_weights=True, start_date=pd.Timestamp(2001, 11, 6),
                                                     end_date=pd.Timestamp(2007, 1, 2))
        # Plot the returns only figure
        fig_returns = trade_signal_test.plot_returns(returns)

        # First check if the plot object has been generated
        self.assertTrue(issubclass(type(fig_full), Figure))
        self.assertTrue(issubclass(type(fig_full_weight), Figure))
        self.assertTrue(issubclass(type(fig_returns), Figure))

        # Check subplots numbers
        self.assertEqual(len(fig_full.get_axes()), 3)
        self.assertEqual(len(fig_full_weight.get_axes()), 3)
        self.assertEqual(len(fig_returns.get_axes()), 1)

        # Check the xlim of the plot when nothing was specified
        ax1, _, ax3 = fig_full.get_axes()
        plot_no_spec_xlim_left = num2date(ax1.get_xlim()[0])
        plot1_day = plot_no_spec_xlim_left.day
        plot1_month = plot_no_spec_xlim_left.month
        plot1_year = plot_no_spec_xlim_left.year

        self.assertEqual(plot1_year, 2001)
        self.assertEqual(plot1_month, 8)
        self.assertEqual(plot1_day, 4)

        # Check y-label when raw signals are plotted
        signal_label = ax3.get_ylabel()
        self.assertEqual(signal_label, "Num. of Shares")

        # Check the xlim of the plot when nothing was specified
        ax1, _, ax3 = fig_full_weight.get_axes()
        plot_no_spec_xlim_left = num2date(ax1.get_xlim()[0])
        plot1_day = plot_no_spec_xlim_left.day
        plot1_month = plot_no_spec_xlim_left.month
        plot1_year = plot_no_spec_xlim_left.year

        self.assertEqual(plot1_year, 2001)
        self.assertEqual(plot1_month, 11)
        self.assertEqual(plot1_day, 6)

        # Check y-label when raw signals are plotted
        signal_label = ax3.get_ylabel()
        self.assertEqual(signal_label, "Portfolio Weights")
