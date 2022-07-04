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

    def test_missing_impute_ffill(self):
        """
        Test the calculation of log prices with forward-fill missing data imputation.
        """

        # Test missing data forward fill imputation
        ffill_test = MultivariateCointegration()
        ffill_test.set_train_dataset(self.train_data)

        # Test if the price dataframe has been properly read
        self.assertEqual(len(ffill_test.asset_df), 1509)

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
        spline_test = MultivariateCointegration()
        spline_test.set_train_dataset(self.train_data)

        # Do missing data imputation
        spline_test.fillna_inplace(nan_method='spline', order=3)

        nan_date1 = pd.Timestamp(1997, 10, 3)
        nan_date2 = pd.Timestamp(1999, 4, 30)
        nan_date3 = pd.Timestamp(1999, 5, 31)
        nan_date4 = pd.Timestamp(2000, 6, 12)

        # See if the forward filling algorithm worked properly
        self.assertAlmostEqual(spline_test.asset_df.loc[nan_date1]['DAX'], 4266.719586327144)
        self.assertAlmostEqual(spline_test.asset_df.loc[nan_date2]['AEX'], 570.6726628680204)
        self.assertAlmostEqual(spline_test.asset_df.loc[nan_date3]['FTSE'], 6231.794624179276)
        self.assertAlmostEqual(spline_test.asset_df.loc[nan_date4]['CAC'], 6483.039776337608)

    def test_missing_impute_error(self):
        """
        Test the calculation of log prices with erroneous input.
        """

        # Test missing data but with wrong parameters
        error_test = MultivariateCointegration()
        error_test.set_train_dataset(self.train_data)

        # Raise ValueError
        self.assertRaises(ValueError, error_test.fillna_inplace, nan_method='ignore')

    def test_calc_log_price_result(self):
        """
        Test the results of log price calculation.
        """

        # Use ffill to do missing value imputation as it is faster
        log_price_test = MultivariateCointegration()
        log_price_test.set_train_dataset(self.train_data)

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
        price_diff_test = MultivariateCointegration()
        price_diff_test.set_train_dataset(self.train_data)

        # Do missing value imputation and price difference calculation
        price_diff_test.fillna_inplace(nan_method='ffill')

        # Use out-of-sample dataframe to test the class property
        price_diff = price_diff_test.calc_price_diff(price_diff_test.asset_df)

        # Test length to see if dropna() worked properly
        self.assertEqual(price_diff.shape[0], 1508)

        # Test the last value
        price_diff_tail = price_diff.tail(1)

        self.assertAlmostEqual(price_diff_tail['AEX'].values[0], 14.8200, 3)
        self.assertAlmostEqual(price_diff_tail['DAX'].values[0], 171.7999, 3)
        self.assertAlmostEqual(price_diff_tail['FTSE'].values[0], 79.5799, 3)
        self.assertAlmostEqual(price_diff_tail['CAC'].values[0], 115.9400, 3)

    def test_fit_sig_level_error(self):
        """
        Test the exception generated by inputting a wrong significance level.
        """

        # Initialize the trading signal generator
        fit_test = MultivariateCointegration()
        fit_test.set_train_dataset(self.train_data)

        # Provide a wrong parameter
        self.assertRaises(ValueError, fit_test.fit, fit_test.asset_df, sig_level='91%')

    def test_fit_no_rolling_window(self):
        """
        Test the cointegration vector fitting procedure with all available data.
        """

        # Initialize the trading signal generator
        roll_window = MultivariateCointegration()
        roll_window.set_train_dataset(self.train_data)

        # Calculating log price with ffill imputation
        roll_window.fillna_inplace(nan_method='ffill')
        log_price = roll_window.calc_log_price(roll_window.asset_df)

        # Use all data, no rolling window
        no_rw_coint_vec = roll_window.fit(log_price)

        self.assertAlmostEqual(no_rw_coint_vec['AEX'], 30.582711178446427)
        self.assertAlmostEqual(no_rw_coint_vec['DAX'], -12.11624038306599)
        self.assertAlmostEqual(no_rw_coint_vec['FTSE'], -27.220035836898937)
        self.assertAlmostEqual(no_rw_coint_vec['CAC'], -2.716097980580956)

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
        no_coint = MultivariateCointegration()
        no_coint.set_train_dataset(no_coint_data)
        with self.assertWarnsRegex(Warning, 'trace'):
            no_coint.fit(no_coint.asset_df, sig_level="99%")
        with self.assertWarnsRegex(Warning, 'eigen'):
            no_coint.fit(no_coint.asset_df, sig_level="99%")

    def test_get_coint_vec(self):
        """
        Test getting cointegration vector.

        Trade summary and plot functions will be tested here as well in order to cut off build time.
        """

        # Initialize two trading signal generator. Short one do not trigger a cointegration vector update
        trade_signal_test = MultivariateCointegration()
        trade_signal_test.set_train_dataset(self.train_data)

        # Impute NaN values
        trade_signal_test.fillna_inplace(nan_method='ffill')

        # Generate cointegration vector
        coint_vec = trade_signal_test.get_coint_vec()

        self.assertAlmostEqual(coint_vec['AEX'], 30.582711178446427)
        self.assertAlmostEqual(coint_vec['DAX'], -12.11624038306599)
        self.assertAlmostEqual(coint_vec['FTSE'], -27.220035836898937)
        self.assertAlmostEqual(coint_vec['CAC'], -2.716097980580956)

    def test_summary(self):
        """
        Test generating summary.
        """

        # Creating returns dataframe
        returns = pd.DataFrame(self.train_data['AEX'].pct_change())

        # Initialize a trading signal generator
        trade_signal_test = MultivariateCointegration()

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

    def test_plot_returns(self):
        """
        Test plotting returns.
        """

        # Creating returns dataframe
        returns = pd.DataFrame(self.train_data['AEX'].pct_change())

        # Initialize a trading signal generator
        trade_signal_test = MultivariateCointegration()

        # Test returns only plot in here
        fig_returns = trade_signal_test.plot_returns(returns)
        fig_returns_with_xlim = trade_signal_test.plot_returns(returns, start_date=pd.Timestamp(2001, 11, 6),
                                                               end_date=pd.Timestamp(2002, 6, 6))

        # Check if the plot object returns
        self.assertTrue(issubclass(type(fig_returns), Figure))
        self.assertTrue(issubclass(type(fig_returns_with_xlim), Figure))

        # Check subplot number
        self.assertEqual(len(fig_returns.get_axes()), 1)
        self.assertEqual(len(fig_returns_with_xlim.get_axes()), 1)

        # Check the xlim of the plot when nothing was specified
        ax1 = fig_returns.get_axes()
        plot_no_spec_xlim_left = num2date(ax1[0].get_xlim()[0])
        plot1_day = plot_no_spec_xlim_left.day
        plot1_month = plot_no_spec_xlim_left.month
        plot1_year = plot_no_spec_xlim_left.year

        self.assertEqual(plot1_year, 1995)
        self.assertEqual(plot1_month, 9)
        self.assertEqual(plot1_day, 18)

        # Check the xlim of the plot when nothing was specified
        ax1 = fig_returns_with_xlim.get_axes()
        plot_no_spec_xlim_left = num2date(ax1[0].get_xlim()[0])
        plot1_day = plot_no_spec_xlim_left.day
        plot1_month = plot_no_spec_xlim_left.month
        plot1_year = plot_no_spec_xlim_left.year

        self.assertEqual(plot1_year, 2001)
        self.assertEqual(plot1_month, 11)
        self.assertEqual(plot1_day, 6)
