# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests Tear Sheets tools.
"""
# pylint: disable=protected-access, invalid-name, protected-access, too-many-locals

import unittest
import os

import pandas as pd
import numpy as np

from arbitragelab.tearsheet.tearsheet import TearSheet


class TestTearSheet(unittest.TestCase):
    """
    Test the TearSheet module.

    Skipping the callback testing here, as this causes extra complications.
    """
    def setUp(self):
        """
        Set the file path for the data and testing variables.
        """

        # Fixing seed for dataset generation
        np.random.seed(0)

        # Loading GLD/GDX dataset
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/gld_gdx_data.csv'
        data = pd.read_csv(self.path)
        data = data.set_index('Date')

        self.dataframe = data[['GLD', 'GDX']]

    def test_cointegration_tearsheet(self):
        """
        Tests if the flask server for the Cointegration Tear Sheet works properly.
        """

        # Calling the tear sheet
        test = TearSheet()

        # Calling the tear sheet, not running servers
        server_python = test.cointegration_tearsheet(self.dataframe, app_display='default')
        server_jupyter = test.cointegration_tearsheet(self.dataframe, app_display='jupyter')

        self.assertEqual(str(type(server_python)), "<class 'dash.dash.Dash'>")
        self.assertEqual(str(type(server_jupyter)), "<class 'jupyter_dash.jupyter_app.JupyterDash'>")

    def test_ou_tearsheet(self):
        """
        Tests if the flask server for the OU Model Tear Sheet works properly.
        """

        # Calling the tear sheet
        test = TearSheet()

        # Calling the tear sheet, not running servers
        server_python = test.ou_tearsheet(self.dataframe, app_display='default')
        server_jupyter = test.ou_tearsheet(self.dataframe, app_display='jupyter')

        self.assertEqual(str(type(server_python)), "<class 'dash.dash.Dash'>")
        self.assertEqual(str(type(server_jupyter)), "<class 'jupyter_dash.jupyter_app.JupyterDash'>")

    def test_get_basic_assets_data(self):
        """
        Tests the basic data transformation, normalization, and ADF statistics calculation.
        """

        # Calling the tear sheet and setting the data parameter
        test = TearSheet()
        test.data = self.dataframe

        # Applying basic data transformation
        transformed_data = test._get_basic_assets_data()
        (asset_name_1, norm_asset_price_1, adf_asset_1, test_stat_1,
         asset_name_2, norm_asset_price_2, adf_asset_2, test_stat_2) = transformed_data

        # Asset names
        self.assertEqual(asset_name_1, 'GLD')
        self.assertEqual(asset_name_2, 'GDX')

        # Normalized prices
        self.assertAlmostEqual(min(norm_asset_price_1), 0.0, places=3)
        self.assertAlmostEqual(min(norm_asset_price_2), 0.0, places=3)
        self.assertAlmostEqual(max(norm_asset_price_1), 1.0, places=3)
        self.assertAlmostEqual(max(norm_asset_price_1), 1.0, places=3)

        # ADF test output
        self.assertAlmostEqual(adf_asset_1['Values'].mean(), -3.006066, places=3)
        self.assertAlmostEqual(adf_asset_2['Values'].mean(), -3.009356, places=3)

        self.assertAlmostEqual(test_stat_1, -1.050169, places=3)
        self.assertAlmostEqual(test_stat_2, -2.278697, places=3)

    def test_residual_analysis(self):
        """
        Tests the residual analysis part of the cointegration tear sheet.
        """

        # Generating datasets that would pass and fail the residuals test
        normal = pd.Series(np.random.normal(loc=0.0, scale=0.1, size=500))
        beta = pd.Series(np.random.beta(a=2, b=8, size=500))

        # Calling the tear sheet
        test = TearSheet()
        results_normal = test._residual_analysis(normal)
        results_beta = test._residual_analysis(beta)

        # Testing all for normal dataset
        (residuals, residuals_dataframe, qq_plot_y, qq_plot_x, pacf_result, acf_result) = results_normal

        # Residuals themself
        self.assertAlmostEqual(residuals.mean(), normal.mean(), places=5)

        # Statistics DataFrame
        self.assertAlmostEqual(residuals_dataframe.loc[residuals_dataframe['Characteristic'] == 'Standard Deviation',
                                                       'Value'].iloc[0], 0.09992, places=5)
        self.assertAlmostEqual(residuals_dataframe.loc[residuals_dataframe['Characteristic'] == 'Half-life',
                                                       'Value'].iloc[0], 0.68553, places=5)
        self.assertAlmostEqual(residuals_dataframe.loc[residuals_dataframe['Characteristic'] == 'Skewness',
                                                       'Value'].iloc[0], 0.02865, places=5)
        self.assertAlmostEqual(residuals_dataframe.loc[residuals_dataframe['Characteristic'] == 'Kurtosis',
                                                       'Value'].iloc[0], -0.13961, places=5)
        self.assertEqual(residuals_dataframe.loc[residuals_dataframe['Characteristic'] == 'Shapiro-Wilk normality test',
                                                 'Value'].iloc[0], 'Passed')

        # QQ Plots data
        self.assertAlmostEqual(qq_plot_y[0][0].mean(), 1, places=5)
        self.assertAlmostEqual(qq_plot_y[0][1].mean(), -0.002535, places=5)
        self.assertAlmostEqual(np.mean(qq_plot_y[1]), 0.332173, places=5)
        self.assertAlmostEqual(qq_plot_x.mean(), 1, places=5)

        # PACF and ACF
        self.assertAlmostEqual(pacf_result.mean(), -0.081438, places=5)
        self.assertAlmostEqual(acf_result.mean(), 0.024990, places=5)

        # Testing statistics for beta dataset
        (_, residuals_dataframe, _, _, _, _) = results_beta

        # Statistics DataFrame
        self.assertAlmostEqual(residuals_dataframe.loc[residuals_dataframe['Characteristic'] == 'Standard Deviation',
                                                       'Value'].iloc[0], 0.11909, places=5)
        self.assertAlmostEqual(residuals_dataframe.loc[residuals_dataframe['Characteristic'] == 'Half-life',
                                                       'Value'].iloc[0], 0.73223, places=5)
        self.assertAlmostEqual(residuals_dataframe.loc[residuals_dataframe['Characteristic'] == 'Skewness',
                                                       'Value'].iloc[0], 0.76284, places=5)
        self.assertAlmostEqual(residuals_dataframe.loc[residuals_dataframe['Characteristic'] == 'Kurtosis',
                                                       'Value'].iloc[0], 0.45668, places=5)
        self.assertEqual(residuals_dataframe.loc[residuals_dataframe['Characteristic'] == 'Shapiro-Wilk normality test',
                                                 'Value'].iloc[0], 'Failed')

    def test_adf_test_result(self):
        """
        Tests the interpretation of ADF test result.
        """

        # Calling the tear sheet and setting the data parameter
        test = TearSheet()
        test.data = self.dataframe

        # Getting ADF statistics table
        transformed_data = test._get_basic_assets_data()
        (_, _, _, _, _, _, adf_asset, _) = transformed_data

        # Some statistics to test
        test_statistics = [-3.6, -2.91, -2.60, -2.5]
        interpretation = []

        # Getting interpretation
        for stat in test_statistics:
            interpretation.append(test._adf_test_result(adf_asset, stat))

        # Testing interpretations
        self.assertEqual(interpretation[0][0], 'The hypothesis is not rejected at 99% confidence level')
        self.assertEqual(interpretation[1][0], 'The hypothesis is not rejected at 95% confidence level')
        self.assertEqual(interpretation[2][0], 'The hypothesis is not rejected at 90% confidence level')
        self.assertEqual(interpretation[3][0], 'HYPOTHESIS REJECTED')

    def test_johansen_test_result(self):
        """
        Tests the interpretation of Johansen test result.
        """

        # Calling the tear sheet and setting the data parameter
        test = TearSheet()
        test.data = self.dataframe

        # Getting Johansen test statistics table
        johansen_data = test._get_johansen_data()
        (test_eigen_dataframe, _, _, _, _, _, _, _, _, _, _, _) = johansen_data

        # Some statistics to test
        test_statistics = [[18.6, 6.65], [14.3, 3.85], [12.3, 2.71], [12.3, 2.7]]
        interpretation = []

        # Getting interpretation
        for stat in test_statistics:
            interpretation.append(test._johansen_test_result(test_eigen_dataframe, stat[0], stat[1]))

        # Testing interpretations
        self.assertEqual(interpretation[0][0], 'The hypothesis is not rejected at 99% confidence level')
        self.assertEqual(interpretation[1][0], 'The hypothesis is not rejected at 95% confidence level')
        self.assertEqual(interpretation[2][0], 'The hypothesis is not rejected at 90% confidence level')
        self.assertEqual(interpretation[3][0], 'HYPOTHESIS REJECTED')

    def test_cointegration_plot(self):
        """
        Tests plotting methods.
        """

        # Calling the tear sheet and setting the data parameter
        test = TearSheet()
        test.data = self.dataframe

        # Getting ADF statistics table
        transformed_data = test._get_basic_assets_data()
        (asset_name_1, norm_asset_price_1, _, _,
         asset_name_2, norm_asset_price_2, _, _) = transformed_data

        engle_granger_data = test._get_engle_granger_data(self.dataframe)
        (_, _, _, portfolio_returns, portfolio_price, residuals,
         _, qq_y, x, pacf_result, acf_result) = engle_granger_data

        asset_prices_plot = test._asset_prices_plot(norm_asset_price_1, norm_asset_price_2, asset_name_1, asset_name_2)
        portfolio_plot = test._portfolio_plot(portfolio_price, portfolio_returns)
        pacf_plot = test._pacf_plot(pacf_result)
        acf_plot = test._acf_plot(acf_result)
        residuals_plot = test._residuals_plot(residuals)
        qq_plot = test._qq_plot(qq_y, x)

        # Testing interpretations
        self.assertEqual(str(type(asset_prices_plot)), "<class 'plotly.graph_objs._figure.Figure'>")
        self.assertEqual(str(type(portfolio_plot)), "<class 'plotly.graph_objs._figure.Figure'>")
        self.assertEqual(str(type(pacf_plot)), "<class 'plotly.graph_objs._figure.Figure'>")
        self.assertEqual(str(type(acf_plot)), "<class 'plotly.graph_objs._figure.Figure'>")
        self.assertEqual(str(type(residuals_plot)), "<class 'plotly.graph_objs._figure.Figure'>")
        self.assertEqual(str(type(qq_plot)), "<class 'plotly.graph_objs._figure.Figure'>")

    def test_cointegration_div(self):
        """
        Tests plotting methods.
        """

        # Calling the tear sheet and setting the data parameter
        test = TearSheet()
        test.data = self.dataframe

        # Getting ADF statistics table
        transformed_data = test._get_basic_assets_data()
        (asset_name_1, _, _, _,
         asset_name_2, _, _, _) = transformed_data

        johansen_data = test._get_johansen_data()
        (test_eigen_dataframe, test_trace_dataframe, eigen_test_statistic_1, eigen_test_statistic_2,
         trace_test_statistic_1, trace_test_statistic_2, scaled_vector_1, _, portfolio_returns_1,
         portfolio_price_1, _, _) = johansen_data


        engle_granger_data = test._get_engle_granger_data(self.dataframe)
        (adf_dataframe, adf_test_stat, cointegration_vector, portfolio_returns, portfolio_price, residuals,
         residuals_dataframe, qq_y, x, pacf_result, acf_result) = engle_granger_data

        jh_coint_test_div = test._jh_coint_test_div(asset_name_1, asset_name_2, test_eigen_dataframe,
                                                    test_trace_dataframe, eigen_test_statistic_1,
                                                    eigen_test_statistic_2, trace_test_statistic_1,
                                                    trace_test_statistic_2)

        jh_div = test._jh_div(asset_name_1, asset_name_2, scaled_vector_1, portfolio_price_1, portfolio_returns_1)

        eg_div = test._eg_div(asset_name_1, asset_name_2, adf_dataframe, adf_test_stat, cointegration_vector.loc[0][1],
                              portfolio_price, portfolio_returns, pacf_result, acf_result, residuals, qq_y, x,
                              residuals_dataframe)

        # Testing interpretations
        self.assertEqual(str(type(jh_coint_test_div)), "<class 'dash_html_components.Div.Div'>")
        self.assertEqual(str(type(jh_div)), "<class 'dash_html_components.Div.Div'>")
        self.assertEqual(str(type(eg_div)), "<class 'dash_html_components.Div.Div'>")
