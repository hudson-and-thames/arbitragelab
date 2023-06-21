# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests function of Statistical Arbitrage Cointegration module:
cointegration_approach/base.py, engle_granger.py, johansen.py, signals.py
"""
import os
import unittest
import warnings

import pandas as pd

from arbitragelab.cointegration_approach import (JohansenPortfolio, EngleGrangerPortfolio,
                                                 get_half_life_of_mean_reversion)


class TestCointegration(unittest.TestCase):
    """
    Test Statistical Arbitrage cointegration functions.
    """

    def setUp(self):
        """
        Creates cointegrated random assets (X, Y, Z)
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_engle_granger(self):
        """
        Tests functions from EngleGrangerPortfolio for cointegration vectors and critical values.
        """
        port = EngleGrangerPortfolio()
        # Test using data set with 2 variables.
        port.fit(price_data=self.data[self.data.columns[:2]])

        self.assertEqual(port.cointegration_vectors[port.dependent_variable][0], 1)  # Dependent coef. is always 1
        self.assertAlmostEqual(port.cointegration_vectors['EWG'][0], -1.57, delta=1e-2)

        # Test using data set with 5 variables
        port.fit(price_data=self.data[self.data.columns[:5]], add_constant=True)

        self.assertAlmostEqual(port.cointegration_vectors.iloc[0].mean(), 0.35, delta=1e-2)
        self.assertAlmostEqual(port.cointegration_vectors['EFA'][0], -2.169, delta=1e-2)
        self.assertAlmostEqual(port.adf_statistics.loc['statistic_value'][0], -3.5, delta=1e-2)
        self.assertAlmostEqual(port.adf_statistics.loc['90%'][0], -2.57, delta=1e-2)

    def test_johansen(self):
        """
        Tests functions from JohansenPortfolio for eigenvectors and critical values
        """

        port = JohansenPortfolio()
        # Test using data set with more than 12 variables.
        # This warning is expected
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Critical values are only available')
            port.fit(price_data=self.data)

        # No critical values available if number of assets > 12
        self.assertEqual(port.johansen_eigen_statistic, None)
        self.assertEqual(port.johansen_trace_statistic, None)

        # Test using data with 5 assets only.
        port.fit(price_data=self.data[self.data.columns[:5]])

        # Check critical values (eigen and trace)
        self.assertAlmostEqual(port.johansen_eigen_statistic.loc['90%'].mean(), 18.05, delta=1e-2)
        self.assertAlmostEqual(port.johansen_trace_statistic.loc['95%'].mean(), 33.36, delta=1e-2)
        self.assertAlmostEqual(port.johansen_trace_statistic.loc['trace_statistic', 'EEM'], 64.01, delta=1e-2)
        self.assertAlmostEqual(port.johansen_trace_statistic.loc['trace_statistic', 'EWG'], 41.44, delta=1e-2)
        self.assertTrue(
            port.johansen_trace_statistic.loc['trace_statistic', 'EFA'] >= port.johansen_trace_statistic.loc[
                '90%', 'EFA'])

        # Check eigenvector
        self.assertAlmostEqual(port.cointegration_vectors.mean().mean(), 0.029, delta=1e-3)
        self.assertAlmostEqual(port.cointegration_vectors.iloc[0]['EFA'], -0.86, delta=1e-2)
        self.assertAlmostEqual(port.cointegration_vectors.iloc[0]['EWJ'], 0.05, delta=1e-2)

        # Test mean-reverting portfolio value formed by different eigenvectors
        mean_reverting_port_1 = port.construct_mean_reverting_portfolio(self.data[self.data.columns[:5]])
        mean_reverting_port_2 = port.construct_mean_reverting_portfolio(self.data[self.data.columns[:5]],
                                                                        cointegration_vector=
                                                                        port.cointegration_vectors.iloc[1])

        self.assertAlmostEqual(mean_reverting_port_1.mean(), -13.65, delta=1e-2)
        self.assertAlmostEqual(mean_reverting_port_2.mean(), 7.57, delta=1e-2)
        self.assertAlmostEqual(mean_reverting_port_1.iloc[5], -13.58, delta=1e-2)
        self.assertAlmostEqual(mean_reverting_port_2.iloc[5], 8.03, delta=1e-2)

        # Test scaled cointegration vector values
        scaled_cointegration_1 = port.get_scaled_cointegration_vector()
        scaled_cointegration_2 = port.get_scaled_cointegration_vector(cointegration_vector=
                                                                      port.cointegration_vectors.iloc[0])

        self.assertAlmostEqual(scaled_cointegration_1.iloc[0], 1.)
        self.assertAlmostEqual(scaled_cointegration_2.iloc[0], 1.)
        self.assertAlmostEqual(scaled_cointegration_1.iloc[1], scaled_cointegration_2.iloc[1], delta=1e-2)
        self.assertAlmostEqual(scaled_cointegration_2.mean(), scaled_cointegration_1.mean(), delta=1e-2)

    def test_half_life(self):
        """
        Tests function get_half_life_of_mean_reversion.
        """

        port = JohansenPortfolio()
        port.fit(price_data=self.data[self.data.columns[:5]])

        # Construct spread
        mean_reverting_port = port.construct_mean_reverting_portfolio(self.data[self.data.columns[:5]])
        half_life = get_half_life_of_mean_reversion(mean_reverting_port)

        self.assertAlmostEqual(half_life, 31.23, delta=1e-2)
