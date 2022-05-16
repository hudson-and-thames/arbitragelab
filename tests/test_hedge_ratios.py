# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module which tests hedge ratios module.
"""
# pylint: disable=invalid-name

import unittest
import pandas as pd
import numpy as np

from arbitragelab.hedge_ratios.linear import get_ols_hedge_ratio, get_tls_hedge_ratio
from arbitragelab.hedge_ratios.half_life import get_minimum_hl_hedge_ratio
from arbitragelab.hedge_ratios.adf_optimal import get_adf_optimal_hedge_ratio
from arbitragelab.hedge_ratios.johansen import get_johansen_hedge_ratio
from arbitragelab.hedge_ratios.box_tiao import get_box_tiao_hedge_ratio


class TestHedgeRatios(unittest.TestCase):
    """
    Tests hedge ratios (OLS, TLS, Min HL).
    """

    def setUp(self):
        """
        Generates a dataset for hedge ratios calculation.
        """

        rs = np.random.RandomState(42)
        X_returns = rs.normal(0, 1, 100)
        X = pd.Series(np.cumsum(X_returns), name='X') + 50

        noise = rs.normal(0, 1, 100)
        Y = 5 * X + noise
        Y.name = 'Y'

        self.cointegrated_series = pd.concat([X, Y], axis=1)

    def test_ols_hedge_ratio(self):
        """
        Test OLS hedge ratio calculation.
        """

        hedge_ratios, _, _, residuals = get_ols_hedge_ratio(price_data=self.cointegrated_series, dependent_variable='Y')
        hedge_ratios_constant, _, _, residuals_const = get_ols_hedge_ratio(price_data=self.cointegrated_series,
                                                                           dependent_variable='Y',
                                                                           add_constant=True)
        self.assertAlmostEqual(hedge_ratios['X'], 5, delta=1e-3)
        self.assertAlmostEqual(hedge_ratios_constant['X'], 5, delta=1e-2)
        self.assertAlmostEqual(residuals.mean(), 0, delta=1e-2)
        self.assertAlmostEqual(residuals_const.mean(), 0, delta=1e-2)

    def test_tls_hedge_ratio(self):
        """
        Test TLS hedge ratio calculation.
        """

        hedge_ratios, _, _, residuals = get_tls_hedge_ratio(price_data=self.cointegrated_series, dependent_variable='Y')
        hedge_ratios_constant, _, _, residuals_const = get_tls_hedge_ratio(price_data=self.cointegrated_series,
                                                                           dependent_variable='Y',
                                                                           add_constant=True)
        self.assertAlmostEqual(hedge_ratios['X'], 5, delta=1e-3)
        self.assertAlmostEqual(hedge_ratios_constant['X'], 5, delta=1e-2)
        self.assertAlmostEqual(residuals.mean(), 0, delta=1e-2)
        self.assertAlmostEqual(residuals_const.mean(), 0, delta=1e-2)

    def test_hl_hedge_ratio(self):
        """
        Test HL hedge ratio calculation.
        """

        hedge_ratios, _, _, residuals, _ = get_minimum_hl_hedge_ratio(price_data=self.cointegrated_series,
                                                                      dependent_variable='Y')
        self.assertAlmostEqual(hedge_ratios['X'], 5, delta=1e-3)
        self.assertAlmostEqual(residuals.mean(), 0.06, delta=1e-2)

    def test_adf_hedge_ratio(self):
        """
        Test ADF optimal hedge ratio calculation.
        """

        hedge_ratios, _, _, residuals, _ = get_adf_optimal_hedge_ratio(price_data=self.cointegrated_series,
                                                                       dependent_variable='Y')
        self.assertAlmostEqual(hedge_ratios['X'], 5.0023, delta=1e-3)
        self.assertAlmostEqual(residuals.mean(), -0.080760, delta=1e-2)

    def test_johansen_hedge_ratio(self):
        """
        Test Johansen hedge ratio calculation.
        """

        hedge_ratios, _, _, residuals = get_johansen_hedge_ratio(price_data=self.cointegrated_series,
                                                                 dependent_variable='Y')
        self.assertAlmostEqual(hedge_ratios['X'], 5.00149, delta=1e-3)
        self.assertAlmostEqual(residuals.mean(), -0.04267, delta=1e-2)

    def test_box_tiao_hedge_ratio(self):
        """
        Test Box-Tiao decomposition hedge ratio calculation.
        """

        hedge_ratios, _, _, residuals = get_box_tiao_hedge_ratio(price_data=self.cointegrated_series,
                                                                 dependent_variable='Y')
        self.assertAlmostEqual(hedge_ratios['X'], 5.0087, delta=1e-3)
        self.assertAlmostEqual(residuals.mean(), -0.3609, delta=1e-2)

    def test_diverging_hedge_ratios(self):
        """
        Test diverging min HL, min ADF hedge ratios.
        """
        diverging_series = self.cointegrated_series.copy()
        diverging_series['Y'] = 1.0
        diverging_series['X'] = 2.0
        hedge_ratios, _, _, residuals, res = get_adf_optimal_hedge_ratio(price_data=diverging_series,
                                                                         dependent_variable='Y')
        self.assertEqual(res.status, 3.0)
        hedge_ratios, _, _, residuals, res = get_minimum_hl_hedge_ratio(price_data=diverging_series,
                                                                        dependent_variable='Y')
        self.assertEqual(res.status, 3.0)
