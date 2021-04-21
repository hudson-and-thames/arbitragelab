"""
Module which tests hedge ratios module.
"""

import unittest
import pandas as pd
import numpy as np
from arbitragelab.hedge_ratios.linear import get_ols_hedge_ratio, get_tls_hedge_ratio
from arbitragelab.hedge_ratios.half_life import get_minimum_hl_hedge_ratio


# pylint: disable=invalid-name
class TestHedgeRatios(unittest.TestCase):
    """
    Tests hedge ratios (OLS, TLS, Min HL).
    """

    def setUp(self):
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
        clf, _, _, residuals = get_ols_hedge_ratio(price_data=self.cointegrated_series, dependent_variable='Y')
        clf_constant, _, _, residuals_const = get_ols_hedge_ratio(price_data=self.cointegrated_series,
                                                                  dependent_variable='Y',
                                                                  add_constant=True)
        self.assertAlmostEqual(clf.coef_[0], 5, delta=1e-3)
        self.assertAlmostEqual(clf_constant.coef_[0], 5, delta=1e-2)
        self.assertAlmostEqual(residuals.mean(), 0, delta=1e-2)
        self.assertAlmostEqual(residuals_const.mean(), 0, delta=1e-2)

    def test_tls_hedge_ratio(self):
        """
        Test TLS hedge ratio calculation.
        """
        clf, _, _, residuals = get_tls_hedge_ratio(price_data=self.cointegrated_series, dependent_variable='Y')
        self.assertAlmostEqual(clf.beta[0], 5, delta=1e-3)
        self.assertAlmostEqual(residuals.mean(), 0, delta=1e-2)

    def test_hl_hedge_ratio(self):
        """
        Test HL hedge ratio calculation.
        """
        clf, _, _, residuals = get_minimum_hl_hedge_ratio(price_data=self.cointegrated_series, dependent_variable='Y')
        self.assertAlmostEqual(clf.x[0], 5, delta=1e-3)
        self.assertAlmostEqual(residuals.mean(), 0.06, delta=1e-2)
