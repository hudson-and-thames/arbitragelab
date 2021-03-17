# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests the PCA Strategy from the Other Approaches module.
"""

import unittest
import os
import pandas as pd
import numpy as np
from arbitragelab.other_approaches import ETFStrategy


class TestPCAStrategy(unittest.TestCase):
    """
    Tests PCAStrategy class.
    """

    def setUp(self):
        """
        Creates dataframe with returns to feed into the ETFStrategy.
        """

        np.random.seed(0)
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        volume_path = project_path + '/test_data/stock_volume.csv'
        etf_path = project_path + '/test_data/etf.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date").pct_change()[1:]
        self.etf = pd.read_csv(etf_path, parse_dates=True, index_col="Date").pct_change()[1:]
        self.volume = pd.read_csv(volume_path, parse_dates=True, index_col="Date")
        self.etf_strategy = ETFStrategy(n_components=10)

    def test_volume_modified_return(self):
        """
        Tests the function for volume modified returns.
        """

        vol_adj_returns = self.etf_strategy.volume_modified_return(self.data, self.volume, k=60)
        pd.testing.assert_index_equal(vol_adj_returns.columns, self.data.columns)

    def test_standardize_data(self):
        """
        Tests the function for input data standardization.
        """

        data_standardized, data_std = self.etf_strategy.standardize_data(self.data)

        # Check the standardized data and standard deviations
        self.assertTrue((data_standardized.mean() < 1e-10).all())
        pd.testing.assert_series_equal(data_std, self.data.std())

    def test_get_residuals(self):
        """
        Tests the function to calculate residuals from given matrices of returns and factor returns.
        """

        # Calculating factor returns
        residual, coefficient, intercept = self.etf_strategy.get_residuals(self.data, self.etf)

        # Check residuals and coefficients
        self.assertAlmostEqual(residual.mean()['EEM'], 0, delta=1e-15)
        self.assertAlmostEqual(residual.mean()['XLF'], 0, delta=1e-15)
        self.assertAlmostEqual(residual.mean()['SPY'], 0, delta=1e-15)

        self.assertAlmostEqual(coefficient.mean()['XLF'], 0.0836902, delta=1e-5)
        self.assertAlmostEqual(coefficient.mean()['XLE'], 0.0835439, delta=1e-5)
        self.assertAlmostEqual(coefficient.mean()['XLK'], 0.0834975, delta=1e-5)

        self.assertAlmostEqual(intercept['XLF'], -0.000083, delta=1e-5)
        self.assertAlmostEqual(intercept['XLE'], -0.000079, delta=1e-5)
        self.assertAlmostEqual(intercept['XLK'], -0.000066, delta=1e-5)

    def test_get_sscores(self):
        """
        Tests the function to calculate S-scores.
        """

        # Calculating residuals

        residual, _, intercept = self.etf_strategy.get_residuals(self.data, self.etf)

        s_scores = self.etf_strategy.get_sscores(residual, intercept, k=1, drift=False)

        # Check S-scores
        self.assertAlmostEqual(s_scores['XLE'], 3.6094269, delta=1e-5)
        self.assertAlmostEqual(s_scores['EWG'], -0.580142, delta=1e-5)
        self.assertAlmostEqual(s_scores.mean(), 1.4296753, delta=1e-5)

        s_scores = self.etf_strategy.get_sscores(residual, intercept, k=2, drift=True, p_value=0.05)

        # Check S-scores
        self.assertAlmostEqual(s_scores['CSJ'], 0.4251045, delta=1e-5)
        self.assertAlmostEqual(s_scores['VPL'], -0.820569, delta=1e-5)
        self.assertAlmostEqual(s_scores.mean(), 2.3425592, delta=1e-5)

    def test_get_signals(self):
        """
        Tests the function to generate trading signals for given returns matrix with parameters.
        """

        # Taking a smaller dataset
        smaller_dataset = self.data[:270]
        smaller_etf = self.etf[:270]
        smaller_volume = self.volume[:270]

        target_weights = self.etf_strategy.get_signals(smaller_etf, smaller_dataset, k=1, corr_window=252,
                                                       residual_window=60, sbo=1.25, sso=1.25, ssc=0.5,
                                                       sbc=0.75, size=1)

        # Check target weights
        self.assertAlmostEqual(target_weights.mean()['EEM'], 0.333333, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['BND'], -0.5, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['SPY'], -0.38888, delta=1e-5)

        # Check drift argument
        target_weights = self.etf_strategy.get_signals(smaller_etf, smaller_dataset, k=1, corr_window=252,
                                                       residual_window=60, sbo=1.25, sso=1.25, ssc=0.5,
                                                       sbc=0.75, size=1, drift=True)

        # Check target weights
        self.assertAlmostEqual(target_weights.mean()['EEM'], 0.333333, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['BND'], -0.5, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['SPY'], -0.38888, delta=1e-5)

        # Check p_value
        target_weights = self.etf_strategy.get_signals(smaller_etf, smaller_dataset, k=1, corr_window=252,
                                                       residual_window=60, sbo=1.25,
                                                       sso=1.25, ssc=0.5, sbc=0.75,
                                                       size=1, p_value=0.2)

        # Check target weights
        self.assertAlmostEqual(target_weights.mean()['EEM'], 0.333333, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['BND'], -0.5, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['SPY'], -0.38888, delta=1e-5)

        # Check volume
        target_weights = self.etf_strategy.get_signals(smaller_etf, smaller_dataset, smaller_volume, k=1,
                                                       corr_window=252,
                                                       residual_window=60, sbo=1.25,
                                                       sso=1.25, ssc=0.5, sbc=0.75,
                                                       size=1)

        # Check target weights
        self.assertAlmostEqual(target_weights.mean()['EEM'], -0.47058, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['BND'], 0.0, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['SPY'], -0.35294, delta=1e-5)
