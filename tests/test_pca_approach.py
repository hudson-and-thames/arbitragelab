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
from arbitragelab.other_approaches import PCAStrategy


class TestPCAStrategy(unittest.TestCase):
    """
    Tests PCAStrategy class.
    """

    def setUp(self):
        """
        Creates dataframe with returns to feed into the PCAStrategy.
        """

        np.random.seed(0)
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date").pct_change()[1:]
        self.volume = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.pca_strategy = PCAStrategy(n_components=10)

    def test_volume_modified_return(self):
        """
        Tests the function for volume modified returns.
        """

        vol_adj_returns = self.pca_strategy.volume_modified_return(self.data, self.volume, k=60)
        pd.testing.assert_index_equal(vol_adj_returns.columns, self.data.columns)

    def test_standardize_data(self):
        """
        Tests the function for input data standardization.
        """

        data_standardized, data_std = self.pca_strategy.standardize_data(self.data)

        # Check the standardized data and standard deviations
        self.assertTrue((data_standardized.mean() < 1e-10).all())
        pd.testing.assert_series_equal(data_std, self.data.std())

    def test_get_factorweights(self):
        """
        Tests the function to calculate weights (scaled eigenvectors).
        """

        factorweights = self.pca_strategy.get_factorweights(self.data, explained_var=0.55)

        # Check factor weights
        self.assertAlmostEqual(factorweights.mean()['EEM'], 11.384, delta=1e-3)
        self.assertAlmostEqual(factorweights.mean()['XLF'], 8.9781, delta=1e-3)
        self.assertAlmostEqual(factorweights.mean()['SPY'], 18.425, delta=1e-3)

    def test_get_asym_factorweights(self):
        """
        Tests the function to calculate asymptotic pca factor weights.
        """
        factorweights = self.pca_strategy.get_asym_factorweights(self.data, explained_var=None)

        # Check factor weights
        self.assertAlmostEqual(factorweights.mean()['EEM'], -7.966358, delta=1e-3)
        self.assertAlmostEqual(factorweights.mean()['XLF'], 2.3900065, delta=1e-3)
        self.assertAlmostEqual(factorweights.mean()['SPY'], 5.6288474, delta=1e-3)

    def test_get_residuals(self):
        """
        Tests the function to calculate residuals from given matrices of returns and factor returns.
        """

        # Calculating factor returns
        factorweights = self.pca_strategy.get_factorweights(self.data, explained_var=0.55)
        factorret = pd.DataFrame(np.dot(self.data, factorweights.transpose()), index=self.data.index)

        residual, coefficient, intercept = self.pca_strategy.get_residuals(self.data, factorret)

        # Check residuals and coefficients
        self.assertAlmostEqual(residual.mean()['EEM'], 0, delta=1e-15)
        self.assertAlmostEqual(residual.mean()['XLF'], 0, delta=1e-15)
        self.assertAlmostEqual(residual.mean()['SPY'], 0, delta=1e-15)

        self.assertAlmostEqual(coefficient.mean()['XLF'], 0.0051443, delta=1e-5)
        self.assertAlmostEqual(coefficient.mean()['XLE'], 0.0045190, delta=1e-5)
        self.assertAlmostEqual(coefficient.mean()['XLK'], 0.0033304, delta=1e-5)

        self.assertAlmostEqual(intercept['XLF'], 0.0000956, delta=1e-5)
        self.assertAlmostEqual(intercept['XLE'], 0.0000527, delta=1e-5)
        self.assertAlmostEqual(intercept['XLK'], 0.0002801, delta=1e-5)

    def test_get_sscores(self):
        """
        Tests the function to calculate S-scores.
        """

        # Calculating residuals
        factorweights = self.pca_strategy.get_factorweights(self.data, explained_var=0.55)
        factorret = pd.DataFrame(np.dot(self.data, factorweights.transpose()), index=self.data.index)
        residual, _, intercept = self.pca_strategy.get_residuals(self.data, factorret)

        s_scores = self.pca_strategy.get_sscores(residual, intercept, k=4, drift=True)

        # Check S-scores
        self.assertAlmostEqual(s_scores['CSJ'], -1.773595, delta=1e-5)
        self.assertAlmostEqual(s_scores['VPL'], -0.739174, delta=1e-5)
        self.assertAlmostEqual(s_scores.mean(), -0.520698, delta=1e-5)

    def test_get_signals(self):
        """
        Tests the function to generate trading signals for given returns matrix with parameters.
        """

        # Taking a smaller dataset
        smaller_dataset = self.data[:270]
        smaller_volume = self.volume[:270]

        target_weights = self.pca_strategy.get_signals(smaller_dataset, k=8.4, corr_window=252,
                                                       residual_window=60, sbo=1.25, sso=1.25, ssc=0.5,
                                                       sbc=0.75, size=1)

        # Check target weights
        self.assertAlmostEqual(target_weights.mean()['EEM'], 0.094400, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['XLF'], 0.243965, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['SPY'], -0.19903, delta=1e-5)

        # Generating weights using higher mean reversion speed threshold

        target_weights = self.pca_strategy.get_signals(smaller_dataset, k=12, corr_window=252,
                                                       residual_window=60, sbo=1.25, sso=1.25, ssc=0.5,
                                                       sbc=0.75, size=1)

        # Check target weights
        self.assertAlmostEqual(target_weights.mean()['EEM'], 0.150266, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['XLF'], 0.303796, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['SPY'], -0.28778, delta=1e-5)

        # Check drift argument
        target_weights = self.pca_strategy.get_signals(smaller_dataset, k=12, corr_window=252,
                                                       residual_window=60, sbo=1.25, sso=1.25, ssc=0.5,
                                                       sbc=0.75, size=1, drift=True)

        # Check target weights
        self.assertAlmostEqual(target_weights.mean()['EEM'], 0.150266, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['XLF'], 0.303796, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['SPY'], -0.28778, delta=1e-5)

        # Check asymptotic PCA and explained_variance argument
        target_weights = self.pca_strategy.get_signals(smaller_dataset, k=8.4, corr_window=252,
                                                       residual_window=60, sbo=1.25,
                                                       sso=1.25, ssc=0.5, sbc=0.75,
                                                       size=1, explained_var=0.45,
                                                       asym=False)
        # Check target weights
        self.assertAlmostEqual(target_weights.mean()['EEM'], 0.074310, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['XLF'], 0.015396, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['SPY'], -0.30074, delta=1e-5)

        # Check asymptotic PCA and explained_variance argument
        target_weights = self.pca_strategy.get_signals(smaller_dataset, k=8.4, corr_window=252,
                                                       residual_window=60, sbo=1.25,
                                                       sso=1.25, ssc=0.5, sbc=0.75,
                                                       size=1, explained_var=0.45,
                                                       asym=True)

        # Check target weights
        self.assertAlmostEqual(target_weights.mean()['EEM'], -0.00703, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['XLF'], 0.010245, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['SPY'], -0.03177, delta=1e-5)

        # Check volume
        target_weights = self.pca_strategy.get_signals(smaller_dataset, smaller_volume, k=8.4,
                                                       corr_window=252,
                                                       residual_window=60, sbo=1.25,
                                                       sso=1.25, ssc=0.5, sbc=0.75,
                                                       size=1, explained_var=0.55,
                                                       asym=False)

        # Check target weights
        self.assertAlmostEqual(target_weights.mean()['EEM'], 0.340827, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['XLF'], -0.12825, delta=1e-5)
        self.assertAlmostEqual(target_weights.mean()['SPY'], -0.07829, delta=1e-5)
