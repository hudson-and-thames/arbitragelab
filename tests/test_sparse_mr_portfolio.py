# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=invalid-name
"""
Tests function of Sparse Mean-reverting Portfolio Selection module:
cointegration_approach/sparse_mr_portfolio.py
"""

import os
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose as allclose
from sklearn.linear_model import LinearRegression

from arbitragelab.cointegration_approach.sparse_mr_portfolio import SparseMeanReversionPortfolio


class TestSparseMeanReversionPortfolio(unittest.TestCase):
    """
    Test Sparse Mean-reverting Portfolio Selection module.
    """

    def setUp(self):
        """
        Set up the data and parameters.

        Data: 45 international equity ETFs starting from Jan 01, 2016 to Jan 27, 2021.
        """

        # Read data
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/Country_ETF.csv'
        self.data = pd.read_csv(data_path, parse_dates=['Dates'])

        # Set date index
        self.data.set_index("Dates", inplace=True)

        # Fill missing values
        self.data.fillna(method='ffill', inplace=True)

        # Tidy up the column names
        self.data.columns = [x.split()[0] for x in self.data.columns]

        # Data with only 2 asset
        self.two_col_data = self.data[['EWG', 'RSX']]

        # Data with only 8 assets
        self.eight_col_data = self.data[['EGPT', 'ARGT', 'EWZ', 'EWW', 'GXG', 'ECH', 'EWY', 'TUR']]

    def test_property(self):
        """
        Test the class properties, i.e. asset prices, zero-mean centered prices, and standardized prices.
        """

        etf_sparse_portf = SparseMeanReversionPortfolio(self.data)

        # Raw price
        self.assertIsNone(allclose(etf_sparse_portf.assets.head(5)['KSA'],
                                   np.array([24.22, 23.53, 23.53, 22.66, 21.89]),
                                   rtol=1e-2))

        # Demeaned price
        self.assertIsNone(allclose(etf_sparse_portf.demeaned.head(5)['KSA'],
                                   np.array([-3.20941798, -3.89931798, -3.89931798, -4.76931798, -5.53931798])))

        # Standardized price
        self.assertIsNone(allclose(etf_sparse_portf.standardized.head(5)['KSA'],
                                   np.array([-0.89618234, -1.08882667, -1.08882667, -1.33176126, -1.54677233])))

    def test_autocov(self):
        """
        Test the autocovariance calculation procedure.
        """

        test_portf = SparseMeanReversionPortfolio(self.two_col_data)

        # Covariance matrix: Standardized
        self.assertIsNone(allclose(test_portf.autocov(0),
                                   np.array([[1.00075586, 0.61322908], [0.61322908, 1.00075586]])))

        # Covariance matrix: Demeaned
        self.assertIsNone(allclose(test_portf.autocov(0, use_standardized=False),
                                   np.array([[8.834219, 4.65518], [4.65518, 6.533043]])))

        # Test if the covariance matrix is symmetric
        self.assertTrue(test_portf.check_symmetric(test_portf.autocov(0, use_standardized=False)))

        # Test if the covariance matrix is positive semidefinite
        self.assertTrue(test_portf.is_semi_pos_def(test_portf.autocov(0, use_standardized=False)))

        # Autocovariance matrix: Demeaned
        self.assertIsNone(allclose(test_portf.autocov(1, use_standardized=False),
                                   np.array([[8.769389, 4.605863], [4.605863, 6.451951]])))

        # Autocovariance matrix: Standardized
        self.assertIsNone(allclose(test_portf.autocov(1),
                                   np.array([[0.99341177, 0.60673257], [0.60673257, 0.98833393]])))

        # Autocovariance matrix: Not symmetrize
        self.assertIsNone(allclose(test_portf.autocov(1, use_standardized=False, symmetrize=False),
                                   np.array([[8.769389, 4.587468], [4.624258, 6.451951]])))

    def test_box_tiao(self):
        """
        Test Box-Tiao canonical decomposition.
        """

        etf_sparse_portf = SparseMeanReversionPortfolio(self.eight_col_data)

        # Perform Box-Tiao canonical decomposition
        bt_weights = etf_sparse_portf.box_tiao()

        # Check two weights
        self.assertIsNone(allclose(bt_weights[:, 0],
                                   np.array([0.05362480, -0.1052639, -0.1946070, 0.0003169,
                                             -0.2509068, -0.4472856, 0.8276406, -0.01281850])))
        self.assertIsNone(allclose(bt_weights[:, -1],
                                   np.array([0.1207307, -0.0278675, 0.269331, 0.5115483,
                                             -0.7883157, 0.1206775, -0.0236125, -0.117747])))

        # Test mean-reversion coefficient
        coeff, hl = etf_sparse_portf.mean_rev_coeff(bt_weights[:, -1], etf_sparse_portf.assets, interval='D')
        self.assertAlmostEqual(coeff, 9.531241879155678)
        self.assertAlmostEqual(hl, 18.32637254575472)

    def test_mean_rev_coeff_error(self):
        """
        Test exception handling of OU mean-reversion speed and half-life calculation.
        """

        etf_sparse_portf = SparseMeanReversionPortfolio(self.eight_col_data)

        # Input a wrong weight vector
        self.assertRaises(np.linalg.LinAlgError, etf_sparse_portf.mean_rev_coeff,
                          np.array([0.1, 0.4, 0.2, 0.3]), etf_sparse_portf.assets, interval='D')

    def test_least_square_VAR_fit(self):
        """
        Test least-square VAR(1) estimate calculation.
        """
        etf_sparse_portf = SparseMeanReversionPortfolio(self.eight_col_data)

        # Test standardized data
        std_matrix = etf_sparse_portf.least_square_VAR_fit(use_standardized=True)
        std_now = etf_sparse_portf.standardized[1:]
        std_lag = etf_sparse_portf.standardized[:-1]

        std_result = LinearRegression().fit(std_lag, std_now)
        self.assertIsNone(allclose(std_matrix, std_result.coef_))

        # Test demeaned data
        demean_matrix = etf_sparse_portf.least_square_VAR_fit(use_standardized=False)
        demean_now = etf_sparse_portf.demeaned[1:]
        demean_lag = etf_sparse_portf.demeaned[:-1]

        demean_result = LinearRegression().fit(demean_lag, demean_now)
        self.assertIsNone(allclose(demean_matrix, demean_result.coef_))

    def test_greedy_search(self):
        """
        Test greedy search algorithm.
        """

        etf_sparse_portf = SparseMeanReversionPortfolio(self.data)

        # Get covariance and VAR(1) estimate without standardization
        full_cov_est = etf_sparse_portf.autocov(0, use_standardized=False)
        full_var_est = etf_sparse_portf.least_square_VAR_fit(use_standardized=False)

        # Do a minimization
        greedy_weight_min = etf_sparse_portf.greedy_search(8, full_var_est, full_cov_est, maximize=False).squeeze()

        # Do a maximization
        greedy_weight_max = etf_sparse_portf.greedy_search(8, full_var_est, full_cov_est, maximize=True).squeeze()

        # Minimization sparse representation
        greedy_weight_min_nonzero_idx = greedy_weight_min.nonzero()
        greedy_weight_min_nonzero_val = greedy_weight_min[greedy_weight_min_nonzero_idx]

        # Maximization sparse representation
        greedy_weight_max_nonzero_idx = greedy_weight_max.nonzero()
        greedy_weight_max_nonzero_val = greedy_weight_max[greedy_weight_max_nonzero_idx]

        # Verify minimization result
        self.assertIsNone(allclose(greedy_weight_min_nonzero_idx[0],
                                   np.array([2, 28, 36, 38, 40, 41, 43, 44])))

        self.assertIsNone(allclose(greedy_weight_min_nonzero_val,
                                   np.array([-0.0350839, -0.332868, 0.2062478, 0.5046533,
                                             -0.6805627, 0.3154246, -0.0264863, -0.165515])))

        # Verify maximization result
        self.assertIsNone(allclose(greedy_weight_max_nonzero_idx[0],
                                   np.array([2, 9, 10, 34, 36, 37, 39, 42])))

        self.assertIsNone(allclose(greedy_weight_max_nonzero_val,
                                   np.array([-0.0755304, -0.0269491, 0.0056099, 0.9055339,
                                             0.0175471, 0.2370267, -0.1867467, -0.2866737])))
