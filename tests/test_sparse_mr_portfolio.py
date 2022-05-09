# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=invalid-name
"""
Tests function of Sparse Mean-reverting Portfolio Selection module:
cointegration_approach/sparse_mr_portfolio.py
"""

import io
import os
import unittest
from unittest.mock import patch

import networkx as nx
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

        # Random seed
        np.random.seed(0)

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

        # The test output below depends on your machine!
        self.assertAlmostEqual(coeff, 9.526000811184625, delta=1e-2)
        self.assertAlmostEqual(hl, 18.33645545106608, delta=1e-2)

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

    def test_sdp_predictability_vol(self):
        """
        Test semidefinite relaxation optimization of predictability with a volatility threshold.
        """

        etf_sparse_portf = SparseMeanReversionPortfolio(self.data)

        # Perform SDP
        sdp_pred_vol_result = etf_sparse_portf.sdp_predictability_vol(rho=0.001, variance=5,
                                                                      max_iter=10000, verbose=False,
                                                                      use_standardized=False)

        sdp_pred_vol_weights = etf_sparse_portf.sparse_eigen_deflate(sdp_pred_vol_result, 8, verbose=False)

        sdp_pred_vol_weights_idx = sdp_pred_vol_weights.nonzero()
        sdp_pred_vol_weights_val = sdp_pred_vol_weights[sdp_pred_vol_weights_idx]

        # Verify minimization result
        self.assertIsNone(allclose(sdp_pred_vol_weights_idx[0],
                                   np.array([2, 5, 7, 11, 16, 28, 31, 41])))

        # The test output below depends on your machine!
        # These values were also changed with dependency updates in MlFinLab v1.6.0
        # Old values are [0.123879, -0.14753, -0.741946, -0.093619, 0.087365, 0.524529, -0.053632, 0.343513]
        np.testing.assert_array_almost_equal(sdp_pred_vol_weights_val,
                                             np.array([0.11, -0.08, -0.73, -0.11, 0.02, 0.54, -0.04, 0.39]),
                                             decimal=2)

    def test_sdp_portmanteau_vol(self):
        """
        Test semidefinite relaxation optimization of predictability with a volatility threshold.
        """

        etf_sparse_portf = SparseMeanReversionPortfolio(self.data)

        # Perform SDP
        sdp_port_vol_result = etf_sparse_portf.sdp_portmanteau_vol(rho=0.001, variance=5, nlags=3,
                                                                   max_iter=10000, verbose=False,
                                                                   use_standardized=False)
        sdp_port_vol_weights = etf_sparse_portf.sparse_eigen_deflate(sdp_port_vol_result, 8, verbose=False)

        sdp_port_vol_weights_idx = sdp_port_vol_weights.nonzero()
        sdp_port_vol_weights_val = sdp_port_vol_weights[sdp_port_vol_weights_idx]

        # Verify minimization result
        self.assertIsNone(allclose(sdp_port_vol_weights_idx[0],
                                   np.array([7, 9, 11, 16, 27, 28, 31, 38])))
        self.assertIsNone(allclose(sdp_port_vol_weights_val,
                                   np.array([0.35675015, -0.41894421, 0.47761845, -0.3175681,
                                             -0.24383743, -0.3019539, 0.29348114, 0.3626047]),
                                   rtol=1e-3))

    def test_sdp_crossing_vol(self):
        """
        Test semidefinite relaxation optimization of predictability with a volatility threshold.
        """

        etf_sparse_portf = SparseMeanReversionPortfolio(self.data)

        # Perform SDP
        sdp_cross_vol_result = etf_sparse_portf.sdp_crossing_vol(rho=0.001, mu=0.01, variance=5, nlags=3,
                                                                 max_iter=10000, verbose=False,
                                                                 use_standardized=False)
        sdp_cross_vol_weights = etf_sparse_portf.sparse_eigen_deflate(sdp_cross_vol_result, 8, verbose=False)

        sdp_cross_vol_weights_idx = sdp_cross_vol_weights.nonzero()
        sdp_cross_vol_weights_val = sdp_cross_vol_weights[sdp_cross_vol_weights_idx]

        # Verify minimization result
        self.assertIsNone(allclose(sdp_cross_vol_weights_idx[0],
                                   np.array([7, 9, 11, 16, 27, 28, 31, 38])))
        self.assertIsNone(allclose(actual=sdp_cross_vol_weights_val,
                                   desired=np.array([0.35380303, -0.41627861, 0.49314636, -0.30469972,
                                                     -0.24946532, -0.29270862, 0.28273044, 0.3710155]),
                                   rtol=1e-03))

    def test_LASSO_VAR_tuning(self):
        """
        Test LASSO paramater tuning for VAR(1) sparse estimate.
        """

        etf_sparse_portf = SparseMeanReversionPortfolio(self.data)

        # Test column-wise LASSO tuning
        best_alpha1 = etf_sparse_portf.LASSO_VAR_tuning(0.5, multi_task_lasso=False,
                                                        alpha_min=4e-4, alpha_max=6e-4,
                                                        n_alphas=10, max_iter=5000)
        self.assertAlmostEqual(best_alpha1, 0.0005111111111111111)

        # Test multi-task LASSO tuning
        best_alpha2 = etf_sparse_portf.LASSO_VAR_tuning(0.5, multi_task_lasso=True,
                                                        alpha_min=0.9, alpha_max=1.5,
                                                        n_alphas=10)
        self.assertAlmostEqual(best_alpha2, 1.3)

        # Test for error
        self.assertRaises(ValueError, etf_sparse_portf.LASSO_VAR_tuning, 0.2,
                          multi_task_lasso=False, alpha_min=0.1, alpha_max=0.2,
                          n_alphas=10, max_iter=1000)
        self.assertRaises(ValueError, etf_sparse_portf.LASSO_VAR_tuning, 0.5,
                          multi_task_lasso=True, alpha_min=0.6, alpha_max=0.7,
                          n_alphas=10, use_standardized=False)

    def test_covar_sparse_tuning(self):
        """
        Test graphical LASSO paramater tuning for covariance matrix sparse estimate.
        """

        etf_sparse_portf = SparseMeanReversionPortfolio(self.data)

        # Test for graphical LASSO tuning
        best_alpha = etf_sparse_portf.covar_sparse_tuning(alpha_min=0.7, alpha_max=0.9, n_alphas=20, clusters=4)

        self.assertAlmostEqual(best_alpha, 0.8157894736842105)

        # Test for errors
        self.assertRaises(ValueError, etf_sparse_portf.covar_sparse_tuning, alpha_min=0.5, alpha_max=0.6,
                          n_alphas=5, clusters=46)

        self.assertRaises(ValueError, etf_sparse_portf.covar_sparse_tuning, alpha_min=0.5, alpha_max=0.6,
                          n_alphas=10, clusters=4)

    def test_find_clusters(self):
        """
        Test clustering algorithm.
        """
        etf_sparse_portf = SparseMeanReversionPortfolio(self.data)

        # First try multi-task LASSO
        sparse_var_est = etf_sparse_portf.LASSO_VAR_fit(1.4, threshold=7, multi_task_lasso=True)
        _, sparse_prec_est = etf_sparse_portf.covar_sparse_fit(0.89)

        multi_LASSO_cluster_graph = etf_sparse_portf.find_clusters(sparse_prec_est, sparse_var_est)
        multi_LASSO_clusters = list(sorted(nx.connected_components(multi_LASSO_cluster_graph), key=len, reverse=True))

        target1 = ['EDEN', 'EWI', 'EWN', 'EWK', 'GXC', 'EWY', 'EIRL', 'EWL',
                   'INDA', 'ENZL', 'EWT', 'EWJ', 'EWG', 'EFNL', 'EWQ']
        target2 = ['GXG', 'EWP', 'EWO', 'THD', 'EZA', 'EWS', 'ECH', 'EWU']

        self.assertListEqual(sorted(list(multi_LASSO_clusters[0])), sorted(target1))
        self.assertListEqual(sorted(list(multi_LASSO_clusters[1])), sorted(target2))

        # Then try column-wise LASSO
        sparse_var_est = etf_sparse_portf.LASSO_VAR_fit(0.001, threshold=7, multi_task_lasso=False)

        column_LASSO_cluster_graph = etf_sparse_portf.find_clusters(sparse_prec_est, sparse_var_est)
        column_LASSO_clusters = list(sorted(nx.connected_components(column_LASSO_cluster_graph), key=len, reverse=True))

        target3 = ['EWG', 'EWK', 'EWN', 'EWY', 'EIRL', 'EFNL', 'ENZL', 'EWI', 'GXC', 'EWJ', 'EWL', 'EDEN', 'EWT', 'EWQ']
        target4 = ['NORW', 'THD', 'EWO', 'PGAL', 'EWS', 'EPU']

        self.assertListEqual(sorted(list(column_LASSO_clusters[0])), sorted(target3))
        self.assertListEqual(sorted(list(column_LASSO_clusters[1])), sorted(target4))

    def test_truncated_power(self):
        """
        Branch testing for leading sparse eigenvector extraction.
        """

        etf_sparse_portf = SparseMeanReversionPortfolio(self.data)

        # Generate a random positive semidefinite matrix
        dim = 10
        semipos_seed = np.random.rand(dim, dim)
        semipos = semipos_seed @ semipos_seed.T

        self.assertRaises(ValueError, etf_sparse_portf.sparse_eigen_deflate, semipos, 0)
        self.assertRaises(ValueError, etf_sparse_portf.sparse_eigen_deflate, semipos, 11)

        with self.assertWarns(Warning):
            etf_sparse_portf.sparse_eigen_deflate(-semipos, 5, verbose=False)

        # Max_iter was set to 2 to let the while loop finish on its own
        with patch('sys.stdout', new=io.StringIO()) as fakeOutput:
            etf_sparse_portf.sparse_eigen_deflate(semipos, 5, max_iter=2)
            self.assertRegex(fakeOutput.getvalue().strip(),
                             r'Iteration: [0-9]+, Objective function value: [-]?[0-9]+[.][0-9]*')
