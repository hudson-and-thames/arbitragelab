# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests for the partner selection approaches from the Copula module.
"""
import os
import unittest
import numpy as np
import pandas as pd

from matplotlib import axes
from arbitragelab.copula_approach.vine_copula_partner_selection import PartnerSelection
from arbitragelab.copula_approach.vine_copula_partner_selection_utils import multivariate_rho_vectorized, get_quantiles_data, \
    extremal_measure, get_co_variance_matrix, get_sector_data, get_sum_correlations_vectorized, diagonal_measure_vectorized


class PartnerSelectionTests(unittest.TestCase):
    """
    Tests Partner Selection Approaches.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Creates dataframes with returns and quantile data to work with corresponding test cases.
        Using setUpClass instead of setUp for performance reasons as setUpClass is executed just once for all test cases.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/sp500_2016_test.csv'
        cls.quadruple = ['A', 'AAL', 'AAP', 'AAPL']
        prices = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna()
        cls.ps = PartnerSelection(prices)

        cls.u = cls.ps.returns.apply(get_quantiles_data, axis=0)

        cls.constituents = pd.read_csv(project_path + '/test_data/sp500_constituents-detailed.csv', index_col='Symbol')

    def test_traditional(self):
        """
        Tests Traditional Approach.
        """
        self.assertEqual(self.ps.traditional(1),[['A', 'AAL', 'AAPL', 'AAP']])

    def test_extended(self):
        """
        Tests Extended Approach.
        """
        self.assertEqual(self.ps.extended(1), [['A', 'AAL', 'AAPL', 'AAP']])

    def test_geometric(self):
        """
        Tests Geometric Approach.
        """
        self.assertEqual(self.ps.geometric(1),[['A', 'AAL', 'AAPL', 'AAP']])

    def test_extremal(self):
        """
        Tests Extremal Approach.
        """
        self.assertEqual(self.ps.extremal(1), [['A', 'AAL', 'AAPL', 'AAP']])

    def test_sum_correlations(self):
        """
        Tests helper function of Traditional Approach.
        """
        self.assertEqual(round(
            get_sum_correlations_vectorized(self.ps.correlation_matrix.loc[self.quadruple, self.quadruple],
                                            np.array([[0, 1, 2, 3]]))[1], 4), 1.9678)

    def test_multivariate_rho(self):
        """
        Tests helper function of Extended Approach.
        """
        self.assertEqual(round(multivariate_rho_vectorized(self.u[self.quadruple], np.array([[0,1,2,3]]))[1], 4), 0.3114)

    def test_diagonal_measure(self):
        """
        Tests helper function of Geometric Approach.
        """
        self.assertEqual(round(diagonal_measure_vectorized(self.ps.ranked_returns[self.quadruple], np.array([[0,1,2,3]]))[1], 4), 91.9374)

    def test_extremal_covariance_matrix(self):
        """
        Tests helper function for extremal approach which calculates the covariance matrix.
        """
        self.assertIsNone(np.testing.assert_almost_equal(get_co_variance_matrix(2), [[ 64., -16., -16.,   4.],
                                                                                     [-16., 64.,   4., -16.],
                                                                                     [-16.,   4.,  64., -16.],
                                                                                     [  4., -16., -16., 64.]] ,7))

    def test_extremal_measure(self):
        """
        Tests helper function of Extremal Approach.
        """
        co_variance_matrix = get_co_variance_matrix(4)
        self.assertEqual(round(extremal_measure(self.ps.ranked_returns[self.quadruple], co_variance_matrix), 4), 108.5128)

    def test_get_sector_data(self):
        """
        Tests Util method which returns sector data.
        """
        self.assertIsInstance(get_sector_data(self.quadruple, self.constituents), pd.DataFrame)

    def test_plot_selected_pairs(self):
        """
        Tests plot_selected_pairs plotting method.
        """
        self.assertIsInstance(self.ps.plot_selected_pairs([self.quadruple]), axes.Axes)
