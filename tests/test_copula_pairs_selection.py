# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Unit tests for pairs selection module under copula_approach.
"""
# pylint: disable = invalid-name, protected-access
import os
import unittest
import numpy as np
import pandas as pd
import arbitragelab.copula_approach.pairs_selection as pairs_selection


class TestPairsSelector(unittest.TestCase):
    """
    Testing methods in PairsSelector.
    """

    def setUp(self):
        # Using saved ETF price series for testing and trading
        project_path = os.path.dirname(__file__)
        data_path = project_path + "/test_data/stock_prices.csv"
        self.stocks = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_rank_pairs(self):
        """
        Testing for rank_pairs method.
        """
        PS = pairs_selection.PairsSelector()

        # Default options
        scores_dis = PS.rank_pairs(self.stocks, 'euc distance')
        scores_rho = PS.rank_pairs(self.stocks, 'spearman rho')
        scores_tau = PS.rank_pairs(self.stocks, 'kendall tau')
        # Check length
        self.assertEqual(len(scores_dis), 253)
        self.assertEqual(len(scores_rho), 253)
        self.assertEqual(len(scores_tau), 253)
        # Sample a few to check
        self.assertAlmostEqual(scores_dis['BND']['CSJ'], -1.3367629104092054, delta=1e-5)
        self.assertAlmostEqual(scores_dis['TIP']['CSJ'], -2.322229913467921, delta=1e-5)
        self.assertAlmostEqual(scores_dis['EFA']['SPY'], -16.65037813410612, delta=1e-5)

        self.assertAlmostEqual(scores_rho['BND']['CSJ'], 0.6877356800107545, delta=1e-5)
        self.assertAlmostEqual(scores_rho['TIP']['CSJ'], 0.7302595008128704, delta=1e-5)
        self.assertAlmostEqual(scores_rho['EFA']['SPY'], 0.6678733772844093, delta=1e-5)

        self.assertAlmostEqual(scores_tau['BND']['CSJ'], 0.5235355150320703, delta=1e-5)
        self.assertAlmostEqual(scores_tau['TIP']['CSJ'], 0.5575605525931701, delta=1e-5)
        self.assertAlmostEqual(scores_tau['EFA']['SPY'], 0.4980204822767728, delta=1e-5)

        # Given the number of pairs to keep
        scores_dis_cut = PS.rank_pairs(self.stocks, 'euc distance', keep_num_pairs=100)
        pd.testing.assert_series_equal(scores_dis_cut, scores_dis[:100])

    @staticmethod
    def test_pre_processing_nan():
        """
        Testing for _pre_processing_nan method.
        """

        # Initiate data and selctor
        PS = pairs_selection.PairsSelector()
        toy_data = {'A': [1, 2, 3, 4, np.NaN, 6],
                    'B': [np.NaN, 2, 3, 4, 5, 6],
                    'C': [1, 2, 3, 4, 5, np.NaN],
                    'D': [np.NaN, np.NaN, 3, 4, 5, 6],
                    'E': [1, 2, 3, 4, np.NaN, np.NaN],
                    'F': [1, 2, np.NaN, np.NaN, 5, 6]}
        toy_df = pd.DataFrame(data=toy_data, dtype=float)

        # Fill NaN
        forward_fill_df = PS._pre_processing_nan(toy_df, 'forward fill')
        linear_interp_df = PS._pre_processing_nan(toy_df, 'linear interp')
        none_df = PS._pre_processing_nan(toy_df, None)

        # Expected data for forward fill
        ff_expect_data = {'A': [1, 2, 3, 4, 4, 6],
                          'B': [np.NaN, 2, 3, 4, 5, 6],
                          'C': [1, 2, 3, 4, 5, 5],
                          'D': [np.NaN, np.NaN, 3, 4, 5, 6],
                          'E': [1, 2, 3, 4, 4, 4],
                          'F': [1, 2, 2, 2, 5, 6]}

        # Expected data for linear interp
        li_expect_data = {'A': [1, 2, 3, 4, 5, 6],
                          'B': [np.NaN, 2, 3, 4, 5, 6],
                          'C': [1, 2, 3, 4, 5, 5],
                          'D': [np.NaN, np.NaN, 3, 4, 5, 6],
                          'E': [1, 2, 3, 4, 4, 4],
                          'F': [1, 2, 3, 4, 5, 6]}

        ff_expect = pd.DataFrame(data=ff_expect_data, dtype=float)
        li_expect = pd.DataFrame(data=li_expect_data, dtype=float)

        # Checking with the result.
        pd.testing.assert_frame_equal(forward_fill_df, ff_expect, check_dtype=False)
        pd.testing.assert_frame_equal(linear_interp_df, li_expect, check_dtype=False)
        pd.testing.assert_frame_equal(none_df, toy_df, check_dtype=False)
