# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/arbitragelab/blob/master/LICENSE.txt
"""
Tests function of ML Pairs Selection module:
ml_based_pairs_selection/pairs_selector.py
"""
import os
import unittest
import pandas as pd
import numpy as np
from arbitragelab.ml_based_pairs_selection import PairsSelector

class TestPairsSelector(unittest.TestCase):
    """
    Tests Pairs Selector class.
    """

    def setUp(self):
        """
        Loads price universe and instantiates the pairs selection class.
        """
        np.random.seed(0)

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/data.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date").dropna()
        self.pair_selector = PairsSelector(self.data)

    def test_dimensionality_reduction(self):
        """
        Runs pca on price universe and verifies std dev for all components.
        """
        self.pair_selector.dimensionality_reduction_by_components(5)

        expected_stds = pd.Series([0.008302, 0.045064, 0.045148, 0.045169, 0.045146])
        actual_stds = self.pair_selector.feature_vector.std()
        pd.testing.assert_series_equal(expected_stds, actual_stds, check_less_precise=True)

    def test_clustering(self):
        """
        Verifies generated clusters from both techniques in the PairsSelector class.
        """
        self.pair_selector.dimensionality_reduction_by_components(5)
        self.pair_selector.cluster_using_optics({'min_samples': 3})
        self.assertEqual(len(np.unique(self.pair_selector.clust.labels_)), 47)
        self.pair_selector.cluster_using_dbscan({'eps': 0.03, 'min_samples': 3})
        self.assertEqual(len(np.unique(self.pair_selector.clust.labels_)), 11)

    def test_hurst_criterion(self):
        """
        Verifies private hurst processing method.
        """
        self.pair_selector.dimensionality_reduction_by_components(5)
        self.pair_selector.cluster_using_optics({'min_samples': 3})
        hedge_ratios = [0.832406370860649, 0.892407527838706]
        idx = [('AJG', 'ICE'), ('AJG', 'MMC')]
        input_pairs = pd.DataFrame(data=hedge_ratios, columns=['hedge_ratio'], index=idx)
        result = self.pair_selector._hurst_criterion(input_pairs)
        pd.testing.assert_series_equal(pd.Series(idx), pd.Series(result[1].index))

    def test_final_criterions(self):
        """
        Verifies private final criterions processing method.
        """
        self.pair_selector.dimensionality_reduction_by_components(5)
        self.pair_selector.cluster_using_optics({'min_samples': 3})
        hedge_ratios = [0.832406370860649, 0.892407527838706]
        idx = [('AJG', 'ICE'), ('AJG', 'MMC')]
        input_pairs = pd.DataFrame(data=hedge_ratios, columns=['hedge_ratio'], index=idx)
        spreads_df, hurst_pass_pairs = self.pair_selector._hurst_criterion(input_pairs)
        hl_pairs, final_pairs = self.pair_selector._final_criterions(spreads_df, hurst_pass_pairs.index.values)
        pd.testing.assert_series_equal(pd.Series(idx), pd.Series(hl_pairs.index))
        pd.testing.assert_series_equal(pd.Series(idx), pd.Series(final_pairs.index))

    def test_criterion_selector(self):
        """
        Verifies final user exposed criterion selection method.
        """
        self.pair_selector.dimensionality_reduction_by_components(5)
        self.pair_selector.cluster_using_optics({'min_samples': 3})
        final_pairs = [('AJG', 'ICE'), ('AJG', 'MMC')]
        coint_pairs = [('MAA', 'UDR'), ('ATO', 'XEL'), ('CCL', 'MAR')]
        input_pairs = final_pairs + coint_pairs
        result = self.pair_selector._criterion_selection(input_pairs)
        pd.testing.assert_series_equal(pd.Series(final_pairs), pd.Series(self.ps.coint_pass_pairs.index))
        pd.testing.assert_series_equal(pd.Series(final_pairs), pd.Series(result))
