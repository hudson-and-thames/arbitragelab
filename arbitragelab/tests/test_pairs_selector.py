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

# pylint: disable=protected-access

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
        self.data = pd.read_csv(
            data_path, parse_dates=True, index_col="Date").dropna()
        self.pair_selector = PairsSelector(self.data)

    def test_dimensionality_reduction(self):
        """
        Runs pca on price universe and verifies std dev for all components.
        """
        self.pair_selector.dimensionality_reduction_by_components(5)

        expected_stds = pd.Series(
            [0.008302, 0.045064, 0.045148, 0.045169, 0.045146])
        actual_stds = self.pair_selector.feature_vector.std()
        pd.testing.assert_series_equal(
            expected_stds, actual_stds, check_less_precise=True)

        with self.assertRaises(Exception):
            pair_selector = PairsSelector(None)
            pair_selector.dimensionality_reduction_by_components(15)

    def test_clustering(self):
        """
        Verifies generated clusters from both techniques in the PairsSelector class.
        """
        with self.assertRaises(Exception):
            pair_selector = PairsSelector(None)
            pair_selector.cluster_using_optics({})

        with self.assertRaises(Exception):
            pair_selector = PairsSelector(None)
            pair_selector.cluster_using_dbscan({})

        self.pair_selector.dimensionality_reduction_by_components(5)

        self.pair_selector.cluster_using_optics({'min_samples': 3})
        self.assertEqual(len(np.unique(self.pair_selector.clust_labels_)), 47)

        self.pair_selector.cluster_using_dbscan(
            {'eps': 0.03, 'min_samples': 3})
        self.assertEqual(len(np.unique(self.pair_selector.clust_labels_)), 11)

    def test_generate_pairwise_combinations(self):
        """
        Verifies pairs generator in the PairsSelector class.
        """

        self.pair_selector.dimensionality_reduction_by_components(5)

        self.pair_selector.cluster_using_optics({'min_samples': 3})

        c_labels = np.unique(self.pair_selector.clust_labels_[ \
                             self.pair_selector.clust_labels_ != -1])

        self.assertEqual(len(c_labels), 46)

        pair_list = self.pair_selector._generate_pairwise_combinations(
            c_labels)

        self.assertEqual(len(pair_list), 631)

        with self.assertRaises(Exception):
            self.pair_selector._generate_pairwise_combinations([])

    def test_hurst_criterion(self):
        """
        Verifies private hurst processing method.
        """
        self.pair_selector.dimensionality_reduction_by_components(5)
        self.pair_selector.cluster_using_optics({'min_samples': 3})

        hedge_ratios = [0.832406370860649, 0.892407527838706, 0.461344]
        idx = [('AJG', 'ICE'), ('AJG', 'MMC'), ('AIG', 'LEG')]
        input_pairs = pd.DataFrame(data=hedge_ratios, columns=[ \
                                   'hedge_ratio'], index=idx)

        result = self.pair_selector._hurst_criterion(input_pairs)

        pd.testing.assert_series_equal(
            pd.Series(idx[0:2]), pd.Series(result[1].index))

        with self.assertRaises(Exception):
            self.pair_selector._hurst_criterion([])

    def test_final_criterions(self):
        """
        Verifies private final criterions processing method.
        """
        self.pair_selector.dimensionality_reduction_by_components(5)
        self.pair_selector.cluster_using_optics({'min_samples': 3})

        hedge_ratios = [0.832406370860649, 0.892407527838706]
        idx = [('AJG', 'ICE'), ('AJG', 'MMC')]
        input_pairs = pd.DataFrame(data=hedge_ratios, columns=[ \
                                   'hedge_ratio'], index=idx)
        spreads_df, hurst_pass_pairs = self.pair_selector._hurst_criterion(
            input_pairs)

        hl_pairs, final_pairs = self.pair_selector._final_criterions(
            spreads_df, hurst_pass_pairs.index.values)
        pd.testing.assert_series_equal(
            pd.Series(idx), pd.Series(hl_pairs.index))
        pd.testing.assert_series_equal(
            pd.Series(idx), pd.Series(final_pairs.index))

        with self.assertRaises(Exception):
            self.pair_selector._final_criterions([], [])

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

        pd.testing.assert_series_equal(pd.Series(final_pairs), pd.Series(
            self.pair_selector.coint_pass_pairs.index))
        pd.testing.assert_series_equal(
            pd.Series(final_pairs), pd.Series(result))

    def test_description_methods(self):
        """
        Tests the various pair description methods.
        """
        self.assertEqual(type(self.pair_selector.describe()), pd.DataFrame)
        self.assertEqual(
            type(self.pair_selector.describe_extra()), pd.DataFrame)

        sectoral_description = self.pair_selector.describe_pairs_sectoral_info(['AJG'], ['ICE'], \
                                                        pd.DataFrame(columns=['ticker', 'sector', 'industry']))
        self.assertEqual(type(sectoral_description), pd.DataFrame)

        sector_info = pd.DataFrame(data=[
            ('AJG', 'sector', 'industry'),
            ('ICE', 'sector', 'industry')
        ])
        sector_info.columns = ['ticker', 'sector', 'industry']
        self.assertEqual(type(self.pair_selector.describe_pairs_sectoral_info(
            ['AJG'], ['ICE'], sector_info)), pd.DataFrame)

    def test_manual_methods(self):
        """
        Tests the pair generator that uses user inputted category based clusters.
        """
        sector_info = pd.DataFrame(data=[
            ('AJG', 'sector'),
            ('ICE', 'sector'),
            ('MMC', '')
        ])
        sector_info.columns = ['ticker', 'sector']

        self.assertEqual(
            len(self.pair_selector.get_pairs_by_sector(sector_info)), 1)
