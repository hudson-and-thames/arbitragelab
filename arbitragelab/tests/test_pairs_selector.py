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
import matplotlib

from arbitragelab.ml_approach import PairsSelector

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
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.data.dropna(inplace=True)
        self.pair_selector = PairsSelector(self.data)

    def test_dimensionality_reduction(self):
        """
        Runs pca on price universe and verifies std dev for all components.
        """

        self.pair_selector.dimensionality_reduction_by_components(5)

        expected_stds = pd.Series([0.0181832, 0.10025142, 0.10044381, 0.100503, 0.10048])
        actual_stds = self.pair_selector.feature_vector.std()
        pd.testing.assert_series_equal(expected_stds, actual_stds, check_less_precise=True)

        self.pair_selector.plot_pca_matrix()

        with self.assertRaises(Exception):
            self.pair_selector.plot_clustering_info(show=False)

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

        self.pair_selector.dimensionality_reduction_by_components(2)

        self.pair_selector.cluster_using_optics({'min_samples': 3})
        self.assertEqual(len(np.unique(self.pair_selector.clust_labels_)), 16)

        self.pair_selector.cluster_using_dbscan({'eps': 0.03, 'min_samples': 3})
        self.assertEqual(len(np.unique(self.pair_selector.clust_labels_)), 1)

    def test_generate_pairwise_combinations(self):
        """
        Verifies pairs generator in the PairsSelector class.
        """

        self.pair_selector.dimensionality_reduction_by_components(2)

        self.pair_selector.cluster_using_optics({'min_samples': 3})

        init_labels = self.pair_selector.clust_labels_
        c_labels = np.unique(init_labels[init_labels != -1])

        self.assertEqual(len(c_labels), 15)

        pair_list = self.pair_selector._generate_pairwise_combinations(c_labels)

        self.assertEqual(len(pair_list), 158)

        with self.assertRaises(Exception):
            self.pair_selector._generate_pairwise_combinations([])

    def test_hurst_criterion(self):
        """
        Verifies private hurst processing method.
        """

        self.pair_selector.dimensionality_reduction_by_components(2)
        self.pair_selector.cluster_using_optics({'min_samples': 3})

        hedge_ratios = [0.832406370860649, 70]
        idx = [('A', 'AVB'), ('ABMD', 'AZO')]
        input_pairs = pd.DataFrame(data=hedge_ratios, index=idx)
        input_pairs.columns = ['hedge_ratio']

        result = self.pair_selector._hurst_criterion(input_pairs)
        hurst_pp = pd.Series(result[1].index)
        pd.testing.assert_series_equal(pd.Series([idx[0]]), hurst_pp)

        with self.assertRaises(Exception):
            self.pair_selector._hurst_criterion([])

    def test_final_criterions(self):
        """
        Verifies private final criterions processing method.
        """

        self.pair_selector.dimensionality_reduction_by_components(2)
        self.pair_selector.cluster_using_optics({'min_samples': 3})

        hedge_ratios = [0.832406370860649, 70]
        idx = [('A', 'AVB'), ('ABMD', 'AZO')]
        input_pairs = pd.DataFrame(data=hedge_ratios, index=idx)
        input_pairs.columns = ['hedge_ratio']

        spreads_df, hurst_pass_pairs = self.pair_selector._hurst_criterion(input_pairs)

        hl_pairs, final_pairs = self.pair_selector._final_criterions(
            spreads_df, hurst_pass_pairs.index.values
        )

        hl_pairs_sr = pd.Series(hl_pairs.index)
        final_pairs_sr = pd.Series(final_pairs.index)

        pd.testing.assert_series_equal(pd.Series([idx[0]]), hl_pairs_sr)

        pd.testing.assert_series_equal(pd.Series([], dtype=object), final_pairs_sr)

        with self.assertRaises(Exception):
            self.pair_selector._final_criterions([], [])

    def test_criterion_selector(self):
        """
        Verifies final user exposed criterion selection method.
        """

        self.pair_selector.dimensionality_reduction_by_components(2)
        self.pair_selector.cluster_using_optics({'min_samples': 3})

        final_pairs = [('BA', 'CF')]
        coint_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        input_pairs = final_pairs + coint_pairs

        result = self.pair_selector._criterion_selection(input_pairs, pvalue_threshold=0.1)
        result = pd.Series(result)

        coint_pp = self.pair_selector.coint_pass_pairs.index
        coint_pp = pd.Series(coint_pp)

        pd.testing.assert_series_equal(pd.Series(final_pairs), coint_pp)
        pd.testing.assert_series_equal(pd.Series(final_pairs), result)

    def test_unsupervised_candidate_pair_selector(self):
        """
        Tests the parent candidate pair selection method.
        """

        with self.assertRaises(Exception):
            self.pair_selector.clust_labels_ = []
            self.pair_selector.unsupervised_candidate_pair_selector()

        self.pair_selector.dimensionality_reduction_by_components(2)
        self.pair_selector.clust_labels_ = np.array([
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        ])
        self.pair_selector.unsupervised_candidate_pair_selector()

        final_pairs = pd.DataFrame(index=[('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')])

        self.pair_selector.final_pairs = final_pairs
        self.pair_selector.plot_selected_pairs()

    def test_description_methods(self):
        """
        Tests the various pair description methods.
        """

        intro_descr = self.pair_selector.describe()
        self.assertEqual(type(intro_descr), pd.DataFrame)

        extended_descr = self.pair_selector.describe_extra()
        self.assertEqual(type(extended_descr), pd.DataFrame)

        empty_sectoral_df = pd.DataFrame(columns=['ticker', 'sector', 'industry'])
        empty_sectoral_descr = self.pair_selector.describe_pairs_sectoral_info(['AJG'], ['ICE'], empty_sectoral_df)
        self.assertEqual(type(empty_sectoral_descr), pd.DataFrame)

        sector_info = pd.DataFrame(data=[
            ('AJG', 'sector', 'industry'),
            ('ICE', 'sector', 'industry')
        ])
        sector_info.columns = ['ticker', 'sector', 'industry']
        full_sectoral_descr = self.pair_selector.describe_pairs_sectoral_info(['AJG'], ['ICE'], sector_info)
        self.assertEqual(type(full_sectoral_descr), pd.DataFrame)

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

        pairs_df = self.pair_selector.get_pairs_by_sector(sector_info)
        self.assertEqual(len(pairs_df), 1)

    def test_plotting_methods(self):
        """
        Tests all plotting methods.
        """

        with self.assertRaises(Exception):
            self.pair_selector.plot_selected_pairs()

        with self.assertRaises(Exception):
            self.pair_selector.plot_clustering_info(show=False)

        self.pair_selector.dimensionality_reduction_by_components(1)
        self.pair_selector.cluster_using_optics({'min_samples': 3})

        knee_plot_pyplot_obj = self.pair_selector.plot_knee_plot()
        self.assertTrue(issubclass(type(knee_plot_pyplot_obj), matplotlib.axes.SubplotBase))

        twod_pyplot_obj = self.pair_selector.plot_clustering_info(n_dimensions=2, show=False)
        self.assertTrue(issubclass(type(twod_pyplot_obj), matplotlib.axes.SubplotBase))

        threed_pyplot_obj = self.pair_selector.plot_clustering_info(n_dimensions=3, show=False)
        self.assertTrue(issubclass(type(threed_pyplot_obj), matplotlib.axes.SubplotBase))

        with self.assertRaises(Exception):
            self.pair_selector.plot_clustering_info(n_dimensions=10, show=False)
