# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests function of ML Pairs Selection module:
ml_approach/pairs_selector.py
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
        data_path = project_path + '/test_data/sp100_prices.csv'
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

        # Check pca reduced dataset components standard deviations.
        pd.testing.assert_series_equal(expected_stds, actual_stds, check_less_precise=True)

        # Test if plotting function returns axes objects.
        pca_plot_obj = self.pair_selector.plot_pca_matrix()
        self.assertTrue(type(pca_plot_obj), list)

        # Test clustering plot before generating any clusters.
        with self.assertRaises(Exception):
            self.pair_selector.plot_clustering_info()

        # Test dimensionality reduction when inputting invalid data.
        with self.assertRaises(Exception):
            pair_selector = PairsSelector(None)
            pair_selector.dimensionality_reduction_by_components(15)

    def test_clustering(self):
        """
        Verifies generated clusters from both techniques in the PairsSelector class.
        """

        # Test Optics clustering without any price data.
        with self.assertRaises(Exception):
            pair_selector = PairsSelector(None)
            pair_selector.cluster_using_optics()

        # Test Dbscan clustering without any price data.
        with self.assertRaises(Exception):
            pair_selector = PairsSelector(None)
            pair_selector.cluster_using_dbscan()

        self.pair_selector.dimensionality_reduction_by_components(2)

        # Test number of generate clusters from the methods offered.

        self.pair_selector.cluster_using_optics(min_samples=3)
        self.assertEqual(len(np.unique(self.pair_selector.clust_labels_)), 16)

        self.pair_selector.cluster_using_dbscan(eps=0.03, min_samples=3)
        self.assertEqual(len(np.unique(self.pair_selector.clust_labels_)), 1)

    def test_generate_pairwise_combinations(self):
        """
        Verifies pairs generator in the PairsSelector class.
        """

        # Setup initial variables needed for the test.
        self.pair_selector.dimensionality_reduction_by_components(2)
        self.pair_selector.cluster_using_optics(min_samples=3)

        init_labels = self.pair_selector.clust_labels_
        c_labels = np.unique(init_labels[init_labels != -1])

        # Check number of unique cluster generated.
        self.assertEqual(len(c_labels), 15)

        # Check number of pairwise combinations using cluster data.
        pair_list = self.pair_selector._generate_pairwise_combinations(c_labels)
        self.assertEqual(len(pair_list), 158)

        # Try to generate combinations without valid input data.
        with self.assertRaises(Exception):
            self.pair_selector._generate_pairwise_combinations([])

    def test_hurst_criterion(self):
        """
        Verifies private hurst processing method.
        """

        # Setup initial variables needed for the test.
        self.pair_selector.dimensionality_reduction_by_components(2)
        self.pair_selector.cluster_using_optics(min_samples=3)

        # Setup needed information to validate the hurst criterion return.
        hedge_ratios = [0.832406370860649, 70]
        idx = [('A', 'AVB'), ('ABMD', 'AZO')]
        input_pairs = pd.DataFrame(data=hedge_ratios, index=idx)
        input_pairs.columns = ['hedge_ratio']

        result = self.pair_selector._hurst_criterion(input_pairs)
        hurst_pp = pd.Series(result[1].index)
        pd.testing.assert_series_equal(pd.Series([idx[0]]), hurst_pp)

        # Test the hurst criterion with invalid input data.
        with self.assertRaises(Exception):
            self.pair_selector._hurst_criterion([])

    def test_final_criterions(self):
        """
        Verifies private final criterions processing method.
        """

        # Setup initial variables needed for the test.
        self.pair_selector.dimensionality_reduction_by_components(2)
        self.pair_selector.cluster_using_optics(min_samples=3)

        hedge_ratios = [0.832406370860649, 70]
        idx = [('A', 'AVB'), ('ABMD', 'AZO')]
        input_pairs = pd.DataFrame(data=hedge_ratios, index=idx)
        input_pairs.columns = ['hedge_ratio']

        # Generate the inputs needed for the final criterions method test.
        spreads_df, hurst_pass_pairs = self.pair_selector._hurst_criterion(input_pairs)

        hl_pairs, final_pairs = self.pair_selector._final_criterions(
            spreads_df, hurst_pass_pairs.index.values
        )

        hl_pairs_sr = pd.Series(hl_pairs.index)
        final_pairs_sr = pd.Series(final_pairs.index)

        # Check that the first pair passes the Half Life Test.
        pd.testing.assert_series_equal(pd.Series([idx[0]]), hl_pairs_sr)

        # Check that no pairs pass through to the final list.
        pd.testing.assert_series_equal(pd.Series([], dtype=object), final_pairs_sr)

        # Test final criterions method using invalid data.
        with self.assertRaises(Exception):
            self.pair_selector._final_criterions([], [])

    def test_criterion_selector(self):
        """
        Verifies final user exposed criterion selection method.
        """

        # Setup initial variables needed for the test.
        self.pair_selector.dimensionality_reduction_by_components(2)
        self.pair_selector.cluster_using_optics(min_samples=3)

        final_pairs = [('BA', 'CF')]
        coint_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        input_pairs = final_pairs + coint_pairs

        result = self.pair_selector._criterion_selection(input_pairs, adf_cutoff_threshold=0.9)
        result = pd.Series(result)

        coint_pp = self.pair_selector.coint_pass_pairs.index
        coint_pp = pd.Series(coint_pp)

        # Assert that only the first pair passes through all the tests.
        pd.testing.assert_series_equal(pd.Series(final_pairs), coint_pp)
        pd.testing.assert_series_equal(pd.Series(final_pairs), result)

    def test_unsupervised_candidate_pair_selector(self):
        """
        Tests the parent candidate pair selection method.
        """

        # Tests pair selector with invalid data seed clustering data.
        with self.assertRaises(Exception):
            self.pair_selector.clust_labels_ = []
            self.pair_selector.unsupervised_candidate_pair_selector()

        # Setup initial variables needed for the test.
        self.pair_selector.dimensionality_reduction_by_components(2)

        # The following will mark a few tickers as valid to be used
        # in the pairs generation process.
        self.pair_selector.clust_labels_ = np.array([-1] * 100)
        np.put(self.pair_selector.clust_labels_, [55, 56, 86], 1)

        self.assertTrue(type(self.pair_selector.unsupervised_candidate_pair_selector()), list)

        final_pairs = pd.DataFrame(index=[('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')])
        self.pair_selector.final_pairs = final_pairs
        selected_pairs_return = self.pair_selector.plot_selected_pairs()

        # Check if returned plot object is a list of Axes objects.
        self.assertTrue(type(selected_pairs_return), list)

        with self.assertRaises(Exception):
            pairs_list = list((('F', 'V'),)*45)
            final_pairs = pd.DataFrame(index=pairs_list)
            self.pair_selector.final_pairs = final_pairs
            self.pair_selector.plot_selected_pairs()

    def test_description_methods(self):
        """
        Tests the various pair description methods.
        """

        # Test return of the describe method.
        intro_descr = self.pair_selector.describe()
        self.assertEqual(type(intro_descr), pd.DataFrame)

        # Test return of the extended describe method.
        extended_descr = self.pair_selector.describe_extra()
        self.assertEqual(type(extended_descr), pd.DataFrame)

        # Test return of the sectoral based method with empty sector info dataframe input.
        empty_sectoral_df = pd.DataFrame(columns=['ticker', 'sector', 'industry'])
        empty_sectoral_descr = self.pair_selector.describe_pairs_sectoral_info(['AJG'], ['ICE'], empty_sectoral_df)
        self.assertEqual(type(empty_sectoral_descr), pd.DataFrame)

        sector_info = pd.DataFrame(data=[
            ('AJG', 'sector', 'industry'),
            ('ICE', 'sector', 'industry')
        ])
        sector_info.columns = ['ticker', 'sector', 'industry']
        full_sectoral_descr = self.pair_selector.describe_pairs_sectoral_info(['AJG'], ['ICE'], sector_info)

        # Test return of the sectoral based method with full sector info dataframe input.
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

        # Test pair generator with argument based category information.
        pairs_df = self.pair_selector.get_pairs_by_sector(sector_info)
        self.assertEqual(len(pairs_df), 1)

    def test_plotting_methods(self):
        """
        Tests all plotting methods.
        """

        # Test the final pairs plotting method with no information.
        with self.assertRaises(Exception):
            self.pair_selector.plot_selected_pairs()

        # Test the clustering plotting method with no information.
        with self.assertRaises(Exception):
            self.pair_selector.plot_clustering_info()

        # Setup initial variables needed for the test.
        self.pair_selector.dimensionality_reduction_by_components(1)
        self.pair_selector.cluster_using_optics(min_samples=3)

        # Test knee plot return object.
        knee_plot_pyplot_obj = self.pair_selector.plot_knee_plot()
        self.assertTrue(issubclass(type(knee_plot_pyplot_obj), matplotlib.axes.SubplotBase))

        # Test single pair plot return object.
        singlepair_pyplot_obj = self.pair_selector.plot_single_pair(('AJG', 'ABMD'))
        self.assertTrue(issubclass(type(singlepair_pyplot_obj), matplotlib.axes.SubplotBase))

        # Test 2d cluster plot return object.
        twod_pyplot_obj = self.pair_selector.plot_clustering_info(n_dimensions=2)
        self.assertTrue(issubclass(type(twod_pyplot_obj), matplotlib.axes.SubplotBase))

        # Test 3d cluster plot return object.
        threed_pyplot_obj = self.pair_selector.plot_clustering_info(n_dimensions=3)
        self.assertTrue(issubclass(type(threed_pyplot_obj), matplotlib.axes.SubplotBase))

        # Test the clustering plotting method with an oversized dimension number.
        with self.assertRaises(Exception):
            self.pair_selector.plot_clustering_info(n_dimensions=10)
