"""
Tests function of ML Pairs Selection module:
ml_approach/pairs_selector.py
"""
# pylint: disable=protected-access

import os
import unittest
import pandas as pd
import numpy as np

from matplotlib.axes import Axes

from arbitragelab.ml_approach import OPTICSDBSCANPairsClustering


class TestDBSCANClustering(unittest.TestCase):
    """
    Tests OPTICSDBSCANPairsClustering class.
    """

    def setUp(self):
        """
        Loads price universe and instantiates the OPTICSDBSCANPairsClustering selection class.
        """

        np.random.seed(0)

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/sp100_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.data.dropna(inplace=True)
        self.pair_selector = OPTICSDBSCANPairsClustering(self.data)

    def test_dimensionality_reduction(self):
        """
        Runs pca on price universe and verifies std dev for all components.
        """

        self.pair_selector.dimensionality_reduction_by_components(5)

        expected_stds = pd.Series([0.0181832, 0.10025142, 0.10044381, 0.100503, 0.10048])
        actual_stds = self.pair_selector.feature_vector.std()

        # Check pca reduced dataset components standard deviations.
        pd.testing.assert_series_equal(expected_stds, actual_stds, atol=0.05)

        # Test if plotting function returns axes objects.
        pca_plot_obj = self.pair_selector.plot_pca_matrix()
        self.assertTrue(type(pca_plot_obj), list)

        # Test clustering plot before generating any clusters.
        with self.assertRaises(Exception):
            self.pair_selector.plot_clustering_info()

        # Test dimensionality reduction when inputting invalid data.
        with self.assertRaises(Exception):
            pair_selector = OPTICSDBSCANPairsClustering(None)
            pair_selector.dimensionality_reduction_by_components(15)

    def test_clustering(self):
        """
        Verifies generated clusters from both techniques in the OPTICSDBSCANPairsClustering class.
        """

        # Test Optics clustering without any price data.
        with self.assertRaises(Exception):
            pair_selector = OPTICSDBSCANPairsClustering(None)
            pair_selector.cluster_using_optics()

        # Test Dbscan clustering without any price data.
        with self.assertRaises(Exception):
            pair_selector = OPTICSDBSCANPairsClustering(None)
            pair_selector.cluster_using_dbscan()

        self.pair_selector.dimensionality_reduction_by_components(2)

        # Test number of generate clusters from the methods offered.

        self.pair_selector.cluster_using_optics(min_samples=3)
        self.assertEqual(len(np.unique(self.pair_selector.clust_labels_)), 16)

        self.pair_selector.cluster_using_dbscan(eps=0.03, min_samples=3)
        self.assertEqual(len(np.unique(self.pair_selector.clust_labels_)), 1)

    def test_generate_pairwise_combinations(self):
        """
        Verifies pairs generator in the OPTICSDBSCANPairsClustering class.
        """

        # Setup initial variables needed for the test.
        self.pair_selector.dimensionality_reduction_by_components(2)
        self.pair_selector.cluster_using_optics(min_samples=3)

        init_labels = self.pair_selector.clust_labels_
        c_labels = np.unique(init_labels[init_labels != -1])

        # Check number of unique cluster generated.
        self.assertEqual(len(c_labels), 15)

        # Check number of pairwise combinations using cluster data.
        a = np.array(init_labels)
        print(len(a[a==-1]))
        print([len(a[a == i]) for i in c_labels])
        pair_list = self.pair_selector._generate_pairwise_combinations(c_labels)
        self.assertEqual(len(pair_list), 158)

        # Try to generate combinations without valid input data.
        with self.assertRaises(Exception):
            self.pair_selector._generate_pairwise_combinations([])

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
        # Test the clustering plotting method with no information.
        with self.assertRaises(Exception):
            self.pair_selector.plot_clustering_info()

        # Setup initial variables needed for the test.
        self.pair_selector.dimensionality_reduction_by_components(1)
        self.pair_selector.cluster_using_optics(min_samples=3)

        # Test knee plot return object.
        knee_plot_pyplot_obj = self.pair_selector.plot_knee_plot()
        self.assertTrue(isinstance(knee_plot_pyplot_obj, Axes))

        # Test 2d cluster plot return object.
        twod_pyplot_obj = self.pair_selector.plot_clustering_info(n_dimensions=2)
        self.assertTrue(isinstance(twod_pyplot_obj, Axes))

        # Test 3d cluster plot return object.
        threed_pyplot_obj = self.pair_selector.plot_clustering_info(n_dimensions=3)
        self.assertTrue(isinstance(threed_pyplot_obj, Axes))

        # Test the clustering plotting method with an oversized dimension number.
        with self.assertRaises(Exception):
            self.pair_selector.plot_clustering_info(n_dimensions=10)
