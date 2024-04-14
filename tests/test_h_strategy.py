"""
Test functions from the H-strategy module.
"""

import unittest
import os
import numpy as np
import pandas as pd

from arbitragelab.time_series_approach.h_strategy import HConstruction, HSelection


class TestHConstruction(unittest.TestCase):
    """
    Tests the H-Construction module
    """

    def setUp(self):
        """
        Set the file path for the data and testing variables.
        """

        # Determining data path
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/gld_gdx_data.csv'

        # Getting data
        data = pd.read_csv(self.path)
        data = data.set_index('Date')
        self.dataframe = data[['GLD', 'GDX']]
        self.series = np.log(data['GLD']) - np.log(data['GDX'])
        self.thresholds = self.series.std()

        # Listing testing values
        self.test_construction_types = ["Kagi", "Renko", "Error"]
        self.test_signal_types = ["contrarian", "momentum", "Error"]

    def test_construct(self):
        """
        Tests functions from H-construction module.
        """

        # Testing different kinds of signal methods
        test = HConstruction(self.series, self.thresholds, self.test_construction_types[0])

        signals = test.get_signals(self.test_signal_types[0])
        self.assertEqual(signals.sum(), 1.0)

        signals = test.get_signals(self.test_signal_types[1])
        self.assertEqual(signals.sum(), -1.0)

        # Testing functions for H-statistics
        test = HConstruction(self.series, self.thresholds, self.test_construction_types[0])
        self.assertEqual(test.h_inversion(), 7)
        self.assertAlmostEqual(test.h_distances(p=1), 0.6463, places=4)
        self.assertAlmostEqual(test.h_volatility(p=1), 0.09233, places=5)

        test = HConstruction(self.series, self.thresholds, self.test_construction_types[1])
        self.assertEqual(test.h_inversion(), 3)
        self.assertAlmostEqual(test.h_distances(p=2), 0.0377857, places=7)
        self.assertAlmostEqual(test.h_volatility(p=2), 0.012595, places=6)

        # Testing function for extending
        half_length = len(self.series) // 2
        original_length = len(self.series[:half_length])
        extend_length = len(self.series[half_length:])

        test = HConstruction(self.series[:half_length], self.thresholds, self.test_construction_types[0])
        test.extend_series(self.series[half_length:])
        self.assertEqual(len(test.series), original_length + extend_length)

    def test_exeptions(self):
        """
        Tests the exceptions.
        """

        # Testing error construction method
        with self.assertRaises(Exception):
            test = HConstruction(self.series, self.thresholds, self.test_construction_types[2])

        # Testing error signal method
        with self.assertRaises(Exception):
            test = HConstruction(self.series, self.thresholds, self.test_construction_types[0])
            test.get_signals(self.test_signal_types[2])

class TestHSelection(unittest.TestCase):
    """
    Tests the H-Selection module
    """

    def setUp(self):
        """
        Set the file path for the data and testing variables.
        """

        # Determining data path
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/sp100_prices.csv'

        # Getting data
        data = pd.read_csv(self.path)
        data = data.set_index('Date')
        self.dataframe = data[data.columns[:20]]
        self.dataframe_break = self.dataframe.copy()
        self.dataframe_break.iloc[0][0] = np.nan

        # Listing testing values
        self.test_getting_types = ["highest", "lowest", "Error"]

    def test_select(self):
        """
        Tests functions from H-Selection module.
        """

        # Testing different kinds of getting methods
        test = HSelection(self.dataframe)
        test.select()

        pairs = test.get_pairs(5, self.test_getting_types[0], True)
        self.assertEqual(pairs[0][0], 43)
        self.assertEqual(pairs[1][0], 36)
        self.assertEqual(pairs[2][0], 29)
        self.assertEqual(pairs[3][0], 29)
        self.assertEqual(pairs[4][0], 29)

        pairs = test.get_pairs(5, self.test_getting_types[0], False)
        self.assertEqual(pairs[0][0], 43)
        self.assertEqual(pairs[1][0], 29)
        self.assertEqual(pairs[2][0], 22)
        self.assertEqual(pairs[3][0], 19)
        self.assertEqual(pairs[4][0], 19)

        pairs = test.get_pairs(5, self.test_getting_types[1], True)
        self.assertEqual(pairs[0][0], 1)
        self.assertEqual(pairs[1][0], 1)
        self.assertEqual(pairs[2][0], 1)
        self.assertEqual(pairs[3][0], 1)
        self.assertEqual(pairs[4][0], 1)

        pairs = test.get_pairs(5, self.test_getting_types[1], False)
        self.assertEqual(pairs[0][0], 1)
        self.assertEqual(pairs[1][0], 1)
        self.assertEqual(pairs[2][0], 1)
        self.assertEqual(pairs[3][0], 1)
        self.assertEqual(pairs[4][0], 2)

        # Testing minimum_length
        test = HSelection(self.dataframe_break)

        test.select(minimum_length = len(self.dataframe_break))
        self.assertEqual(len(test.results), 171)

        test.select(minimum_length = len(self.dataframe_break) - 1)
        self.assertEqual(len(test.results), 190)

    def test_exeptions(self):
        """
        Tests the exceptions.
        """

        # Testing error getting method
        test = HSelection(self.dataframe)
        test.select()

        with self.assertRaises(Exception):
            test.get_pairs(5, self.test_getting_types[2], True)

        with self.assertRaises(Exception):
            test.get_pairs(5, self.test_getting_types[2], False)
