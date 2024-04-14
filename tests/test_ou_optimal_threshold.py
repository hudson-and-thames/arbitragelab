"""
Tests functions from O-U Model Optimal Threshold module.
"""

# pylint: disable=protected-access
import unittest
import os
import numpy as np
import pandas as pd

from arbitragelab.time_series_approach.ou_optimal_threshold import OUModelOptimalThreshold


class TestOUModelOptimalThreshold(unittest.TestCase):
    """
    Tests the base class of O-U Model Optimal Threshold module.
    """

    def setUp(self):
        """
        Set the file path for the data and testing variables.
        """

        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/gld_gdx_data.csv'  # Data Path

        data = pd.read_csv(self.path)
        data = data.set_index('Date')
        self.dataframe = data[['GLD', 'GDX']]

        self.assets = np.array(self.dataframe)
        self.assets_incorrect = np.zeros((4, 3))  # Data with incorrect dimensions

        asset_trans = self.assets.transpose()
        self.spread_series = (asset_trans[0][:] - asset_trans[0][0]) - 0.2 * \
                             (asset_trans[1][:] - asset_trans[1][0])
        self.spread_series = np.exp(self.spread_series)

        # List with testing values for data frequency
        self.test_data_frequency = ["D", "M", "Y", "N"]

    def test_construct(self):
        """
        Tests functions for O-U process construction.
        """

        # Creating an object of class
        test = OUModelOptimalThreshold()

        # Testing normal usage
        test.construct_ou_model_from_given_parameters(0, 0, 0)
        self.assertEqual(test.theta, 0)
        self.assertEqual(test.mu, 0)
        self.assertEqual(test.sigma, 0)

        # Testing different types of data input
        test.fit_ou_model_to_data(self.dataframe, self.test_data_frequency[0])
        test.fit_ou_model_to_data(self.assets, self.test_data_frequency[0])
        test.fit_ou_model_to_data(self.spread_series, self.test_data_frequency[0])

        # Testing different types of data frequency
        test.fit_ou_model_to_data(self.dataframe, self.test_data_frequency[0])
        test.fit_ou_model_to_data(self.dataframe, self.test_data_frequency[1])
        test.fit_ou_model_to_data(self.dataframe, self.test_data_frequency[2])

    def test_exeptions(self):
        """
        Tests exceptions in the module.
        """

        # Creating an object of class
        test = OUModelOptimalThreshold()

        # Testing for wrong data dimensions
        with self.assertRaises(Exception):
            test.fit_ou_model_to_data(self.assets_incorrect, self.test_data_frequency[0])

        # Testing for wrong data frequency
        with self.assertRaises(Exception):
            test.fit_ou_model_to_data(self.dataframe, self.test_data_frequency[3])

    def test_numerical(self):
        """
        Tests functions for numerical calculation.
        """

        # Creating an object of class
        test = OUModelOptimalThreshold()

        # Testing whether the output value is correct
        self.assertAlmostEqual(test._w1(0), 0.0, places=1)
        self.assertAlmostEqual(test._w1(0.1), 0.00251386, places=5)
        self.assertAlmostEqual(test._w1(-0.1), -test._w1(0.1), places=5)
        self.assertAlmostEqual(test._w1(1), 3.566894, places=3)
        self.assertAlmostEqual(test._w1(-1), -test._w1(1), places=3)
        self.assertAlmostEqual(test._w1(100), 131.229, places=1)
        self.assertAlmostEqual(test._w1(-100), -test._w1(100), places=1)

        self.assertAlmostEqual(test._w2(0), 0.0, places=1)
        self.assertAlmostEqual(test._w2(0.1), -0.34718291, places=5)
        self.assertAlmostEqual(test._w2(-0.1), -test._w2(0.1), places=5)
        self.assertAlmostEqual(test._w2(1), -3.123865, places=3)
        self.assertAlmostEqual(test._w2(-1), -test._w2(1), places=3)
        self.assertAlmostEqual(test._w2(100), -11643.138, places=1)
        self.assertAlmostEqual(test._w2(-100), -test._w2(100), places=1)
