# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests utility dataset split functions:
util/split_dataset.py
"""

import unittest
import os

import numpy as np
import pandas as pd

from arbitragelab.util.split_dataset import train_test_split


class TestSplitDataset(unittest.TestCase):
    """
    Test Split Dataset file.
    """

    def setUp(self):
        """
        Set up the data and parameters.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/ANZ-ADB.csv'
        self.data = pd.read_csv(data_path, parse_dates=['Date'])
        self.data.set_index("Date", inplace=True)

    def test_split_dataset(self):
        """
        Unit tests for cointegration coefficient calculation.
        """

        # Cutoff by date
        train_date, test_date = train_test_split(self.data, date_cutoff=pd.Timestamp(2002, 1, 1))

        # Cutoff by number
        with self.assertWarns(Warning):
            # Expected warning here that date cutoff input is not used
            train_number, test_number = train_test_split(self.data, date_cutoff=None, num_cutoff=253)

        # No cutoff, should result in same dataset being returned twice
        train_same, test_same = train_test_split(self.data, date_cutoff=None)

        # Test output dataframe shapes
        self.assertTupleEqual(train_date.shape, (253, 2))
        self.assertTupleEqual(test_date.shape, (168, 2))

        # Test outputs are the same
        pd.testing.assert_frame_equal(train_date, train_number)
        pd.testing.assert_frame_equal(test_date, test_number)

        # Test no cutoff returns same dataframe
        pd.testing.assert_frame_equal(test_same, self.data)
        pd.testing.assert_frame_equal(train_same, self.data)

    def test_split_dataset_errors(self):
        """
        Unit tests for cointegration coefficient calculation.
        """

        # Test for warning when the Index is not of type pd.DatetimeIndex
        bad_data = self.data.copy()
        bad_data.index = np.zeros(len(bad_data.index))

        self.assertRaises(AssertionError, train_test_split, bad_data, pd.Timestamp(2002, 1, 1))

        # Test for warning when the date cutoff point is out of range
        self.assertRaises(AssertionError, train_test_split, self.data, pd.Timestamp(2021, 1, 1))
