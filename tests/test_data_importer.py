# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests functionality of Data Importer:
util/data_importer.py
"""

import os
import unittest
import pandas as pd
import numpy as np
from arbitragelab.util import DataImporter

class TestDataImporter(unittest.TestCase):
    """
    Tests Data Importer class.
    """
    # pylint: disable=pointless-string-statement

    def setUp(self):
        """
        Loads price universe.
        """

        np.random.seed(0)
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/sp100_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.data.dropna(inplace=True)

    #These tests are breaking due to yahoo_fin issue now: https://github.com/atreadw1492/yahoo_fin/issues/36
    '''
    def test_ticker_collectors(self):
        """
        Tests ticker collection collectors.
        """

        self.assertTrue(len(DataImporter.get_sp500_tickers()) > 400)
        self.assertTrue(len(DataImporter.get_dow_tickers()) > 20)
    '''

    def test_preprocessing_methods(self):
        """
        Tests preprocessing methods.
        """

        # Generate a valid DataFrame full of ones.
        sample_df = pd.DataFrame(data=np.ones((200, 20)))
        # Append a column with contents of nan values.
        sample_df[20] = np.nan

        # Assert that the remove_nuns method removes the last column.
        self.assertEqual(len(DataImporter.remove_nuns(sample_df).columns), 20)

        # Get the first column from the dataset, process to returns and check
        # its mean.
        returns_mean = DataImporter.get_returns_data(self.data.iloc[:, 1]).mean()

        self.assertAlmostEqual(returns_mean, 0, places=1)

    def test_price_retriever(self):
        """
        Tests asset prices retriever.
        """

        # Download a year worth of GOOG daily price data and check it's length.
        price_df = DataImporter.get_price_data('GOOG', '2015-01-01', '2016-01-01', '1d')
        self.assertTrue(len(price_df) > 200)

    @staticmethod
    def test_ticker_sector_info():
        """
        Tests ticker information augmentor.
        """

        data_importer = DataImporter()

        # Make the expected sector info DataFrame, that has the basic structure of columns
        # as follows: [ticker_symbol, industry, sector].
        expected_result = pd.DataFrame(data=[
            ('GOOG', 'Internet Content & Information', 'Communication Services'),
            ('FB', 'Internet Content & Information', 'Communication Services')
        ])
        expected_result.columns = ['ticker', 'industry', 'sector']

        # Call the get_ticker_sector_info method to request the necessary data.
        augmented_ticker_df = data_importer.get_ticker_sector_info(['GOOG', 'FB'], 1)
        pd.testing.assert_frame_equal(augmented_ticker_df, expected_result)
