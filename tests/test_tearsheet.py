import unittest
import os
import pandas as pd
import numpy as np
from arbitragelab.arblab_tearsheet.tearsheet import TearSheet

class TestTearSheet(unittest.TestCase):
    """
    Test the TearSheet module
    """
    def setUp(self):
        """
        Set the file path for the data and testing variables.
        """

        np.random.seed(0)

        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/gld_gdx_data.csv'
        data = pd.read_csv(self.path)
        data = data.set_index('Date')

        self.dataframe = data[['GLD', 'GDX']]

    def test_server(self):
        """
        Tests if the server for the web app runs properly
        """
        test = TearSheet()

        test.cointegration_tearsheet(self.dataframe)
        test.ou_tearsheet(self.dataframe)


