# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Test functions for the Mudchanatongsuk OU model in the Stochastic Control Approach module.
"""

import unittest
import os
import numpy as np
import pandas as pd

from arbitragelab.stochastic_control_approach.ou_model_mudchanatongsuk import OUModelMudchanatongsuk


class TestOUModelMudchanatongsuk(unittest.TestCase):
    """
    Tests the Mudchanatongsuk OU model in the Stochastic Control Approach module.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Setup data and params.
        """

        np.random.seed(0)

        project_path = os.path.dirname(__file__)

        path = project_path + '/test_data/gld_gdx_data.csv'
        data = pd.read_csv(path)
        data = data.set_index('Date')
        cls.dataframe = data[['GLD', 'GDX']]


    def test_fit(self):
        """
        Tests the fit method in the class.
        """

        # Creating an object of the class.
        sc_mudchana = OUModelMudchanatongsuk()

        sc_mudchana.fit(self.dataframe)

        # Checking parameter values for spread calculation.
        self.assertAlmostEqual(np.mean(sc_mudchana.spread), 2.0465361303, delta=1e-7)
        self.assertAlmostEqual(sc_mudchana.spread[7], 2.1073878043, delta=1e-7)
        self.assertAlmostEqual(sc_mudchana.spread[28], 2.0496029865, delta=1e-7)
        self.assertAlmostEqual(sc_mudchana.spread[-1], 2.0202245834, delta=1e-7)

        # Checking other parameter values.
        self.assertAlmostEqual(sc_mudchana.sigma, 0.503695, delta=1e-4)
        self.assertAlmostEqual(sc_mudchana.mu, 0.114877, delta=1e-4)
        self.assertAlmostEqual(sc_mudchana.k, 3.99205, delta=1e-3)
        self.assertAlmostEqual(sc_mudchana.theta, 1.98816, delta=1e-4)
        self.assertAlmostEqual(sc_mudchana.eta, 0.404292, delta=1e-4)
        self.assertAlmostEqual(sc_mudchana.rho, 0.96202, delta=1e-4)


    def test_describe(self):
        """
        Tests the describe method in the class.
        """

        # Creating an object of the class.
        sc_mudchana = OUModelMudchanatongsuk()

        # Testing for the run fit before this method exception.
        with self.assertRaises(Exception):
            sc_mudchana.describe()

        sc_mudchana.fit(self.dataframe)

        index = ['Ticker of first stock', 'Ticker of second stock',
                 'long-term mean of spread', 'rate of mean reversion of spread', 'standard deviation of spread', 'half-life of spread',
                 'Drift of stock B', 'standard deviation of stock B']

        data = ['GLD', 'GDX', 1.98816, 3.99205, 0.404292, 0.173632, 0.114877, 0.503695]

        # Testing the output of describe method.
        pd.testing.assert_series_equal(pd.Series(index=index,data=data), sc_mudchana.describe(), check_exact=False, atol=1e-3)


    def test_optimal_weights(self):
        """
        Tests the optimal portfolio weights method in the class.
        """

        # Creating an object of the class.
        sc_mudchana = OUModelMudchanatongsuk()

        # Testing for the run fit before this method exception.
        with self.assertRaises(Exception):
            sc_mudchana.optimal_portfolio_weights(self.dataframe, gamma = -10)

        sc_mudchana.fit(self.dataframe)

        # Testing for invalid value of gamma exception.
        with self.assertRaises(Exception):
            sc_mudchana.optimal_portfolio_weights(self.dataframe, gamma = 10)

        weights = sc_mudchana.optimal_portfolio_weights(self.dataframe, gamma = -10)
        # Checking the values of weights.
        self.assertAlmostEqual(np.mean(weights), 0.4986890920, delta=1e-5)
        self.assertAlmostEqual(weights[7], 0.5117099817, delta=1e-4)
        self.assertAlmostEqual(weights[28], 0.5246204647, delta=1e-4)
        self.assertAlmostEqual(weights[-1], 0.4043368460, delta=1e-4)
