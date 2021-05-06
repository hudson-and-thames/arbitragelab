# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Test functions for the Optimal Convergence models in the Stochastic Control Approach module.
"""
# pylint: disable=protected-access

import warnings
import unittest
import os

import numpy as np
import pandas as pd

from arbitragelab.stochastic_control_approach.optimal_convergence import OptimalConvergence

class TestOptimalConvergence(unittest.TestCase):
    """
    Test Optimal Convergence model in Stochastic Control Approach module
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up data and parameters
        """

        np.random.seed(0)

        project_path = os.path.dirname(__file__)

        # Setting up the first dataset
        path = project_path + '/test_data/gld_gdx_data.csv'
        data = pd.read_csv(path)
        data = data.set_index('Date')
        cls.dataframe = data[['GLD', 'GDX']]

        # Setting up the second dataset
        path = project_path + '/test_data/shell-rdp-close_USD.csv'
        data = pd.read_csv(path, index_col='Date').ffill()
        data.index = pd.to_datetime(data.index, format="%d/%m/%Y")
        cls.shell_rdp_data = data


    def test_fit(self):
        """
        Tests the fit method in the class.
        """

        # Creating an object of the class
        oc = OptimalConvergence()

        oc.fit(self.dataframe, mu_m=0.05, sigma_m=0.35, r=0.02)

        # Checking parameter values.
        self.assertAlmostEqual(oc.lambda_1, 0, delta=1e-4)
        self.assertAlmostEqual(oc.lambda_2, 0, delta=1e-4)
        self.assertAlmostEqual(oc.b_squared, 0, delta=1e-3)
        self.assertAlmostEqual(oc.sigma_squared, 0, delta=1e-4)
        self.assertAlmostEqual(oc.beta, 0, delta=1e-4)


    def test_describe(self):
        """
        Tests the describe method in the class.
        """

        # Creating an object of the class
        oc = OptimalConvergence()

        # Testing for the run fit before this method exception
        with self.assertRaises(Exception):
            oc.describe()

        oc.fit(self.dataframe, mu_m=0.05, sigma_m=0.35, r=0.02)

        index = ['Ticker of first stock', 'Ticker of second stock',
                 'lambda_1', 'lambda_2', 'b_squared', 'sigma_squared',
                 'beta']

        data = ['GLD', 'GDX', 0, 0, 0, 0, 0]

        # Testing the output of describe method
        pd.testing.assert_series_equal(pd.Series(index=index,data=data), oc.describe(), check_exact=False, atol=1e-3)


    def test_unconstrained_continuous(self):

        # Creating an object of the class
        oc = OptimalConvergence()

        # Testing for the run fit before this method exception
        with self.assertRaises(Exception):
            oc.unconstrained_portfolio_weights_continuous(self.dataframe, gamma=4)

        oc.fit(self.dataframe, mu_m=0.05, sigma_m=0.35, r=0.02)

        # Testing for the positive gamma exception
        with self.assertRaises(Exception):
            oc.unconstrained_portfolio_weights_continuous(self.dataframe, gamma=-4)

        phi_1, phi_2, phi_m = oc.unconstrained_portfolio_weights_continuous(self.dataframe, gamma=4)

        # Checking the values of phi_1 weights
        self.assertAlmostEqual(np.mean(phi_1), 0, delta=1e-5)
        self.assertAlmostEqual(phi_1[7],  0, delta=1e-4)
        self.assertAlmostEqual(phi_1[28], 0, delta=1e-4)
        self.assertAlmostEqual(phi_1[-1], 0, delta=1e-4)

        # Checking the values of phi_2 weights
        self.assertAlmostEqual(np.mean(phi_2), 0, delta=1e-5)
        self.assertAlmostEqual(phi_2[7],  0, delta=1e-4)
        self.assertAlmostEqual(phi_2[28], 0, delta=1e-4)
        self.assertAlmostEqual(phi_2[-1], 0, delta=1e-4)

        # Checking the values of phi_m weights
        self.assertAlmostEqual(np.mean(phi_m), 0, delta=1e-5)
        self.assertAlmostEqual(phi_m[7],  0, delta=1e-4)
        self.assertAlmostEqual(phi_m[28], 0, delta=1e-4)
        self.assertAlmostEqual(phi_m[-1], 0, delta=1e-4)


    def test_delta_neutral_continuous(self):

        # Creating an object of the class
        oc = OptimalConvergence()

        # Testing for the run fit before this method exception
        with self.assertRaises(Exception):
            oc.delta_neutral_portfolio_weights_continuous(self.dataframe, gamma=4)

        oc.fit(self.dataframe, mu_m=0.05, sigma_m=0.35, r=0.02)

        # Testing for the positive gamma exception
        with self.assertRaises(Exception):
            oc.delta_neutral_portfolio_weights_continuous(self.dataframe, gamma=-4)

        phi_1, phi_2, phi_m = oc.delta_neutral_portfolio_weights_continuous(self.dataframe, gamma=4)

        # Checking the values of phi_1 weights
        self.assertAlmostEqual(np.mean(phi_1), 0, delta=1e-5)
        self.assertAlmostEqual(phi_1[7], 0, delta=1e-4)
        self.assertAlmostEqual(phi_1[28], 0, delta=1e-4)
        self.assertAlmostEqual(phi_1[-1], 0, delta=1e-4)

        # Checking the values of phi_2 weights
        self.assertAlmostEqual(np.mean(phi_2), 0, delta=1e-5)
        self.assertAlmostEqual(phi_2[7], 0, delta=1e-4)
        self.assertAlmostEqual(phi_2[28], 0, delta=1e-4)
        self.assertAlmostEqual(phi_2[-1], 0, delta=1e-4)

        # Checking the values of phi_m weights
        self.assertAlmostEqual(np.mean(phi_m), 0, delta=1e-5)
        self.assertAlmostEqual(phi_m[7], 0, delta=1e-4)
        self.assertAlmostEqual(phi_m[28], 0, delta=1e-4)
        self.assertAlmostEqual(phi_m[-1], 0, delta=1e-4)


    def test_wealth_gain_continuous(self):

        # Creating an object of the class
        oc = OptimalConvergence()

        # Testing for the run fit before this method exception
        with self.assertRaises(Exception):
            oc.wealth_gain_continuous(self.dataframe, gamma=4)

        oc.fit(self.dataframe, mu_m=0.05, sigma_m=0.35, r=0.02)

        # Testing for the positive gamma exception
        with self.assertRaises(Exception):
            oc.wealth_gain_continuous(self.dataframe, gamma=-4)

        R = oc.wealth_gain_continuous(self.dataframe, gamma=4)

        # Checking the values of phi_1 weights
        self.assertAlmostEqual(np.mean(R), 0, delta=1e-5)
        self.assertAlmostEqual(R[7], 0, delta=1e-4)
        self.assertAlmostEqual(R[28], 0, delta=1e-4)
        self.assertAlmostEqual(R[-1], 0, delta=1e-4)
