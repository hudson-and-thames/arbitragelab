# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Test functions for the Optimal Convergence models in the Stochastic Control Approach module.
"""

import unittest
import os

import numpy as np
import pandas as pd

from arbitragelab.stochastic_control_approach.optimal_convergence import OptimalConvergence

class TestOptimalConvergence(unittest.TestCase):
    """
    Test Optimal Convergence model in Stochastic Control Approach module.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up data and parameters.
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
        sc_liu = OptimalConvergence()

        sc_liu.fit(self.dataframe, mu_m=0.05, sigma_m=0.35, r=0.02)

        # Checking parameter values.
        self.assertAlmostEqual(sc_liu.lambda_1, -0.009828, delta=1e-4)
        self.assertAlmostEqual(sc_liu.lambda_2, 0.116263, delta=1e-4)
        self.assertAlmostEqual(sc_liu.b_squared, 0.083477, delta=1e-3)
        self.assertAlmostEqual(sc_liu.sigma_squared, -1.034854, delta=1e-4)
        self.assertAlmostEqual(sc_liu.beta, -2.985708, delta=1e-4)


    def test_describe(self):
        """
        Tests the describe method in the class.
        """

        # Creating an object of the class
        sc_liu = OptimalConvergence()

        # Testing for the run fit before this method exception
        with self.assertRaises(Exception):
            sc_liu.describe()

        sc_liu.fit(self.dataframe, mu_m=0.05, sigma_m=0.35, r=0.02)

        index = ['Ticker of first stock', 'Ticker of second stock',
                 'lambda_1', 'lambda_2', 'b_squared', 'sigma_squared',
                 'beta']

        data = ['GLD', 'GDX', -0.009828, 0.116263, 0.083477, -1.034854, -2.985708]

        # Testing the output of describe method
        pd.testing.assert_series_equal(pd.Series(index=index, data=data), sc_liu.describe(), check_exact=False, atol=1e-3)


    def test_unconstrained_continuous(self):
        """
        Tests the method which returns optimal portfolio weights in continuous case.
        """

        # Creating an object of the class
        sc_liu = OptimalConvergence()

        # Testing for the run fit before this method exception
        with self.assertRaises(Exception):
            sc_liu.unconstrained_portfolio_weights_continuous(self.dataframe, gamma=4)

        sc_liu.fit(self.dataframe, mu_m=0.05, sigma_m=0.35, r=0.02)

        # Testing for the positive gamma exception
        with self.assertRaises(Exception):
            sc_liu.unconstrained_portfolio_weights_continuous(self.dataframe, gamma=-4)

        phi_1, phi_2, phi_m = sc_liu.unconstrained_portfolio_weights_continuous(self.dataframe, gamma=4)

        # Checking the values of phi_1 weights
        self.assertAlmostEqual(np.mean(phi_1), -0.346020, delta=1e-5)
        self.assertAlmostEqual(phi_1[7],  -0.359256, delta=1e-4)
        self.assertAlmostEqual(phi_1[28], -0.347385, delta=1e-4)
        self.assertAlmostEqual(phi_1[-1], -0.338012, delta=1e-4)

        # Checking the values of phi_2 weights
        self.assertAlmostEqual(np.mean(phi_2), 0.313540, delta=1e-5)
        self.assertAlmostEqual(phi_2[7],  0.325810, delta=1e-4)
        self.assertAlmostEqual(phi_2[28], 0.314856, delta=1e-4)
        self.assertAlmostEqual(phi_2[-1], 0.305949, delta=1e-4)

        # Checking the values of phi_m weights
        self.assertAlmostEqual(np.mean(phi_m), 0.005066, delta=1e-5)
        self.assertAlmostEqual(phi_m[7],  0.002182, delta=1e-4)
        self.assertAlmostEqual(phi_m[28], 0.004920, delta=1e-4)
        self.assertAlmostEqual(phi_m[-1], 0.006312, delta=1e-4)


    def test_delta_neutral_continuous(self):
        """
        Tests the method which returns delta neutral portfolio weights in continuous case.
        """

        # Creating an object of the class
        sc_liu = OptimalConvergence()

        # Testing for the run fit before this method exception
        with self.assertRaises(Exception):
            sc_liu.delta_neutral_portfolio_weights_continuous(self.dataframe, gamma=4)

        sc_liu.fit(self.dataframe, mu_m=0.05, sigma_m=0.35, r=0.02)

        # Testing for the positive gamma exception
        with self.assertRaises(Exception):
            sc_liu.delta_neutral_portfolio_weights_continuous(self.dataframe, gamma=-4)

        phi_1, phi_2, phi_m = sc_liu.delta_neutral_portfolio_weights_continuous(self.dataframe, gamma=4)

        # Checking the values of phi_1 weights
        self.assertAlmostEqual(np.mean(phi_1), -0.330006, delta=1e-5)
        self.assertAlmostEqual(phi_1[7], -0.342951, delta=1e-4)
        self.assertAlmostEqual(phi_1[28], -0.331399, delta=1e-4)
        self.assertAlmostEqual(phi_1[-1], -0.321981, delta=1e-4)

        # Checking the values of phi_2 weights
        self.assertAlmostEqual(np.mean(phi_2), 0.330006, delta=1e-5)
        self.assertAlmostEqual(phi_2[7], 0.342951, delta=1e-4)
        self.assertAlmostEqual(phi_2[28], 0.331399, delta=1e-4)
        self.assertAlmostEqual(phi_2[-1], 0.321981, delta=1e-4)

        # Checking the values of phi_m weights
        self.assertAlmostEqual(np.mean(phi_m), 0.102041, delta=1e-5)
        self.assertAlmostEqual(phi_m[7], 0.102041, delta=1e-4)
        self.assertAlmostEqual(phi_m[28], 0.102041, delta=1e-4)
        self.assertAlmostEqual(phi_m[-1], 0.102041, delta=1e-4)


    def test_wealth_gain_continuous(self):
        """
        Tests the method which returns wealth gain in continuous case.
        """

        # Creating an object of the class
        sc_liu = OptimalConvergence()

        # Testing for the run fit before this method exception
        with self.assertRaises(Exception):
            sc_liu.wealth_gain_continuous(self.dataframe, gamma=4)

        sc_liu.fit(self.dataframe, mu_m=0.05, sigma_m=0.35, r=0.02)

        # Testing for the positive gamma exception
        with self.assertRaises(Exception):
            sc_liu.wealth_gain_continuous(self.dataframe, gamma=-4)

        wealth_gain = sc_liu.wealth_gain_continuous(self.dataframe, gamma=4)

        # Checking the values of phi_1 weights
        self.assertAlmostEqual(np.mean(wealth_gain), 0.999688, delta=1e-5)
        self.assertAlmostEqual(wealth_gain[7], 0.999411, delta=1e-4)
        self.assertAlmostEqual(wealth_gain[28], 0.999617, delta=1e-4)
        self.assertAlmostEqual(wealth_gain[-1], 1.0, delta=1e-4)