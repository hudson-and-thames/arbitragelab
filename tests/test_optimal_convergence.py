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

        # Testing negative values of sigma warning
        with self.assertWarns(UserWarning):
            sc_liu.fit(self.dataframe, mu_m=0.05, sigma_m=0.35, r=0.02)

        sc_liu.fit(self.shell_rdp_data, mu_m=0.05, sigma_m=0.35, r=0.02)

        # Checking parameter values
        self.assertAlmostEqual(sc_liu.lambda_1, -0.000662, delta=1e-4)
        self.assertAlmostEqual(sc_liu.lambda_2, 0.004175, delta=1e-4)
        self.assertAlmostEqual(sc_liu.b_squared, 0.014012, delta=1e-3)
        self.assertAlmostEqual(sc_liu.sigma_squared, 0.036801, delta=1e-4)
        self.assertAlmostEqual(sc_liu.beta, -0.498947, delta=1e-4)


    def test_describe(self):
        """
        Tests the describe method in the class.
        """

        # Creating an object of the class
        sc_liu = OptimalConvergence()

        # Testing for the run fit before this method exception
        with self.assertRaises(Exception):
            sc_liu.describe()

        sc_liu.fit(self.shell_rdp_data, mu_m=0.05, sigma_m=0.35, r=0.02)

        index = ['Ticker of first stock', 'Ticker of second stock',
                 'lambda_1', 'lambda_2', 'b_squared', 'sigma_squared',
                 'beta']

        data = ['Shell', 'RDP', -0.000662, 0.004175, 0.014012, 0.036801, -0.498947]

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
            sc_liu.unconstrained_portfolio_weights_continuous(self.shell_rdp_data, gamma=4)

        sc_liu.fit(self.shell_rdp_data, mu_m=0.05, sigma_m=0.35, r=0.02)

        # Testing for the positive gamma exception
        with self.assertRaises(Exception):
            sc_liu.unconstrained_portfolio_weights_continuous(self.shell_rdp_data, gamma=-4)

        phi_1, phi_2, phi_m = sc_liu.unconstrained_portfolio_weights_continuous(self.shell_rdp_data, gamma=4)

        # Checking the values of phi_1 weights
        self.assertAlmostEqual(np.mean(phi_1), -0.054245, delta=1e-5)
        self.assertAlmostEqual(phi_1[7],  -0.051831, delta=1e-4)
        self.assertAlmostEqual(phi_1[28], -0.052293, delta=1e-4)
        self.assertAlmostEqual(phi_1[-1], -0.057334, delta=1e-4)

        # Checking the values of phi_2 weights
        self.assertAlmostEqual(np.mean(phi_2), 0.083759, delta=1e-5)
        self.assertAlmostEqual(phi_2[7],  0.078997, delta=1e-4)
        self.assertAlmostEqual(phi_2[28], 0.079711, delta=1e-4)
        self.assertAlmostEqual(phi_2[-1], 0.089716, delta=1e-4)

        # Checking the values of phi_m weights
        self.assertAlmostEqual(np.mean(phi_m), 0.116766, delta=1e-5)
        self.assertAlmostEqual(phi_m[7],  0.115595, delta=1e-4)
        self.assertAlmostEqual(phi_m[28], 0.115720, delta=1e-4)
        self.assertAlmostEqual(phi_m[-1], 0.118197, delta=1e-4)


    def test_delta_neutral_continuous(self):
        """
        Tests the method which returns delta neutral portfolio weights in continuous case.
        """

        # Creating an object of the class
        sc_liu = OptimalConvergence()

        # Testing for the run fit before this method exception
        with self.assertRaises(Exception):
            sc_liu.delta_neutral_portfolio_weights_continuous(self.shell_rdp_data, gamma=4)

        sc_liu.fit(self.shell_rdp_data, mu_m=0.05, sigma_m=0.35, r=0.02)

        # Testing for the positive gamma exception
        with self.assertRaises(Exception):
            sc_liu.delta_neutral_portfolio_weights_continuous(self.shell_rdp_data, gamma=-4)

        phi_1, phi_2, phi_m = sc_liu.delta_neutral_portfolio_weights_continuous(self.shell_rdp_data, gamma=4)

        # Checking the values of phi_1 weights
        self.assertAlmostEqual(np.mean(phi_1), -0.068540, delta=1e-5)
        self.assertAlmostEqual(phi_1[7], -0.064546, delta=1e-4)
        self.assertAlmostEqual(phi_1[28], -0.065130, delta=1e-4)
        self.assertAlmostEqual(phi_1[-1], -0.073525, delta=1e-4)

        # Checking the values of phi_2 weights
        self.assertAlmostEqual(np.mean(phi_2), 0.068540, delta=1e-5)
        self.assertAlmostEqual(phi_2[7], 0.064546, delta=1e-4)
        self.assertAlmostEqual(phi_2[28], 0.065130, delta=1e-4)
        self.assertAlmostEqual(phi_2[-1], 0.073525, delta=1e-4)

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

        # Testing for the positive gamma exception
        with self.assertRaises(Exception):
            sc_liu.wealth_gain_continuous(gamma=-4)

        wealth_gain = sc_liu.wealth_gain_continuous(gamma=4)

        # Checking the values of phi_1 weights
        self.assertAlmostEqual(np.mean(wealth_gain), 1.035109, delta=1e-5)
        self.assertAlmostEqual(wealth_gain[7], 1.030627, delta=1e-4)
        self.assertAlmostEqual(wealth_gain[28], 1.030783, delta=1e-4)
        self.assertAlmostEqual(wealth_gain[-1], 1.044104, delta=1e-4)
