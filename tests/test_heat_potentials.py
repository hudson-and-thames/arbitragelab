# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests the heat potentials approach from the HeatPotentials module of ArbitrageLab.
"""
import unittest
import numpy as np
import pandas as pd

from arbitragelab.optimal_mean_reversion.heat_potentials import HeatPotentials


class TestHeatPotentials(unittest.TestCase):
    """
    Tests the HeatPotentials module.
    """

    def setUp(self):
        """
        Sets up the universal testing values.
        """

        self.params = (1.8557, 0.00653, 0.15)

    def test_fit(self):
        """
        Tests the correctness of the fit to a steady-state distribution.
        """

        # Setting up the model
        test = HeatPotentials()

        test.fit(self.params, 0.01, 300)

        # Test the fitted parameters
        self.assertAlmostEqual(test.theta, 1, delta=1e-2)

        self.assertAlmostEqual(test.max_trade_duration, 1.959, delta=1e-3)

        # Tests calling the description function
        descr = test.description()
        self.assertIsInstance(descr, pd.Series)

    def test_helper_functions(self):
        """
        Tests the helper functions.
        """

        # Setting up the instance of the class
        test = HeatPotentials()

        test.fit(self.params, 0.1, 300)

        # Setting up the grid
        grid = test.v(test.max_trade_duration)

        # Calculating helper values
        upsilon = test.upsilon(test.max_trade_duration)

        omega = test.omega(test.max_trade_duration)

        # Testing helper functions calculation
        self.assertAlmostEqual(grid[-1], upsilon, delta=1e-4)

        self.assertAlmostEqual(omega, -0.14095, delta=1e-4)

        # Tests if the description function returns the instance of the correct class
        self.assertIsInstance(test.description(), pd.Series)

    def test_core_functionality(self):
        """
        Tests the core functionality.
        """

        # Setting up the instance of the class
        test = HeatPotentials()

        test.fit(self.params, 0.1, 300)

        # Setting the expected output
        expected_output = (5.2423, -3.243, 1.2267)

        # Testing the optimal levels and sharpe calculation
        np.testing.assert_almost_equal(test.optimal_levels(), expected_output, decimal=4)

        self.assertAlmostEqual(test.sharpe_calculation(test.max_trade_duration, 5.2423, -3.243),
                               expected_output[2], delta=1e-3)
