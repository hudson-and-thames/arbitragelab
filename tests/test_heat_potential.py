import unittest
import numpy as np

from arbitragelab.heat_potential_approach.heat_potential import HeatPotentials


class TestHeatPotentials(unittest.TestCase):

    def setUp(self) -> None:

        self.params = (1.8557, 0.00653, 0.15)

    def test_fit(self) -> None:
        """
        Tests the correctness of the fit to a steady-state distribution
        """

        # Setting up the model
        test = HeatPotentials()

        test.fit(self.params, 0.01, 300)

        # Test the fitted parameters
        self.assertAlmostEqual(test.theta, 1, delta=1e-2)

        self.assertAlmostEqual(test.max_trade_duration, 1.959, delta=1e-3)

        test.description()

    def test_helper_functions(self) -> None:
        """
        Tests the helper functions
        """
        # Setting up the instance of the class
        test = HeatPotentials()

        test.fit(self.params, 0.1, 300)

        # Setting up the grid
        grid = test.v(test.max_trade_duration)

        # Calculating helper values
        upsilon = test.upsilon(test.max_trade_duration)

        omega = test.omega(test.max_trade_duration)

        # Testing
        self.assertAlmostEqual(grid[-1], upsilon, delta=1e-4)

        self.assertAlmostEqual(omega, -0.14095, delta=1e-4)

        test.description()

    def test_core_functionality(self) -> None:
        """
        Tests the core functionality
        """
        # Setting up the instance of the class
        test = HeatPotentials()

        test.fit(self.params, 0.1, 300)

        # Setting the expected output
        expected_output = (5.2423, -3.243, 1.2267)

        # Testing the optimal levels and sharpe calculation
        np.testing.assert_almost_equal(test.optimal_levels(), expected_output, decimal=4)

        self.assertAlmostEqual(test.sharpe_calculation(test.T, 5.2423, -3.243), expected_output[2], delta=1e-3)
