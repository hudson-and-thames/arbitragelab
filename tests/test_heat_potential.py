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

        test = HeatPotentials()

        test.fit(self.params, 0.01, 300)

        self.assertEqual(test.theta, 1)

        self.assertAlmostEqual(test.max_trade_duration,1.959999,delta=1e-4)


    def test_helper_functions(self) -> None:
        """
        Tests the helper functions
        """
        test = HeatPotentials()

        test.fit(self.params, 0.1, 300)

        grid = test.v(test.max_trade_duration)

        upsilon = test.upsilon(test.max_trade_duration)

        omega = test.omega(test.max_trade_duration)

        self.assertEqual(grid[-1], upsilon)

        self.assertEqual(grid[0], 0)

        self.assertAlmostEqual(omega, -0.14085, delta=1e-4)

        test.description()



    def test_core_functionality(self) -> None:
        """
        Tests the core functionality
        """
        test = HeatPotentials()

        test.fit(self.params, 0.1, 300)

        expected_output = (5.24263, -3.24266, 1.22724)

        np.testing.assert_almost_equal(test.optimal_levels(), expected_output, decimal=4)

        self.assertAlmostEqual(test.sharpe, expected_output[2], delta=1e-4)
