# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Test functions from Optimal Mean Reversion module.
"""

# pylint: disable=protected-access
import unittest
import numpy as np
import pandas as pd
import matplotlib
from arbitragelab.optimal_mean_reversion.cir_model import CoxIngersollRoss

class TestCoxIngersollRoss(unittest.TestCase):
    """
    Test the Cox-Ingersoll-Ross module
    """
    def setUp(self):
        """
        Set up testing data
        """
        test = CoxIngersollRoss()

        # Correct data for testing the module
        np.random.seed(30)
        self.delta_t = 1/252
        self.cir_example = test.cir_model_simulation(n=1000, theta_given=0.2,
                                                     mu_given=0.2, sigma_given=0.3,
                                                     delta_t_given=self.delta_t)


    def test_cir_generation(self):
        """
        Tests the cir process generation
        """
        # Assign the element of the class
        test = CoxIngersollRoss()
        # Generating the cir data based on given parameters

        test.fit(self.cir_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.001, 0.001])

        # Test generation of the cir data based on fitted parameters
        self.assertEqual(len(test.cir_model_simulation(n=1000)), 1000)

    def test_descriptive(self):
        """
        Tests descriptive functions
        """
        test = CoxIngersollRoss()
        # Generating the cir data based on given parameters
        test.fit(self.cir_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.001, 0.001])
        # Tests the plotting on one-dimensional array and displays optimal switching levels
        self.assertIsInstance(test.cir_plot_levels(self.cir_example, switching=True), matplotlib.figure.Figure)
        # Tests the plotting on one-dimensional array and doesn't display optimal switching levels
        test.cir_plot_levels(self.cir_example, switching=False)
        # Tests calling the description function that displays optimal switching levels
        self.assertIsInstance(test.cir_description(switching=True), pd.core.series.Series)
        # Tests calling the description function that doesn't display optimal switching levels
        test.cir_description(switching=False)

    def test_exceptions(self):
        """
        Tests if the exceptions are raised when unfitted data is passed.
        """
        # Assign the element of the class
        test = CoxIngersollRoss()

        # Test the non-optimal re-entering scenario
        test.fit(self.cir_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.001, 0.001])

        test.liquidation_level[0] = 1000
        test._check_optimal_switching()
        with self.assertWarns(Warning):
            test.optimal_switching_levels()

    def test_optimal_stopping(self):
        """
        Checks that optimal stopping problem is solved correctly
        """
        test = CoxIngersollRoss()
        # Fitting the data
        test.fit(self.cir_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.001, 0.001])

        # Calculating the optimal levels and then testing that it recalls it from memory correctly
        optimal_stopping_levels = [test.optimal_liquidation_level(),
                                   test.optimal_entry_level(),
                                   test.optimal_liquidation_level(),
                                   test.optimal_entry_level(),
                                   ]
        # Result we will be comparing our calculations to
        desired_result = [0.47420, 0.07594,
                          0.47420, 0.07594,]

        np.testing.assert_almost_equal(optimal_stopping_levels, desired_result, decimal=4)

    def test_optimal_switching(self):
        """
        Tests the optimal switching
        """

        test = CoxIngersollRoss()
        # Fitting the data
        test.fit(self.cir_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.001, 0.001])
        # Calculating the optimal switching levels and then testing that it recalls it from memory correctly
        optimal_switching_levels = [test.optimal_switching_levels()[0], test.optimal_switching_levels()[1],
                                    test.optimal_switching_levels()[0], test.optimal_switching_levels()[1]]

        desired_result = [0.18014812, 0.26504224, 0.18014812, 0.26504224]

        np.testing.assert_almost_equal(optimal_switching_levels, desired_result, decimal=3)
