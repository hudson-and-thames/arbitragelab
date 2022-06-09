# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Test functions from CIR model of the Optimal Mean Reversion module.
"""

# pylint: disable=protected-access
import os
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

        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/incorrect_data.csv'
        data = pd.read_csv(self.path)
        data = data.set_index('Date')

        # Incorrect data (non-logarithmized asset prices) for testing the module exceptions
        self.dataframe = data[['GDX', 'GLD']]

        test = CoxIngersollRoss()

        # Correct data for testing the module
        np.random.seed(30)
        self.delta_t = 1/252
        self.cir_example = test.cir_model_simulation(n=1000, theta_given=0.2,
                                                     mu_given=0.2, sigma_given=0.3,
                                                     delta_t_given=self.delta_t)

    def test_cir_generation(self):
        """
        Tests the cir process generation.
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
        Tests descriptive functions.
        """

        # Assign the element of the class
        test = CoxIngersollRoss()

        # Generating the cir data based on given parameters
        test.fit(self.cir_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.001, 0.001])

        # Tests the plotting on one-dimensional array and displays optimal switching levels
        plot_levels = test.cir_plot_levels(self.cir_example, switching=True)
        self.assertIsInstance(plot_levels, matplotlib.figure.Figure)

        # Tests the plotting on one-dimensional array and doesn't display optimal switching levels
        plot_nolevels = test.cir_plot_levels(self.cir_example, switching=False)
        self.assertIsInstance(plot_nolevels, matplotlib.figure.Figure)

        # Tests calling the description function that displays optimal switching levels
        descr_with_levels = test.cir_description(switching=True)
        self.assertIsInstance(descr_with_levels, pd.Series)

        # Tests calling the description function that doesn't display optimal switching levels
        descr_without_levels = test.cir_description(switching=False)
        self.assertIsInstance(descr_without_levels, pd.Series)

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

        # Assert warning about not optimal market re-entering
        with self.assertWarns(Warning):
            test.fit(self.dataframe, data_frequency="D", discount_rate=0.05,
                     transaction_cost=[0.001, 0.001])
            test.optimal_switching_levels()

    def test_optimal_stopping(self):
        """
        Checks that optimal stopping problem is solved correctly.
        """

        # Assign the element of the class
        test = CoxIngersollRoss()

        # Fitting the data
        test.fit(self.cir_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.001, 0.001])

        # Calculating the optimal levels and then testing that it recalls it from memory correctly
        optimal_stopping_levels = [test.optimal_liquidation_level(),
                                   test.optimal_entry_level(),
                                   test.optimal_liquidation_level(),
                                   test.optimal_entry_level()]

        # Result we will be comparing our calculations to
        desired_result = [0.47420, 0.07594,
                          0.47420, 0.07594]

        # Testing values
        np.testing.assert_almost_equal(optimal_stopping_levels, desired_result, decimal=2)

    def test_optimal_switching(self):
        """
        Tests the optimal switching.
        """

        # Assign the element of the class
        test = CoxIngersollRoss()

        # Fitting the data
        test.fit(self.cir_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.001, 0.001])

        # Calculating the optimal switching levels and then testing that it recalls it from memory correctly
        optimal_switching_levels = [test.optimal_switching_levels()[0], test.optimal_switching_levels()[1],
                                    test.optimal_switching_levels()[0], test.optimal_switching_levels()[1]]

        desired_result = [0.18014812, 0.26504224,
                          0.18014812, 0.26504224]

        # Testing values
        np.testing.assert_almost_equal(optimal_switching_levels, desired_result, decimal=3)
