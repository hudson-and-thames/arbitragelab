# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Test functions from Optimal Mean Reversion module.
"""

# pylint: disable=protected-access
import os
import unittest
import numpy as np
import pandas as pd
import matplotlib

from arbitragelab.optimal_mean_reversion.xou_model import ExponentialOrnsteinUhlenbeck

class TestExponentialOrnsteinUhlenbeck(unittest.TestCase):
    """
    Test the Exponential Ornstein-Uhlenbeck module.
    """
    def setUp(self):
        """
        Set up testing data.
        """

        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/incorrect_data.csv'
        data = pd.read_csv(self.path)
        data = data.set_index('Date')

        # Incorrect data (non-logarithmized asset prices) for testing the module exceptions
        self.dataframe = data[['GDX', 'GLD']]

        test = ExponentialOrnsteinUhlenbeck()

        # Correct data for testing the module
        np.random.seed(31)
        self.delta_t = 1/252
        self.xou_example = test.ou_model_simulation(n=1000, theta_given=1,
                                                    mu_given=0.6, sigma_given=0.2,
                                                    delta_t_given=self.delta_t)

    def test_xou_generation(self):
        """
        Tests the XOU process generation.
        """

        # Assign the element of the class
        test = ExponentialOrnsteinUhlenbeck()

        # Generating the XOU data based on given parameters
        np.random.seed(31)
        xou_generated = test.xou_model_simulation(n=1000, theta_given=1,
                                                  mu_given=0.6, sigma_given=0.2,
                                                  delta_t_given=self.delta_t)
        # Calculating the summary difference
        diff = sum(np.exp(self.xou_example) - xou_generated)

        # Testing
        self.assertEqual(diff, 0)

        test.fit(self.xou_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.02, 0.02])

        # Test generation of the XOU data based on fitted parameters
        np.random.seed(31)
        xou_fitted_generated = test.xou_model_simulation(n=1000)

        np.random.seed(31)
        ou_fitted_generated = test.ou_model_simulation(n=1000)

        # Calculating the summary difference
        diff_fitted = sum(np.exp(ou_fitted_generated) - xou_fitted_generated)

        # Testing that the generated values are close
        self.assertEqual(diff_fitted, 0)

    def test_descriptive(self):
        """
        Tests descriptive functions.
        """

        # Assign the element of the class
        test = ExponentialOrnsteinUhlenbeck()

        # Generating the XOU data based on given parameters
        test.fit(self.xou_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.02, 0.02])

        # "Creating" a portfolio of two assets
        asset_prices = np.array([self.xou_example, self.xou_example]).transpose()
        test.B_value = 0

        # Tests the plotting on two-dimensional array and displays optimal switching levels
        plot_twodim = test.xou_plot_levels(asset_prices, switching=True)
        self.assertIsInstance(plot_twodim, matplotlib.figure.Figure)

        # Tests the plotting on one-dimensional array and displays optimal switching levels
        plot_onedim = test.xou_plot_levels(self.xou_example, switching=True)
        self.assertIsInstance(plot_onedim, matplotlib.figure.Figure)

        # Tests the plotting on one-dimensional array and doesn't display optimal switching levels
        plot_nolevels = test.xou_plot_levels(self.xou_example, switching=False)
        self.assertIsInstance(plot_nolevels, matplotlib.figure.Figure)

        # Tests calling the description function that displays optimal switching levels
        descr_with_levels = test.xou_description(switching=True)
        self.assertIsInstance(descr_with_levels, pd.Series)

        # Tests calling the description function that doesn't display optimal switching levels
        descr_without_levels = test.xou_description(switching=False)
        self.assertIsInstance(descr_without_levels, pd.Series)

    def test_exceptions(self):
        """
        Tests if the exceptions are raised when unfitted data is passed.
        """

        # Assign the element of the class
        test = ExponentialOrnsteinUhlenbeck()

        # Generating incorrect data
        np.random.seed(31)
        incorrect_data = test.ou_model_simulation(n=1000, theta_given=1,
                                                  mu_given=30, sigma_given=0.2,
                                                  delta_t_given=self.delta_t)

        # Assert unsuitable data error
        test.fit(incorrect_data, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.02, 0.02])
        #with self.assertRaises(Exception):
        #    test.xou_optimal_entry_interval()

        # Assert warning about not optimal market re-entering
        test.fit(self.dataframe, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.02, 0.02])
        with self.assertWarns(Warning):
            test.optimal_switching_levels()
        with self.assertRaises(Exception):
            test.xou_plot_levels(self.dataframe)

        # Assert warning if the (a) condition is not satisfied
        test.fit(self.xou_example, data_frequency="D", discount_rate=10.05,
                 transaction_cost=[0.02, 0.02])
        test.entry_level[0] = [0, None]
        test._condition_optimal_switching_a()
        with self.assertWarns(Warning):
            with self.assertRaises(ValueError):
                test.optimal_switching_levels()

        # Assert warning if the optimal switching inequality is not satisfied
        test.fit(self.xou_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.02, 0.02])
        test.entry_level[0] = [0.1, None]
        test._condition_optimal_switching_inequality()
        with self.assertWarns(Warning):
            test.optimal_switching_levels()

    def test_optimal_stopping(self):
        """
        Checks that optimal stopping problem is solved correctly.
        """

        # Assign the element of the class
        test = ExponentialOrnsteinUhlenbeck()

        # Fitting the data
        test.fit(self.xou_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.02, 0.02])

        # Calculating the optimal levels and then testing that it recalls it from memory correctly
        optimal_stopping_levels = [test.xou_optimal_liquidation_level(),
                                   test.xou_optimal_entry_interval()[0],
                                   test.xou_optimal_entry_interval()[1],
                                   test.xou_optimal_liquidation_level(),
                                   test.xou_optimal_entry_interval()[0],
                                   test.xou_optimal_entry_interval()[1]]

        # Result we will be comparing our calculations to
        desired_result = [1.05623, -9.7139, 0.70871,
                          1.05623, -9.7139, 0.70871]

        # Testing
        np.testing.assert_almost_equal(optimal_stopping_levels, desired_result, decimal=3)

    def test_optimal_switching(self):
        """
        Tests the optimal switching.
        """

        # Assign the element of the class
        test = ExponentialOrnsteinUhlenbeck()

        # Fitting the data
        test.fit(self.xou_example, data_frequency="D", discount_rate=0.05,
                 transaction_cost=[0.02, 0.02])

        # Calculating the optimal switching levels and then testing that it recalls it from memory correctly
        optimal_switching_levels = [test.optimal_switching_levels()[0], test.optimal_switching_levels()[1],
                                    test.optimal_switching_levels()[0], test.optimal_switching_levels()[1]]

        # Result we will be comparing our calculations to
        desired_result = [0.83884, 0.97899,
                          0.83884, 0.97899]

        # Testing
        np.testing.assert_almost_equal(optimal_switching_levels, desired_result, decimal=3)
