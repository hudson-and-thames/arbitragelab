# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests functions from O-U Model Optimal Threshold module.
"""

# pylint: disable=invalid-name, protected-access
import unittest
import numpy as np
import matplotlib.pyplot as plt

from arbitragelab.time_series_approach.ou_optimal_threshold_zeng import OUModelOptimalThresholdZeng


class TestOUModelOptimalThresholdZeng(unittest.TestCase):
    """
    Tests the class of O-U Model Optimal Threshold module (Zeng).
    """

    def setUp(self):
        """
        Set the testing variables.
        """

        # List with testing values for plotting function
        self.test_target = ["a_s", "b_s", "a_l", "b_l", "expected_return", "return_variance", "sharpe_ratio",
                            "expected_trade_length", "trade_length_variance", "error"]
        self.test_method = ["conventional_optimal_rule", "new_optimal_rule", "error"]

    def test_metrics(self):
        """
        Tests functions for metrics calculation.
        """

        # Creating an object of class
        test = OUModelOptimalThresholdZeng()

        # Initializing OU-process parameter
        test.construct_ou_model_from_given_parameters(theta=3.4241, mu=0.0237, sigma=0.0081)

        # Getting optimal thresholds by Conventional Optimal Rule.
        a_s, b_s, a_l, b_l = test.get_threshold_by_conventional_optimal_rule(c=0.02)

        # Testing
        self.assertAlmostEqual(test.return_variance(a=a_s, b=b_s, c=0.02), 2.5454987392050962e-05, places=7)
        self.assertAlmostEqual(test.expected_return(a=a_s, b=b_s, c=0.02), 0.0003012195, places=6)
        self.assertAlmostEqual(test.return_variance(a=a_l, b=b_l, c=0.02), 2.207110248316157e-05, places=7)
        self.assertAlmostEqual(test.expected_return(a=a_l, b=b_l, c=0.02), 0.0003012195, places=6)

        # Getting optimal thresholds by New Optimal Rule.
        a_s, b_s, a_l, b_l = test.get_threshold_by_new_optimal_rule(c=0.02)

        # Testing
        self.assertAlmostEqual(test.return_variance(a=a_s, b=b_s, c=0.02), 3.467078460345948e-05, places=7)
        self.assertAlmostEqual(test.expected_return(a=a_s, b=b_s, c=0.02), 0.00043061662, places=6)
        self.assertAlmostEqual(test.sharpe_ratio(a=a_s, b=b_s, c=0.02, rf=0), 0.073132, places=4)
        self.assertAlmostEqual(test.return_variance(a=a_l, b=b_l, c=0.02), 3.467078460345948e-05, places=7)
        self.assertAlmostEqual(test.expected_return(a=a_l, b=b_l, c=0.02), 0.00043061662, places=6)
        self.assertAlmostEqual(test.sharpe_ratio(a=a_l, b=b_l, c=0.02, rf=0), 0.073132, places=4)

    def test_plot(self):
        """
        Tests functions for plotting.
        """

        # Creating an object of class
        test = OUModelOptimalThresholdZeng()

        # Initializing OU-process parameter
        test.construct_ou_model_from_given_parameters(theta=3.4241, mu=0.0237, sigma=0.0081)

        # Testing all valid options
        c_list = np.linspace(0, 0.01, 2)
        for t in self.test_target[:-1]:
            for m in self.test_method[:-1]:
                fig = test.plot_target_vs_c(target=t, method=m, c_list=c_list)
                self.assertEqual(type(fig), type(plt.figure()))
            plt.close("all")

        rf_list = np.linspace(0, 0.05, 2)
        for m in self.test_method[:-1]:
            fig = test.plot_sharpe_ratio_vs_rf(method=m, rf_list=rf_list, c=0)
            self.assertEqual(type(fig), type(plt.figure()))
            plt.close("all")

    def test_exeptions(self):
        """
        Tests exceptions in the module.
        """

        # Creating an object of class
        test = OUModelOptimalThresholdZeng()

        # Initializing OU-process parameter
        test.construct_ou_model_from_given_parameters(theta=3.4241, mu=0.0237, sigma=0.0081)

        # Testing invalid options
        c_list = np.linspace(0, 0.01, 2)
        with self.assertRaises(Exception):
            test.plot_target_vs_c(target=self.test_target[-1], method=self.test_method[0], c_list=c_list)

        with self.assertRaises(Exception):
            test.plot_target_vs_c(target=self.test_target[0], method=self.test_method[-1], c_list=c_list)

        rf_list = np.linspace(0, 0.05, 2)
        with self.assertRaises(Exception):
            test.plot_sharpe_ratio_vs_rf(method=self.test_method[-1], rf_list=rf_list, c=0)

    def test_numerical(self):
        """
        Tests functions for numerical calculation.
        """

        # Creating an object of class
        test = OUModelOptimalThresholdZeng()

        # Initializing OU-process parameter
        test.construct_ou_model_from_given_parameters(theta=3.4241, mu=0.0237, sigma=0.0081)

        # Testing whether the output value is correct
        self.assertAlmostEqual(test._transform_to_dimensionless(0), -92.034486, places=2)
        self.assertAlmostEqual(test._transform_to_dimensionless(1), -65.156040, places=2)
        self.assertAlmostEqual(test._transform_to_dimensionless(-1), -118.912932, places=2)

        self.assertAlmostEqual(test._back_transform_from_dimensionless(0), 3.4241, places=2)
        self.assertAlmostEqual(test._back_transform_from_dimensionless(1), 3.4613, places=2)
        self.assertAlmostEqual(test._back_transform_from_dimensionless(-1), 3.386895, places=2)

        self.assertAlmostEqual(test._g_1(0, 0), -1.2337, places=2)
        self.assertAlmostEqual(test._g_1(0, 1), -0.58633, places=3)
        self.assertAlmostEqual(test._g_1(0, -1), test._g_1(0, 1), places=3)
        self.assertAlmostEqual(test._g_1(1, 0), -2.0699895, places=2)
        self.assertAlmostEqual(test._g_1(-1, 0), test._g_1(1, 0), places=2)
        self.assertAlmostEqual(test._g_1(1, 1), -1.486195, places=2)
        self.assertAlmostEqual(test._g_1(-1, -1), test._g_1(1, 1), places=2)

        self.assertAlmostEqual(test._g_2(0, 0), -1.2337, places=2)
        self.assertAlmostEqual(test._g_2(0, 1), -0.66610, places=3)
        self.assertAlmostEqual(test._g_2(0, -1), test._g_2(0, 1), places=3)
        self.assertAlmostEqual(test._g_2(1, 0), -2.84992, places=2)
        self.assertAlmostEqual(test._g_2(-1, 0), test._g_2(1, 0), places=2)
        self.assertAlmostEqual(test._g_2(1, 1), -1.486195, places=2)
        self.assertAlmostEqual(test._g_2(-1, -1), test._g_2(1, 1), places=2)

    def test_none(self):
        """
        Tests functions with default arguments equal None.
        """

        # Creating an object of class
        test = OUModelOptimalThresholdZeng()

        # Initializing OU-process parameter
        test.construct_ou_model_from_given_parameters(theta=3.4241, mu=0.0237, sigma=0.0081)

        # Setting initial guess
        c_trans = 0.02 * np.sqrt((2 * test.mu)) / test.sigma
        initial_guess = c_trans + 1e-2 * np.sqrt((2 * test.mu)) / test.sigma

        # Getting optimal thresholds by Conventional Optimal Rule.
        a_s, b_s, a_l, b_l = test.get_threshold_by_conventional_optimal_rule(c=0.02, initial_guess=initial_guess)
        self.assertAlmostEqual(test.expected_return(a=a_s, b=b_s, c=0.02), 0.0003012195, places=6)

        # Getting optimal thresholds by New Optimal Rule.
        a_s, b_s, a_l, b_l = test.get_threshold_by_new_optimal_rule(c=0.02, initial_guess=initial_guess)
        self.assertAlmostEqual(test.expected_return(a=a_l, b=b_l, c=0.02), 0.00043061662, places=6)
