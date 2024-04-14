"""
Tests functions from O-U Model Optimal Threshold module.
"""

# pylint: disable=invalid-name, protected-access
import unittest
import numpy as np
import matplotlib.pyplot as plt

from arbitragelab.time_series_approach.ou_optimal_threshold_bertram import OUModelOptimalThresholdBertram


class TestOUModelOptimalThresholdBertram(unittest.TestCase):
    """
    Tests the class of O-U Model Optimal Threshold module (Bertram).
    """

    def setUp(self):
        """
        Set the testing variables.
        """

        # List with testing values for plotting function
        self.test_target = ["a", "m", "expected_return", "return_variance", "sharpe_ratio",
                            "expected_trade_length", "trade_length_variance", "error"]
        self.test_method = ["maximize_expected_return", "maximize_sharpe_ratio", "error"]

    def test_metrics(self):
        """
        Tests functions for metrics calculation.
        """

        # Creating an object of class
        test = OUModelOptimalThresholdBertram()

        # Initializing OU-process parameter
        test.construct_ou_model_from_given_parameters(theta=0, mu=180.9670, sigma=0.1538)

        # Getting optimal thresholds by maximizing the expected return
        a, m = test.get_threshold_by_maximize_expected_return(c=0.001)

        # Testing
        self.assertAlmostEqual(test.expected_trade_length(a=a, m=m), 0.017122, places=4)
        self.assertAlmostEqual(test.trade_length_variance(a=a, m=m), 0.00015438, places=6)
        self.assertAlmostEqual(test.return_variance(a=a, m=m, c=0.001), 0.0021857, places=5)
        self.assertAlmostEqual(test.expected_return(a=a, m=m, c=0.001), 0.492358, places=3)

        # Getting optimal thresholds by maximizing the Sharpe ratio
        a, m = test.get_threshold_by_maximize_sharpe_ratio(c=0.001, rf=0.01)

        # Testing
        self.assertAlmostEqual(test.sharpe_ratio(a=a, m=m, c=0.001, rf=0.01), 3.86293, places=2)

    def test_plot(self):
        """
        Tests functions for plotting.
        """

        # Creating an object of class
        test = OUModelOptimalThresholdBertram()

        # Initializing OU-process parameter
        test.construct_ou_model_from_given_parameters(theta=0, mu=180.9670, sigma=0.1538)

        # Testing all valid options
        c_list = np.linspace(0, 0.01, 1)
        for t in self.test_target[:-1]:
            for m in self.test_method[:-1]:
                fig = test.plot_target_vs_c(target=t, method=m, c_list=c_list)
                self.assertEqual(type(fig), type(plt.figure()))
            plt.close("all")

        rf_list = np.linspace(0, 0.05, 1)
        for t in self.test_target[:-1]:
            for m in self.test_method[:-1]:
                fig = test.plot_target_vs_rf(target=t, method=m, rf_list=rf_list, c=0)
                self.assertEqual(type(fig), type(plt.figure()))
            plt.close("all")

    def test_exeptions(self):
        """
        Tests exceptions in the module.
        """

        # Creating an object of class
        test = OUModelOptimalThresholdBertram()

        # Initializing OU-process parameter
        test.construct_ou_model_from_given_parameters(theta=0, mu=180.9670, sigma=0.1538)

        # Testing invalid options
        c_list = np.linspace(0, 0.01, 2)
        with self.assertRaises(Exception):
            test.plot_target_vs_c(target=self.test_target[-1], method=self.test_method[0], c_list=c_list)

        with self.assertRaises(Exception):
            test.plot_target_vs_c(target=self.test_target[0], method=self.test_method[-1], c_list=c_list)

        rf_list = np.linspace(0, 0.05, 2)
        with self.assertRaises(Exception):
            test.plot_target_vs_rf(target=self.test_target[-1], method=self.test_method[0], rf_list=rf_list, c=0)

        with self.assertRaises(Exception):
            test.plot_target_vs_rf(target=self.test_target[0], method=self.test_method[-1], rf_list=rf_list, c=0)

    def test_numerical(self):
        """
        Tests functions for numerical calculation.
        """

        # Creating an object of class
        test = OUModelOptimalThresholdBertram()

        # Initializing OU-process parameter
        test.construct_ou_model_from_given_parameters(theta=0, mu=180.9670, sigma=0.1538)

        # Testing whether the output value is correct
        self.assertAlmostEqual(test._erfi_scaler(0), 0.0, places=1)
        self.assertAlmostEqual(test._erfi_scaler(0.01), 1.3087, places=2)
        self.assertAlmostEqual(test._erfi_scaler(-0.01), -test._erfi_scaler(0.01), places=2)
        self.assertAlmostEqual(test._erfi_scaler(1), np.inf)
        self.assertAlmostEqual(test._erfi_scaler(-1), -test._erfi_scaler(1))

    def test_none(self):
        """
        Tests functions with default arguments equal None.
        """

        # Creating an object of class
        test = OUModelOptimalThresholdBertram()

        # Initializing OU-process parameter
        test.construct_ou_model_from_given_parameters(theta=0, mu=180.9670, sigma=0.1538)

        # Getting optimal thresholds by maximizing the expected return
        a, m = test.get_threshold_by_maximize_expected_return(c=0.001, initial_guess=test.theta - 0.001 - 1e-2)
        self.assertAlmostEqual(test.expected_return(a=a, m=m, c=0.001), 0.492358, places=3)

        # Getting optimal thresholds by maximizing the Sharpe ratio
        a, m = test.get_threshold_by_maximize_sharpe_ratio(c=0.001, rf=0.01, initial_guess=test.theta - 0.001 - 1e-2)
        self.assertAlmostEqual(test.sharpe_ratio(a=a, m=m, c=0.001, rf=0.01), 3.86293, places=2)
