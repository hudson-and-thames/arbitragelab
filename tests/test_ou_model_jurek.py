# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Test functions for the Jurek OU model in the Stochastic Control Approach module.
"""
import warnings
import unittest
import os
from unittest import mock
import numpy as np
import pandas as pd

from arbitragelab.stochastic_control_approach.ou_model_jurek import OUModelJurek

# pylint: disable=protected-access

class TestOUModelJurek(unittest.TestCase):
    """
    Test Jurek OU model in Stochastic Control Approach module.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up data and parameters.
        """

        np.random.seed(0)

        project_path = os.path.dirname(__file__)

        # Setting up the first dataset.
        path = project_path + '/test_data/gld_gdx_data.csv'
        data = pd.read_csv(path)
        data = data.set_index('Date')
        cls.dataframe = data[['GLD', 'GDX']]

        # Setting up the second dataset.
        path = project_path + '/test_data/shell-rdp-close_USD.csv'
        data = pd.read_csv(path, index_col='Date').ffill()
        data.index = pd.to_datetime(data.index, format="%d/%m/%Y")
        cls.shell_rdp_data = data


    def test_fit(self):
        """
        Tests the fit method in the class.
        """

        # Creating an object of the class.
        sc_jurek = OUModelJurek()

        # Testing for the adf test statistic warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=True, significance_level=0.95)

        # Checking parameter values for spread calculation.
        self.assertAlmostEqual(np.mean(sc_jurek.spread), 0.5328233514, delta=1e-7)
        self.assertAlmostEqual(sc_jurek.spread[7], 0.5532053519, delta=1e-7)
        self.assertAlmostEqual(sc_jurek.spread[28], 0.5423901587, delta=1e-7)
        self.assertAlmostEqual(sc_jurek.spread[-1], 0.5054215932, delta=1e-7)

        # Checking other parameter values.
        self.assertAlmostEqual(sc_jurek.mu, 0.532823, delta=1e-4)
        self.assertAlmostEqual(sc_jurek.k, 10.2728, delta=1e-4)
        self.assertAlmostEqual(sc_jurek.sigma, 0.0743999, delta=1e-4)

        # Testing for the adf_test=False flag.
        sc_jurek.fit(self.shell_rdp_data, delta_t=1 / 252, adf_test=False)


    def test_describe(self):
        """
        Tests the describe method in the class.
        """

        # Creating an object of the class.
        sc_jurek = OUModelJurek()

        # Testing for the run fit before this method exception.
        with self.assertRaises(Exception):
            sc_jurek.describe()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=False)

        index = ['Ticker of first stock', 'Ticker of second stock', 'Scaled Spread weights',
                 'long-term mean', 'rate of mean reversion', 'standard deviation', 'half-life']

        data = ['GLD', 'GDX', [0.779, -0.221], 0.532823, 10.2728, 0.0743999, 0.067474]

        # Testing the output of describe method.
        pd.testing.assert_series_equal(pd.Series(index=index,data=data), sc_jurek.describe(), check_exact=False, atol=1e-4)


    def test_optimal_weights(self):
        """
        Tests the optimal portfolio weights method in the class.
        """

        # Creating an object of the class.
        sc_jurek = OUModelJurek()

        # Testing for the run fit before this method exception.
        with self.assertRaises(Exception):
            sc_jurek.optimal_portfolio_weights(self.dataframe, beta=0.01, gamma=0.5, utility_type=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=False)

        # Testing for the invalid utility_type exception.
        with self.assertRaises(Exception):
            sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 0.5, utility_type=10)

        # Testing for the invalid gamma exception.
        with self.assertRaises(Exception):
            sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = -1)


        weights = sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 0.5, utility_type=1)
        # Checking the values of weights for gamma = 0.5 and utility_type=1.
        self.assertAlmostEqual(np.mean(weights), -2.7252320178, delta=1e-7)
        self.assertAlmostEqual(weights[7], -58.5665057992, delta=1e-7)
        self.assertAlmostEqual(weights[28], -30.3169549896, delta=1e-7)
        self.assertAlmostEqual(weights[-1], 92.5766316230, delta=1e-7)

        weights = sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 1, utility_type=1)
        # Checking the values of weights for gamma = 1 and utility_type=1.
        self.assertAlmostEqual(np.mean(weights), -4.8129160068, delta=1e-7)
        self.assertAlmostEqual(weights[7], -42.82308043, delta=1e-7)
        self.assertAlmostEqual(weights[28], -22.6539479, delta=1e-7)
        self.assertAlmostEqual(weights[-1], 46.28831581, delta=1e-7)

        weights = sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 0.5, utility_type=2)
        # Checking the values of weights for gamma = 0.5 and utility_type=2.
        self.assertAlmostEqual(np.mean(weights), -2947786.5697284327, delta=1e-7)
        self.assertAlmostEqual(weights[7], -9803080.87074925, delta=1e-7)
        self.assertAlmostEqual(weights[28], -1762115.70956469, delta=1e-7)
        self.assertAlmostEqual(weights[-1], -54007.10111104, delta=1e-7)

        weights = sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 2, utility_type=1)
        # Checking the values of weights for gamma = 2 and utility_type=1.
        self.assertAlmostEqual(np.mean(weights), -5.3267767287, delta=1e-7)
        self.assertAlmostEqual(weights[7], -31.15603589, delta=1e-7)
        self.assertAlmostEqual(weights[28], -16.4211391034, delta=1e-7)
        self.assertAlmostEqual(weights[-1], 23.14415791, delta=1e-7)

        weights = sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 2, utility_type=2)
        # Checking the values of weights for gamma = 2 and utility_type=2.
        self.assertAlmostEqual(np.mean(weights), 28.7539986245, delta=1e-7)
        self.assertAlmostEqual(weights[7], 34.13526822, delta=1e-7)
        self.assertAlmostEqual(weights[28], 28.56326627, delta=1e-7)
        self.assertAlmostEqual(weights[-1], 23.14415791, delta=1e-7)


    def test_stabilization_region(self):
        """
        Tests the stabilization region method in the class.
        """

        # Creating an object of the class.
        sc_jurek = OUModelJurek()

        # Testing for the run fit before this method exception.
        with self.assertRaises(Exception):
            sc_jurek.stabilization_region(self.dataframe, beta=0.01, gamma=0.5, utility_type=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=False)

        # Testing for the invalid utility_type exception.
        with self.assertRaises(Exception):
            sc_jurek.stabilization_region(self.dataframe, beta=0.01, gamma=0.5, utility_type=10)

        # Testing for the invalid gamma exception.
        with self.assertRaises(Exception):
            sc_jurek.stabilization_region(self.dataframe, beta = 0.01, gamma = -1)

        # Testing for gamma = 1 warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.stabilization_region(self.dataframe, gamma=1, utility_type=1)


        spread, min_bound, max_bound = sc_jurek.stabilization_region(self.dataframe, beta = 0.01, gamma = 0.5, utility_type=1)
        # Checking the mean value of spread, min_bound and max_bound.
        self.assertAlmostEqual(np.mean(spread), 0.5328233514, delta=1e-7)
        self.assertAlmostEqual(np.mean(min_bound), 0.511643929, delta=1e-7)
        self.assertAlmostEqual(np.mean(max_bound), 0.549827869, delta=1e-7)

        # Checking the min value of spread, min_bound and max_bound.
        self.assertAlmostEqual(np.min(spread), 0.494787089, delta=1e-7)
        self.assertAlmostEqual(np.min(min_bound), 0.511372451, delta=1e-7)
        self.assertAlmostEqual(np.min(max_bound), 0.546616686, delta=1e-7)

        # Checking the max value of spread, min_bound and max_bound.
        self.assertAlmostEqual(np.max(spread), 0.558869423, delta=1e-7)
        self.assertAlmostEqual(np.max(min_bound), 0.513868404, delta=1e-7)
        self.assertAlmostEqual(np.max(max_bound), 0.550466393, delta=1e-7)


        spread, min_bound, max_bound = sc_jurek.stabilization_region(self.dataframe, beta=0.01, gamma=2, utility_type=2)
        # Checking the mean value of spread, min_bound and max_bound.
        self.assertAlmostEqual(np.mean(spread), 0.5328233514, delta=1e-7)
        self.assertAlmostEqual(np.mean(min_bound), 0.5276749084, delta=1e-7)
        self.assertAlmostEqual(np.mean(max_bound), 0.5850429718, delta=1e-7)

        # Checking the min value of spread, min_bound and max_bound.
        self.assertAlmostEqual(np.min(spread), 0.494787089, delta=1e-7)
        self.assertAlmostEqual(np.min(min_bound), 0.4974942641, delta=1e-7)
        self.assertAlmostEqual(np.min(max_bound), 0.5625208403, delta=1e-7)

        # Checking the max value of spread, min_bound and max_bound.
        self.assertAlmostEqual(np.max(spread), 0.558869423, delta=1e-7)
        self.assertAlmostEqual(np.max(min_bound), 0.5556367547, delta=1e-7)
        self.assertAlmostEqual(np.max(max_bound), 0.6107958587, delta=1e-7)


    def test_optimal_weights_fund_flows(self):
        """
        Tests the optimal weights with fund flows method in the class.
        """

        # Creating an object of the class.
        sc_jurek = OUModelJurek()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=False)

        weights = sc_jurek.optimal_portfolio_weights_fund_flows(self.dataframe, f=0.2, gamma = 0.5)

        self.assertAlmostEqual(np.mean(weights), -2.2710266815, delta=1e-7)
        self.assertAlmostEqual(weights[7], -48.8054214993, delta=1e-7)
        self.assertAlmostEqual(weights[28], -25.2641291580, delta=1e-7)
        self.assertAlmostEqual(weights[-1], 77.1471930191, delta=1e-7)


    def test_private_methods(self):
        """
        Function tests special cases for code coverage.
        """

        # Creating an object of the class.
        sc_jurek = OUModelJurek()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=False)

        # Representative values of parameters.
        time_array = np.arange(0, len(self.dataframe)) * (1/252)
        tau = time_array[-1] - time_array
        c_1 = 0.0221413
        c_2 = -20.59561
        c_3 = 9625.4458424
        disc = -844.234913

        # For utility_type=1.
        sc_jurek.optimal_portfolio_weights(self.dataframe, beta=0.01, gamma=0.5, utility_type=1)

        # Running below cases with gamma values for code coverage.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc_jurek._A_calc_1(tau, c_1, c_2, disc, 0.5)
            sc_jurek._A_calc_1(tau, c_1, c_2, disc, 0.9)
            sc_jurek._B_calc_1(tau, c_1, c_2, c_3, disc, 0.5)
            sc_jurek._B_calc_1(tau, c_1, c_2, c_3, disc, 0.9)

        # For utility_type = 2.
        sc_jurek.optimal_portfolio_weights(self.dataframe, beta=0.01, gamma=0.5, utility_type=2)

        # Running below cases with gamma values for code coverage.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc_jurek._B_calc_2(tau, c_1, c_2, c_3, disc, 0.5)
            sc_jurek._B_calc_2(tau, c_1, c_2, c_3, disc, 0.9)


    @mock.patch("arbitragelab.stochastic_control_approach.ou_model_jurek.plt")
    def test_plotting(self, mock_plt):
        """
        Tests the plot_results method in the class.
        """

        # Creating an object of the class.
        sc_jurek = OUModelJurek()

        # Tests not datetime index exception.
        with self.assertRaises(Exception):
            sc_jurek.plot_results(self.dataframe)

        self.dataframe.index = pd.to_datetime(self.dataframe.index)

        # Tests length of dataframe exception.
        with self.assertRaises(Exception):
            sc_jurek.plot_results(self.dataframe)

        # Using a different dataset with atleast 10 years of data.
        sc_jurek.plot_results(self.shell_rdp_data)

        # Assert plt.figure got called
        assert mock_plt.show.called
