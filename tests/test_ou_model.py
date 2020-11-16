# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Test functions from the OU-model of the Optimal Mean Reversion module.
"""

import unittest
import os
import numpy as np
import pandas as pd
from arbitragelab.optimal_mean_reversion.ou_model import OrnsteinUhlenbeck

class TestOrnsteinUhlenbeck(unittest.TestCase):
    """
    Test the Ornstein-Uhlenbeck module
    """

    def setUp(self):
        """
        Set the file path for the data and testing variables.
        """

        np.random.seed(0)

        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/gld_gdx_data.csv'
        data = pd.read_csv(self.path)
        data = data.set_index('Date')

        self.dataframe = data[['GLD', 'GDX']]

        # Formatting the input data into np.array of asset prices
        # Time series of two prices
        self.assetprices = np.array(self.dataframe)

        # Data with incorrect dimensions
        self.assetprices_incorrect = np.zeros((4, 3))

        # Create a helper function to construct a portfolio
        asset_helper = self.assetprices.transpose()

        # Constructing portfolio price identical to self.assetprices
        # optimal portfolio
        self.portfolioprices = ((1 /asset_helper[0][0]) * asset_helper[0][:]
                                - (0.212909090909 / asset_helper[1][0]) * asset_helper[1][:])

        # List with testing values for data frequency
        self.test_data_frequency = ["D", "M", "Y", True]

    def test_fit(self):
        """
        Tests that the fit function works correctly with appropriate inputs
        """
        # Creating an object of class
        test = OrnsteinUhlenbeck()

        # Allocating data as pd.DataFrame, discount rates and transaction costs as tuples,
        # defined stop-loss level, data frequency - daily, training interval not defined
        test.fit(self.dataframe, self.test_data_frequency[0],
                 discount_rate=[0.05, 0.05], transaction_cost=[0.02, 0.02],
                 start="8/24/2015", end="12/8/2015",
                 stop_loss=0.2)

        # Retraining the model without using the additionally provided data
        test.fit_to_assets(start="8/24/2015", end="12/8/2015")

        # Testing allocated parameters
        self.assertAlmostEqual(test.delta_t, 0.00396, delta=1e-4)
        np.testing.assert_array_equal(test.r, [0.05, 0.05])
        np.testing.assert_array_equal(test.c, [0.02, 0.02])
        self.assertEqual(test.L, 0.2)

        # Allocating data as pd.DataFrame, discount rates and transaction costs as tuples,
        # defined stop-loss level, data frequency - daily, training interval is defined
        test.fit(self.dataframe, self.test_data_frequency[0],
                 discount_rate=[0.05, 0.05], transaction_cost=[0.02, 0.02],
                 stop_loss=0.2)

        # Testing allocated parameters
        self.assertAlmostEqual(test.delta_t, 0.00396, delta=1e-4)
        np.testing.assert_array_equal(test.r, [0.05, 0.05])
        np.testing.assert_array_equal(test.c, [0.02, 0.02])
        self.assertEqual(test.L, 0.2)

        # Allocating data as np.array, discount rates and transaction costs as tuples,
        # defined stop-loss level, data frequency - daily
        test.fit(self.portfolioprices, self.test_data_frequency[0],
                 discount_rate=[0.05, 0.05], transaction_cost=[0.02, 0.02],
                 stop_loss=0.2)

        # Retraining the model without using the additionally provided data
        test.fit_to_portfolio()

        # Testing allocated parameters
        self.assertAlmostEqual(test.delta_t, 0.00396, delta=1e-4)
        np.testing.assert_array_equal(test.r, [0.05, 0.05])
        np.testing.assert_array_equal(test.c, [0.02, 0.02])
        self.assertEqual(test.L, 0.2)

        # Allocating data as np.array, discount rates and transaction costs as single values,
        # defined stop-loss level, data frequency - monthly
        test.fit(self.assetprices, self.test_data_frequency[1],
                 discount_rate=0.05, transaction_cost=0.02, stop_loss=0.2)

        # Testing allocated parameters
        self.assertAlmostEqual(test.delta_t, 0.0833, delta=1e-4)
        np.testing.assert_array_equal(test.r, [0.05, 0.05])
        np.testing.assert_array_equal(test.c, [0.02, 0.02])
        self.assertEqual(test.L, 0.2)

        # Allocating data as np.array, discount rates and transaction costs as single values,
        # stop-loss level not defined, data frequency - monthly
        test.fit(self.assetprices, self.test_data_frequency[2],
                 discount_rate=0.05, transaction_cost=0.02)

        # Testing allocated parameters
        self.assertEqual(test.delta_t, 1)
        np.testing.assert_array_equal(test.r, [0.05, 0.05])
        np.testing.assert_array_equal(test.c, [0.02, 0.02])
        self.assertIsNone(test.L)

    def test_fit_exeptions(self):
        """
        Tests that exceptions are raised correctly during parameters
        fit process if inputs are wrong.
        """
        # Creating an object of class
        test_exception = OrnsteinUhlenbeck()

        # Testing for wrong parameters type for discount rate
        with self.assertRaises(Exception):
            test_exception.fit(self.assetprices, self.test_data_frequency[1],
                               discount_rate="a", transaction_cost=0.02)

        # Testing for wrong parameters type for transaction cost
        with self.assertRaises(Exception):
            test_exception.fit(self.assetprices, self.test_data_frequency[1],
                               discount_rate=0.5, transaction_cost="a")

        # Testing for wrong stop-loss level type
        with self.assertRaises(Exception):
            test_exception.fit(self.assetprices, self.test_data_frequency[1],
                               discount_rate=0.5, transaction_cost=0.02, stop_loss="a")

        # Testing for wrong data dimensions
        with self.assertRaises(Exception):
            test_exception.fit(self.assetprices_incorrect, self.test_data_frequency[2],
                               discount_rate=0.05, transaction_cost=0.02, stop_loss=0.2)

        # Testing for wrong parameters and data input
        with self.assertRaises(Exception):
            test_exception.fit(self.assetprices_incorrect, self.test_data_frequency[3],
                               discount_rate="a", transaction_cost="a")

    def test_parameters(self):
        """
        Tests if the parameter calculation is correct
        """
        # Creating two objects of OUM class to compare
        portfolio = OrnsteinUhlenbeck()
        assets = OrnsteinUhlenbeck()
        dataframe = OrnsteinUhlenbeck()

        # Allocate data as prices of two assets in a form of pd.DataFrame
        dataframe.fit(self.dataframe, self.test_data_frequency[0],
                      discount_rate=[0.05, 0.05],
                      transaction_cost=[0.02, 0.02], stop_loss=0.02)

        # Allocate data as prices of two assets from np.array
        assets.fit(self.assetprices, self.test_data_frequency[0],
                   discount_rate=[0.05, 0.05],
                   transaction_cost=[0.02, 0.02], stop_loss=0.02)

        # Allocate data using a portfolio identical to self.assetprices optimal portfolio from np.array
        portfolio.fit(self.portfolioprices, self.test_data_frequency[0],
                      discount_rate=[0.05, 0.05], transaction_cost=[0.02, 0.02],
                      stop_loss=0.2)

        # Optimal parameters calculated for different input options of the same data and then
        # tested on template data
        portfolio_parameters = [portfolio.theta, portfolio.mu, portfolio.sigma_square, portfolio.half_life()]

        assets_parameters = [assets.theta, assets.mu, assets.sigma_square, assets.half_life(), assets.B_value]

        dataframe_parameters = [dataframe.theta, dataframe.mu, dataframe.sigma_square, dataframe.half_life(),
                                dataframe.B_value]

        expected_output = [0.71758, 6.306, 0.00698, 0.10992, 0.21291]

        # Testing optimal parameters fit to the portfolio constructed from asset prices from pd.DataFrame
        np.testing.assert_almost_equal(dataframe_parameters, expected_output, decimal=3)

        # Testing optimal parameters fit to the portfolio constructed from asset prices from np.array
        np.testing.assert_almost_equal(assets_parameters, expected_output, decimal=3)

        # Testing optimal parameters fit to the given portfolio from np.array
        np.testing.assert_almost_equal(portfolio_parameters, expected_output[:-1], decimal=3)

    def test_optimal_levels(self):
        """
        Test the result of functions that are responsible for the optimal stopping problem solution
        """
        # Creating two objects of OUM class to compare
        portfolio = OrnsteinUhlenbeck()
        assets = OrnsteinUhlenbeck()

        # Allocate data as prices of two assets
        assets.fit(self.assetprices, self.test_data_frequency[0],
                   discount_rate=0.05, transaction_cost=0.02, stop_loss=0.2)

        # Allocate data using a portfolio identical to self.assetprices optimal portfolio
        portfolio.fit(self.portfolioprices, self.test_data_frequency[0],
                      discount_rate=[0.05, 0.05], transaction_cost=[0.02, 0.02],
                      stop_loss=0.2)


        # Optimal exit and entry levels calculated for different input options of the same data
        # tested on the template data
        optimal_levels_portfolio = [portfolio.optimal_liquidation_level(),
                                    portfolio.optimal_entry_level(),
                                    portfolio.optimal_liquidation_level_stop_loss(),
                                    portfolio.optimal_entry_interval_stop_loss()[0],
                                    portfolio.optimal_entry_interval_stop_loss()[1]]
        optimal_levels_assets = [assets.optimal_liquidation_level(),
                                 assets.optimal_entry_level(),
                                 assets.optimal_entry_interval_stop_loss()[0],
                                 assets.optimal_entry_interval_stop_loss()[1],
                                 assets.optimal_liquidation_level_stop_loss()]

        # Expected values
        expected_optimal_levels_portfolio = [0.7443, 0.651, 0.7443, 0.2066, 0.651]

        expected_optimal_levels_assets = [0.7443, 0.651, 0.2066, 0.651, 0.7443]

        # Testing
        np.testing.assert_almost_equal(optimal_levels_portfolio, expected_optimal_levels_portfolio, decimal=4)
        np.testing.assert_almost_equal(optimal_levels_assets, expected_optimal_levels_assets, decimal=4)

        # Testing the fitness check function
        assets.check_fit()
        # Testing the description function
        assets.description()
        # Testing the description function without stop-loss value
        assets.L = None
        assets.description()

    def test_discounted_values(self):
        """
        Test the result of functions that are calculating discounted expected values
        """

        # Creating two objects of OUM class to compare
        portfolio = OrnsteinUhlenbeck()
        assets = OrnsteinUhlenbeck()

        # Allocate data as prices of two assets
        assets.fit(self.assetprices, self.test_data_frequency[0],
                   discount_rate=0.05, transaction_cost=0.02, stop_loss=0.2)

        # Allocate data using a portfolio identical to self.assetprices optimal portfolio
        portfolio.fit(self.portfolioprices, self.test_data_frequency[0],
                      discount_rate=[0.05, 0.05], transaction_cost=[0.02, 0.02],
                      stop_loss=0.2)

        # Optimal exit and entry discounted values calculated for different input options of the same data
        # are calculated and then tested if they are equal to the template data
        optimal_value_portfolio = [portfolio.V(0.8), portfolio.V_sl(0.8)]

        optimal_value_assets = [assets.V(0.8), assets.V_sl(0.8)]

        # Expected values
        expected_optimal_values = [0.78, 0.78]

        # Testing
        np.testing.assert_almost_equal(optimal_value_portfolio, expected_optimal_values, decimal=5)
        np.testing.assert_almost_equal(optimal_value_assets, expected_optimal_values, decimal=5)

        # Tests OU process generation
        assets.ou_model_simulation(100, 0.6, 12, 0.1, 0.00396)

        # Tests plotting method
        assets.plot_levels(self.dataframe, stop_loss=True)
        assets.plot_levels(self.assetprices, stop_loss=True)
        assets.plot_levels(self.portfolioprices)


    def test_functions_exceptions(self):
        """
        Test the exceptions raised by functions that are responsible for the optimal stopping problem solution
        """
        test = OrnsteinUhlenbeck()
        test_condition = OrnsteinUhlenbeck()

        # Allocation without the stop-loss level
        test.fit(self.assetprices, self.test_data_frequency[0],
                 discount_rate=0.05, transaction_cost=0.02)

        # Testing if exceptions are raised when there is no set stop-loss level in
        # functions that require it
        with self.assertRaises(Exception):
            test.optimal_liquidation_level_stop_loss()
        with self.assertRaises(Exception):
            test.V_sl(0.1)
        with self.assertRaises(Exception):
            test.V_sl_derivative(0.1)
        with self.assertRaises(Exception):
            test.optimal_entry_interval_stop_loss()

        # Allocation with the incorrect stop-loss level regarding the data
        test_condition.fit(self.assetprices, self.test_data_frequency[0],
                           discount_rate=0.05, transaction_cost=0.02,
                           stop_loss=0.7)

        # Testing if exception is raised when there is incorrect stop-loss level regarding the data
        with self.assertRaises(Exception):
            test_condition.optimal_entry_interval_stop_loss()

        # Testing if exception is raised when the data for plotting has incorrect dimensions
        with self.assertRaises(Exception):
            test.plot_levels(self.assetprices_incorrect)
