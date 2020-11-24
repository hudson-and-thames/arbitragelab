# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=invalid-name
"""
This module allows simulation of cointegrated time series pairs.
"""

from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from arbitragelab.cointegration_approach.engle_granger import EngleGrangerPortfolio


class CointegrationSimulation:
    """
    This is a class that can be used to simulate cointegrated price series pairs.

    The class will generate a price first-order difference time series defined by an AR(1) process,
    a cointegration error series defined by an AR(1) process, and calculate the other price series
    based on the cointegration equation.
    """

    def __init__(self, ts_num: int, ts_length: int):
        """
        Initialize the simulation class.

        Specify the number of time series to be simulated and define the length of each time series.
        Generate a default parameter set with the initialize_params method.

        :param ts_num: (int) Number of time series to simulate.
        :param ts_length: (int) Length of each time series to simulate.
        """
        self.ts_num = ts_num
        self.ts_length = ts_length
        self.__price_params, self.__coint_params = self.initialize_params()

    @staticmethod
    def initialize_params() -> Tuple[dict, dict]:
        """
        Initialize the default parameters for first-order difference of share S2 price series and cointegration error.

        :return: (dict, dict) Necessary parameters for share S2 price simulation;
            necessary parameters for cointegration error simulation.
        """
        price_params = {
            "ar_coeff": 0.1,
            "white_noise_var": 0.5,
            "constant_trend": 13.
        }
        coint_params = {
            "ar_coeff": 0.2,
            "white_noise_var": 1.,
            "constant_trend": 13.,
            "beta": -0.2
        }
        return price_params, coint_params

    def get_price_params(self) -> dict:
        """
        Getter for price simulation parameters.

        :return: price_params: (dict) Necessary parameters for share S2 price simulation.
        """
        return self.__price_params

    def get_coint_params(self) -> dict:
        """
        Getter for cointegration error simulation parameters.

        :return: coint_params: (dict) Necessary parameters for cointegration error simulation.
        """
        return self.__coint_params

    def set_price_params(self, param: str, value: float):
        """
        Setter for price simulation parameters.

        Change one specific parameter to designated value. Possible parameters are
            ["ar_coeff", "white_noise_var", "constant_trend"].

        :param param: (str) Parameter dictionary key
        :param value: (float) Parameter value
        """
        if param not in self.__price_params:
            raise KeyError("Parameter doesn't exist!")
        self.__price_params[param] = value

    def set_coint_params(self, param: str, value: float):
        """
        Setter for cointegration error simulation parameters.

        Change one specific parameter to designated value. Possible parameters are
            ["ar_coeff", "white_noise_var", "constant_trend", "beta"]

        :param param: (str) Parameter dictionary key
        :param value: (float) Parameter value
        """
        if param not in self.__coint_params:
            raise KeyError("Parameter doesn't exist!")
        self.__coint_params[param] = value

    def load_params(self, params: dict, target: str = "price"):
        """
        Setter for simulation parameters.

        Change the entire parameter sets by loading the dictionary.

        :param params: (dict) Parameter dictionary.
        :param target: (str) Indicate which parameter to load. Possible values are "price" and "coint".
        """
        # Check which parameters to change
        target_types = ('price', 'coint')
        if target not in target_types:
            raise ValueError("Invalid parameter dictionary type. Expect one of: {}".format(target_types))

        # Check if all necessary parameters are in the provided dictionary
        if target == "price":
            default_keys = set(self.__price_params.keys())
        else:
            default_keys = set(self.__coint_params.keys())

        new_keys = set(params.keys())
        if not default_keys <= new_keys:
            missing_keys = default_keys - new_keys
            raise KeyError("Key parameters {} missing!".format(*missing_keys))

        # Set the parameters
        if target == "price":
            self.__price_params = params
        else:
            self.__coint_params = params

    def simulate_ar(self, params: dict, burn_in: int = 50, use_statsmodel: bool = True) -> np.array:
        """
        Simulate an AR(1) process without using the statsmodel package.
        The AR(1) process is defined as the following recurrence relation.

        .. math::
            y_t = \\mu + \\phi y_{t-1} + e_t, \\quad e_t \\sim N(0, \\sigma^2) \\qquad \\mathrm{i.i.d}

        :param params: (dict) A parameter dictionary containing AR(1) coefficient, constant trend,
            and white noise variance.
        :param burn_in: (int) The amount of data used to burn in the process.
        :param use_statsmodel: (bool) If True, use statsmodel;
            otherwise, directly calculate recurrence.
        :return: (np.array) ts_num simulated series generated.
        """
        # Store the series
        series_list = []

        # Read the parameters from the dictionary
        try:
            constant_trend = params['constant_trend']
            ar_coeff = params['ar_coeff']
            white_noise_var = params['white_noise_var']
        except KeyError:

            raise KeyError("Missing crucial parameters. The parameter dictionary should contain"
                           " the following keys:\n"
                           "1. constant_trend\n"
                           "2. ar_coeff\n"
                           "3. white_noise_var\n"
                           "Call initialize_params() to reset the configuration of the "
                           "parameters to default.")

        # If using statsmodel
        if use_statsmodel:
            # Specify AR(1) coefficient
            ar = np.array([1, -ar_coeff])

            # No MA component, but need a constant
            ma = np.array([1])

            # Initialize an ArmaProcess model
            process = sm.tsa.ArmaProcess(ar, ma)

            # Generate the samples
            ar_series = process.generate_sample(nsample=(self.ts_length, self.ts_num),
                                                burnin=burn_in,
                                                scale=np.sqrt(white_noise_var))

            # Add constant trend
            ar_series += constant_trend

            return ar_series

        for _ in range(self.ts_num):
            # Setting an initial point. It does not matter due to the burn-in process.
            # We just need to get the recurrence started.
            series = [np.random.normal()]

            # Now set up the recurrence
            for _ in range(self.ts_length + burn_in):
                y_new = constant_trend + ar_coeff * series[-1] + np.random.normal(0, np.sqrt(white_noise_var))
                series.append(y_new)

            # Reshape the 1-D array into a matrix
            final_series = np.array(series[(burn_in + 1):]).reshape(-1, 1)

            # Use hstack to get the full matrix
            series_list.append(final_series)

        if self.ts_num == 1:
            return series_list[0]
        return np.hstack(tuple(series_list))

    def _simulate_cointegration(self, price_params: dict, coint_params: dict,
                                initial_price: float = 100.) -> Tuple[np.array, np.array, np.array]:
        """
        Use the statsmodel to generate two price series that are cointegrated. The hedge ratio is defined by beta.

        :param price_params: (dict) Parameter dictionary for share S2 price simulation.
        :param coint_params: (dict) Parameter dictionary for cointegration error simulation.
        :param initial_price: (float) Initial price of share S2.
        :return: (np.array, np.array, np.array) Price series of share S1, price series of share S2,
            and cointegration error.
        """
        # Read the parameters from the param dictionary
        beta = coint_params['beta']

        share_s2_diff = self.simulate_ar(price_params, use_statsmodel=True)

        # Do a cumulative sum to get share s2 price for each column
        share_s2 = initial_price + np.cumsum(share_s2_diff, axis=0)

        # Now generate the cointegration series
        coint_error = self.simulate_ar(coint_params, use_statsmodel=True)

        # Generate share s1 price according to the cointegration relation
        share_s1 = coint_error - beta * share_s2

        return share_s1, share_s2, coint_error

    def _simulate_cointegration_raw(self, price_params: dict, coint_params: dict,
                                    initial_price: float = 100.) -> Tuple[np.array, np.array, np.array]:
        """
        Use the raw simulation method to generate two price series that are cointegrated. The hedge ratio is defined by beta.

        :param price_params: (dict) Parameter dictionary for share S2 price simulation.
        :param coint_params: (dict) Parameter dictionary for cointegration error simulation.
        :param initial_price: (float) Initial price of share S2.
        :return: (np.array, np.array, np.array) Price series of share S1, price series of share S2,
            and cointegration error.
        """
        # Read the parameters from the param dictionary
        beta = coint_params['beta']

        # Generate share S2 price difference with an AR(1) process
        share_s2_diff = self.simulate_ar(price_params, use_statsmodel=False)

        # Sum up to get share S2 price
        share_s2 = initial_price + np.cumsum(share_s2_diff, axis=0)

        # Generate cointegration error with an AR(1) process
        coint_error = self.simulate_ar(coint_params, use_statsmodel=False)

        # Get share S1 price according to the hedge ratio
        share_s1 = coint_error - beta * share_s2

        return share_s1, share_s2, coint_error

    def simulate_coint(self, initial_price: float,
                       use_statsmodel: bool = False) -> Tuple[np.array, np.array, np.array]:
        """
        Generate cointegrated price series and cointegration error series.

        :param initial_price: (float) Starting price of share S2.
        :param use_statsmodel: (bool) Use statsmodel API or use raw method.
            If True, then statsmodel API will be used.
        :return: (np.array, np.array, np.array) Price series of share S1, price series of share S2,
            and cointegration error.
        """
        if use_statsmodel:
            return self._simulate_cointegration(self.__price_params,
                                                self.__coint_params,
                                                initial_price=initial_price)

        return self._simulate_cointegration_raw(self.__price_params,
                                                self.__coint_params,
                                                initial_price=initial_price)

    def verify_ar(self, price_matrix: np.array) -> Tuple[float, Optional[float]]:
        """
        Test function to confirm that the simulated price series is an AR(1) process.

        :param price_matrix: (np.array) A matrix where each column is a hypothetical AR(1) process.
        :return: (float, float) The mean AR(1) coefficient of the process;
            the standard deviation of AR(1) coefficient of the process.
        """
        # Store all the AR(1) coefficients
        ar_coeff_list = []

        # Use statsmodel to fit the AR(1) process
        for idx in range(self.ts_num):
            # Specify constant trend as the simulated process has one
            ts_fit = sm.tsa.ARMA(price_matrix[:, idx], (1, 0)).fit(trend='c', disp=0)

            # Retrieve the constant trend and the AR(1) coefficient
            _, ar_coeff = ts_fit.params

            # Save the AR(1) coefficients
            ar_coeff_list.append(ar_coeff)

        # Cover the corner case where there is only one series
        if self.ts_num == 1:
            return ar_coeff_list[0], None

        ar_coeff = np.array(ar_coeff_list)
        return ar_coeff.mean(), ar_coeff.std()

    def verify_coint(self, price_series_x: np.array, price_series_y: np.array,
                     x_name: str = "Share S1", y_name: str = "Share S2") -> Tuple[float, Optional[float]]:
        """
        Use Engle-Granger test to verify if the simulated series are cointegrated.

        :param price_series_x: (np.array) A matrix where each column is a simulated
            price series of share S1.
        :param price_series_y: (np.array) A matrix where each column is a simulated
            price series of share S2.
        :param x_name: (str) Column name for share S1 column of Engle-Granger input dataframe.
        :param y_name: (str) Column name for share S2 column of Engle-Granger input dataframe.
        :return: (float, float) Mean of hedge ratio; standard deviation of hedge ratio
        """
        # List to store the hedge ratios
        betas_list = []

        # Initialize an Engle-Granger test class in mlfinlab
        eg_portfolio = EngleGrangerPortfolio()

        for idx in range(self.ts_num):
            # Get one simulate price series of share s1 and share s2, respectively, from the matrix
            share_s1 = price_series_x[:, idx].reshape(-1, 1)
            share_s2 = price_series_y[:, idx].reshape(-1, 1)

            # Convert the two price series into a pd.DataFrame
            coint_pair_df = pd.DataFrame(np.hstack((share_s1, share_s2)))

            # Rename the columns so we know which column is which
            coint_pair_df.columns = [x_name, y_name]

            # Calculate the hedge ratio
            eg_portfolio.fit(coint_pair_df, add_constant=True)

            # Store the results
            coint_vec = eg_portfolio.cointegration_vectors

            # The Engle-Granger test will always output the hedge ratio to the second column
            betas_list.append(coint_vec[y_name].values[0])

        # Cover the corner case where there is only one series for testing
        if self.ts_num == 1:
            return betas_list[0], None

        # Calculate the standard and mean of the hedge ratio of all simulated series
        betas = np.array(betas_list)
        return betas.mean(), betas.std()

    @staticmethod
    def plot_coint_series(series_x: np.array, series_y: np.array, coint_error: np.array,
                          figw: float = 15., figh: float = 10.):
        """
        Plot the simulated cointegrated series.

        :param series_x: (np.array) Price series of share S1
        :param series_y: (np.array) price series of share S2
        :param coint_error: (np.array) Cointegration error.
        :param figw: (float) Figure width.
        :param figh: (float) Figure height.
        """
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(figw, figh), gridspec_kw={'height_ratios': [2.5, 1]})

        # Plot prices
        ax1.plot(series_x, label="Share S1")
        ax1.plot(series_y, label="Share S2")
        ax1.legend(loc='best', fontsize=12)
        ax1.tick_params(axis='y', labelsize=14)

        # Plot cointegration error
        ax2.plot(coint_error, label='spread')
        ax2.legend(loc='best', fontsize=12)

        return fig
