import numpy as np
import pandas as pd
import statsmodels.api as sm
from mlfinlab.statistical_arbitrage import EngleGrangerPortfolio


class MinimumProfitSimulation:
    """
    This is a class that can be used to simulate cointegrated price series pairs.

    The class will generate a price first-order difference time series defined by an AR(1) process,
    a cointegration error series defined by an AR(1) process, and calculate the other price series
    based on the cointegration equation.

    Methods:
        initialize_params(): Set the default values of the parameters for AR(1) processes.
        simulate_ar(params, **kwarg): Simulate an AR(1) process with specified parameters.
        simulate_coint(price_params, coint_params, **kwarg): Simulate two cointegrated price series with
            a specified cointegration coefficient.
        verify_ar(price_matrix): Verify if a group of simulated time series are generated by an AR(1) process,
            the output is the mean and standard deviation of the AR(1) coefficient.
        verify_coint(price_series_x, price_series_y, x_name, y_name):
            Verify if a group of simulated price series pairs are cointegrated by a specific hedge ratio,
            the output is the mean and standard deviation of the cointegration coefficient, beta.
    """

    def __init__(self, ts_num, ts_length):
        """
        Initialize the simulation class.

        Specify the number of time series to be simulated and define the length of each time series.
        Generate a default parameter set with the initialize_params method.
        :param ts_num: (int) Number of time series to simulate
        :param ts_length: (int) Length of each time series to simulate
        """
        self.ts_num = ts_num
        self.ts_length = ts_length
        self.__price_params, self.__coint_params = self.initialize_params()

    @staticmethod
    def initialize_params():
        """
        Initialize the AR(1) process parameter for first-order difference of price series of share S2.
        Initialize the AR(1) process parameter for cointegration error.
        :return:
            price_params: (dict) Necessary parameters for share S2 price simulation
            coint_params: (dict) Necessary parameters for cointegration error simulation
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

    def get_price_params(self):
        """
        Getter for price simulation parameters.
        :return: price_params: (dict) Necessary parameters for share S2 price simulation
        """
        return self.__price_params

    def get_coint_params(self):
        """
        Getter for cointegration error simulation parameters.
        :return: coint_params: (dict) Necessary parameters for cointegration error simulation.
        """
        return self.__coint_params

    def set_price_params(self, param, value):
        """
        Setter for price simulation parameters
        :param param: (str) Parameter dictionary key
        :param value: (float) Parameter value
        :return:
        """
        assert param in self.__price_params, "Parameter doesn't exist!"
        self.__price_params[param] = value

    def set_coint_params(self, param, value):
        """
        Setter for cointegration error simulation parameters
        :param param: (str) Parameter dictionary key
        :param value: (float) Parameter value
        :return:
        """
        assert param in self.__coint_params, "Parameter doesn't exist!"
        self.__coint_params[param] = value

    def simulate_ar(self, params, burn_in=50, use_statsmodel=True):
        """
        Simulate an AR(1) process without using the statsmodel package.
        The AR(1) process is defined as a following recurrence relation.

        y_t = \\mu + \\phi * y_{t-1} + e_t, e_t ~ i.i.d N(0, \\sigma^2)

        :param params: (dict) A parameter dictionary containing AR(1) coefficient, constant trend,
            and white noise variance.
        :param burn_in: (int) The amount of data used to burn in the process
        :param use_statsmodel: (bool) If True, use statsmodel; otherwise, directly calculate recurrence
        :return: ar_process: (np.array) ts_num simulated series generated
        """
        # Store the series
        series_list = []

        # Read the parameters from the dictionary
        try:
            constant_trend = params['constant_trend']
            ar_coeff = params['ar_coeff']
            white_noise_var = params['white_noise_var']
        except KeyError:
            print("Missing crucial parameters. The parameter dictionary should contain the following keys:\n"
                  "1. constant_trend\n"
                  "2. ar_coeff\n"
                  "3. white_noise_var\n"
                  "Call initialize_params() to reset the configuration of the parameters to default.")
            raise

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
            # Setting an initial point. It does not matter as the first few data points are burnt in.
            # We just need to get the recurrence started.
            y0 = np.random.normal()

            # Now set up the recurrence
            series = [y0]
            for i in range(self.ts_length + burn_in):
                y_new = constant_trend + ar_coeff * series[-1] + np.random.normal(0, np.sqrt(white_noise_var))
                series.append(y_new)

            # Reshape the 1-D array into a matrix
            final_series = np.array(series[burn_in:]).reshape(-1, 1)

            # Use hstack to get the full matrix
            series_list.append(final_series)

        return np.hstack(tuple(series_list))

    def _simulate_cointegration(self, price_params, coint_params, initial_price=100.):
        """
        Use the statsmodel to generate two price series that are cointegrated.
        The hedge ratio is defined by beta.

        :param price_params: (dict) Parameter dictionary for share S2 price simulation.
        :param coint_params: (dict) Parameter dictionary for cointegration error simulation.
        :param initial_price: (float) Initial price of share S2.
        :return:
            (np.array, np.array, np.array) Price series of share S1, price series of share S2, and cointegration error.
        """
        # Read the parameters from the param dictionary
        try:
            beta = coint_params['beta']

        except KeyError:
            print("Missing crucial parameters.\n"
                  "Call initialize_params() to reset the configuration of the parameters to default.")
            raise

        share_s2_diff = self.simulate_ar(price_params, use_statsmodel=True)

        # Do a cumulative sum to get share s2 price for each column
        share_s2 = initial_price + np.cumsum(share_s2_diff, axis=0)

        # Now generate the cointegration series
        coint_error = self.simulate_ar(coint_params, use_statsmodel=True)

        # Generate share s1 price according to the cointegration relation
        share_s1 = coint_error - beta * share_s2

        return share_s1, share_s2, coint_error

    def _simulate_cointegration_raw(self, price_params, coint_params, initial_price=100.):
        """
        Use the raw simulation method to generate two price series that are cointegrated.
        The hedge ratio is defined by beta.

        :param price_params: (dict) Parameter dictionary for share S2 price simulation.
        :param coint_params: (dict) Parameter dictionary for cointegration error simulation.
        :param initial_price: (float) Initial price of share S2
        :return:
            (np.array, np.array, np.array) Price series of share S1, price series of share S2, and cointegration error.
        """
        # Read the parameters from the param dictionary
        try:
            beta = coint_params['beta']

        except KeyError:
            print("Missing crucial parameters.\n"
                  "Call initialize_params() to reset the configuration of the parameters to default.")
            raise

        # Generate share S2 price difference with an AR(1) process
        share_s2_diff = self.simulate_ar(price_params, use_statsmodel=False)

        # Sum up to get share S2 price
        share_s2 = initial_price + np.cumsum(share_s2_diff, axis=0)

        # Generate cointegration error with an AR(1) process
        coint_error = self.simulate_ar(coint_params, use_statsmodel=False)

        # Get share S1 price according to the hedge ratio
        share_s1 = coint_error - beta * share_s2

        return share_s1, share_s2, coint_error

    def simulate_coint(self, price_params, coint_params, initial_price, use_statsmodel=False):
        """
        Generate cointegrated price series and cointegration error series.

        :param price_params: (dict) Parameter dictionary for share S2 price simulation.
        :param coint_params: (dict) Parameter dictionary for cointegration error simulation.
        :param initial_price: (float) Starting price of share S2.
        :param use_statsmodel: (bool) Use statsmodel API or use raw method. If True, then statsmodel API will be used.
        :return:
            (np.array, np.array, np.array) Price series of share S1, price series of share S2, and cointegration error.
        """
        if use_statsmodel:
            return self._simulate_cointegration(price_params, coint_params, initial_price=initial_price)

        return self._simulate_cointegration_raw(price_params, coint_params, initial_price=initial_price)

    def verify_ar(self, price_matrix):
        """
        Test function to confirm that the simulated price series is AR(1).

        :param price_matrix: (np.array) A matrix where each column is a hypothetical AR(1) process
        :return:
        ar_coeff_mean: (float) The mean AR(1) coefficient of the process
        ar_coeff_std: (float) The standard deviation of AR(1) coefficient of the process
        """
        # Store all the AR(1) coefficients
        ar_coeff_list = []

        # Use statsmodel to fit the AR(1) process
        for idx in range(self.ts_num):
            # Specify constant trend as the simulated process has one
            ts_fit = sm.tsa.ARMA(price_matrix[:, idx], (1, 0)).fit(trend='c', disp=0)

            # Retrieve the constant trend and the AR(1) coefficient
            const_trend, ar_coeff = ts_fit.params

            # Save the AR(1) coefficients
            ar_coeff_list.append(ar_coeff)

        # Cover the corner case where there is only one series
        if self.ts_num == 1:
            return ar_coeff_list[0], None

        ar_coeff = np.array(ar_coeff_list)
        return ar_coeff.mean(), ar_coeff.std()

    def verify_coint(self, price_series_x, price_series_y, x_name="Share S1", y_name="Share S2"):
        """
        Use Engle-Granger test to verify if the simulated series are cointegrated
        :param price_series_x: (np.array) A matrix where each column is a simulated price series of share S1
        :param price_series_y: (np.array) A matrix where each column is a simulated price series of share S2
        :param x_name: (str) Column name for share S1 column of Engle-Granger input dataframe
        :param y_name: (str) Column name for share S2 column of Engle-Granger input dataframe
        :return:
        beta_mean: Mean of hedge ratio
        beta_std: Standard deviation of hedge ratio
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
