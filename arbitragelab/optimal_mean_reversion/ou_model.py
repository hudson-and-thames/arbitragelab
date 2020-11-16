# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

# pylint: disable=missing-module-docstring, invalid-name, too-many-instance-attributes
import warnings
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import scipy.optimize as so
import pandas as pd


class OrnsteinUhlenbeck:
    """
    This class implements the algorithm for solving the optimal stopping problem in
    markets with mean-reverting tendencies based on the Ornstein-Uhlenbeck model
    mentioned in the following publication:'Tim Leung and Xin Li Optimal Mean
    reversion Trading: Mathematical Analysis and Practical Applications(November 26, 2015)'
    <https://www.amazon.com/Optimal-Mean-Reversion-Trading-Mathematical/dp/9814725919>`_

    Constructing a portfolio with mean-reverting properties is usually attempted by
    simultaneously taking a position in two highly correlated or co-moving assets and is
    labeled as "pairs trading". One of the most important problems faced by investors is
    to determine when to open and close a position.

    To find the liquidation and entry price levels we formulate an optimal double-stopping
    problem that gives the optimal entry and exit level rules. Also, a stop-loss
    constraint is incorporated into this trading problem and solutions are also provided
    by this module.
    """

    def __init__(self):

        self.theta = None  # Long-term mean
        self.mu = None  # Speed at which the values will regroup around the long-term mean
        self.sigma_square = None  # The amplitude of randomness in the system
        self.delta_t = None  # Delta between observations, calculated in years
        self.c = None  # Transaction costs for liquidating or entering the position
        self.r = None  # Discount rate at the moment of liquidating or entering the position
        self.L = None  # Stop-loss level
        self.B_value = None  # Optimal ratio between two assets
        self.entry_level = None  # Optimal entry levels without and with the stop-loss in respective order
        self.liquidation_level = None  # Optimal exit levels without and with the stop-loss in respective order
        self.data = None  # Training data provided by the user
        self.training_period = None  # Current training period
        self.mll = None  # Maximum log likelihood

    def fit(self, data, data_frequency, discount_rate, transaction_cost, start=None, end=None,
            stop_loss=None):
        """
        Fits the Ornstein-Uhlenbeck model to given data and assigns the discount rates,
        transaction costs and stop-loss level for further exit or entry-level calculation.

        :param data: (np.array/pd.DataFrame) An array with time series of portfolio prices / An array with
            time series of of two assets prices. The dimensions should be either nx1 or nx2.
        :param data_frequency: (str) Data frequency ["D" - daily, "M" - monthly, "Y" - yearly].
        :param discount_rate: (float/tuple) A discount rate either for both entry and exit time
            or a list/tuple of discount rates with exit rate and entry rate in respective order.
        :param transaction_cost: (float/tuple) A transaction cost either for both entry and exit time
            or a list/tuple of transaction costs with exit cost and entry cost in respective order.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        :param stop_loss: (float/int) A stop-loss level - the position is assumed to be closed
            immediately upon reaching this pre-defined price level.
        """

        # Creating variables for the discount rate and transaction cost
        self.r = [0, 0]
        self.c = [0, 0]

        # Setting delta parameter using data frequency
        self._fit_delta(data_frequency=data_frequency)

        # Setting discount rate
        self.r = self._fit_rate_cost(input_data=discount_rate)

        # Setting transaction cost
        self.c = self._fit_rate_cost(input_data=transaction_cost)

        # Setting stop loss level if it's given as a correct data type
        if stop_loss is None or isinstance(stop_loss, (float, int)):
            self.L = stop_loss
        else:
            raise Exception("Wrong stop-loss level data type. Please use float.")

        # Allocating portfolio parameters
        if len(data.shape) == 1:  # If the input is series of prices of a portfolio
            self.fit_to_portfolio(data=data, start=start, end=end)
        elif data.shape[1] == 2:  # If the input is series of prices of assets
            self.fit_to_assets(data=data, start=start, end=end)
        else:
            raise Exception("The number of dimensions for input data is incorrect. "
                            "Please provide a 1 or 2-dimensional array or dataframe.")

    def _fit_data(self, start, end):
        """
        Allocates all the training data parameters
        and fits the OU model to this data.

        Helper function used in self.fit().

        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        :return: (np.array) An array of prices to construct a portfolio from.
        """

        # Checking the training data type
        if isinstance(self.data, np.ndarray):
            self.data = self.data.transpose()
            data = self.data

        else:

            # Checking if the starting and ending timestamps were specified
            if all(timestamp is not None for timestamp in [start, end]):

                # Selecting a slice of the dataframe chosen by the user
                data = self.data.loc[start:end]
                # Setting the training interval
                self.training_period = [start, end]

            else:

                data = self.data
                self.training_period = [self.data.first_valid_index(),
                                        self.data.last_valid_index()]

            # Transforming pd.Dataframe into a numpy array
            data = data.to_numpy().transpose()

        return data

    @staticmethod
    def _fit_rate_cost(input_data):
        """
        Sets the value for cost and rate parameters.

        Helper function used in self.fit().

        :param input_data: (float/tuple) Input for cost or rate.
        :return: A tuple of two elements with allocated data for cost/rate
        """

        # If given a single value, it's duplicated for exit and entry levels
        if isinstance(input_data, float):
            parameters = [input_data, input_data]

        # If given two values, they are treated as data for exit and entry levels
        elif isinstance(input_data, (tuple, list)) and len(input_data) == 2:
            parameters = input_data
        else:
            raise Exception("Wrong discount rate or transaction cost data type. "
                            "Please use float or tuple with 2 elements.")

        return parameters

    def _fit_delta(self, data_frequency):
        """
        Sets the value of the delta-t parameter,
        depending on data frequency input.

        Helper function used in self.fit().

        :param data_frequency: (str) Data frequency
            ["D" - daily, "M" - monthly, "Y" - yearly].
        """

        if data_frequency == "D":
            self.delta_t = 1 / 252
        elif data_frequency == "M":
            self.delta_t = 1 / 12
        elif data_frequency == "Y":
            self.delta_t = 1
        else:
            raise Exception("Incorrect data frequency. "
                            "Please use one of the options [\"D\", \"M\", \"Y\"].")

    def fit_to_portfolio(self, data=None, start=None, end=None):
        """
        Fits the Ornstein-Uhlenbeck model to time series
        for portfolio prices.

        :param data: (np.array) All given prices of two assets to construct a portfolio from.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        """
        # Nullifying the training interval data
        self.training_period = [None, None]

        # Setting the new training data pool
        if data is not None:
            self.data = data

        # Getting the needed data
        portfolio = self._fit_data(start, end)

        # Nullifying optimal entry and exit values during model retraining
        self.entry_level = [None, None]
        self.liquidation_level = [None, None]

        # Fitting the model
        parameters = self.optimal_coefficients(portfolio)

        # Setting the OU model parameters
        self.theta = parameters[0]
        self.mu = parameters[1]
        self.sigma_square = parameters[2]
        self.mll = parameters[3]

    @staticmethod
    def portfolio_from_prices(prices, b_variable):
        """
        Constructs a portfolio based on two given asset prices
        and the relative amount of investment for one of them.

        :param prices: (np.array) An array of prices of the two assets
            used to create a portfolio.
        :param b_variable: (float) A coefficient representing the investment.
            into the second asset, investing into the first one equals one.
        :return: (np.array) Portfolio prices. (p. 11)
        """

        # Calculated as: alpha * Asset_1 - beta * Asset_2
        portfolio_price = ((1 / prices[0][0]) * prices[0][:]
                           - (b_variable / prices[1][0]) * prices[1][:])

        return portfolio_price

    def half_life(self):
        """
        Returns the half-life of the fitted OU process. Half-life stands for the average time that
        it takes for the process to revert to its long term mean on a half of its initial deviation.

        :return: (float) Half-life of the fitted OU process
        """
        # Calculating the half-life
        output = np.log(2) / self.mu

        return output

    def fit_to_assets(self, data=None, start=None, end=None):
        """
        Creates the optimal portfolio in terms of Ornstein-Uhlenbeck model
        from two given time series for asset prices and fits the values
        of the model's parameters. (p.13)

        :param data: (np.array) All given prices of two assets to construct a portfolio from.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        """
        # Nullifying the training interval data
        self.training_period = [None, None]

        # Setting the new training data pool
        if data is not None:
            self.data = data

        # Getting the needed data
        prices = self._fit_data(start, end)

        # Nullifying optimal entry and exit values during model retraining
        self.entry_level = [None, None]
        self.liquidation_level = [None, None]

        # Lambda function that calculates the optimal OU model coefficients
        # for the portfolio constructed from given prices and any given
        # coefficient B_value
        compute_coefficients = lambda x: self.optimal_coefficients(self.portfolio_from_prices(prices, x))

        # Speeding up the calculations
        vectorized = np.vectorize(compute_coefficients)
        linspace = np.linspace(.001, 1, 100)
        res = vectorized(linspace)

        # Picking the argmax of beta
        index = res[3].argmax()

        # Setting the OU model parameters
        self.theta = res[0][index]
        self.mu = res[1][index]
        self.sigma_square = res[2][index]
        self.B_value = linspace[index]
        self.mll = res[3][index]

    def ou_model_simulation(self, n, theta_given=None, mu_given=None, sigma_given=None, delta_t_given=None):
        """
        Simulates values of an OU process with given parameters or parameters fitted to our data

        :param n: (int) Number of simulated values.
        :param theta_given: (float) Long-term mean.
        :param mu_given: (float) Mean revesion speed.
        :param sigma_given: (float) The amplitude of randomness in the system.
        :param delta_t_given: (float) Delta between observations, calculated in years.
        :return: (np.array) simulated portfolio prices.
        """
        # Initializing the variable for process values
        x = np.zeros(n)

        # Checking whether to use given parameters or parameters of the fitted model

        if all(param is not None for param in
               [theta_given, mu_given, sigma_given, delta_t_given]):  # Use given data parameters
            x[0] = theta_given
            theta = theta_given
            mu = mu_given
            sigma = sigma_given
            delta_t = delta_t_given

        else:  # Use fitted data parameters
            x[0] = self.theta
            theta = self.theta
            mu = self.mu
            sigma = np.sqrt(self.sigma_square)
            delta_t = self.delta_t

        # Simulating the OU process values
        for i in range(n - 1):
            x[i + 1] = (x[i] + mu * (theta - x[i]) * delta_t
                        + sigma * np.sqrt(delta_t) * np.random.randn())

        return x

    def optimal_coefficients(self, portfolio):
        """
        Finds the optimal Ornstein-Uhlenbeck model coefficients depending
        on the portfolio prices time series given.(p.13)

        :param portfolio: (np.array) Portfolio prices.
        :return: (tuple) Optimal parameters (theta, mu, sigma_square, max_LL)
        """

        # Setting bounds
        # Theta  R, mu > 0, sigma_squared > 0
        bounds = ((None, None), (1e-5, None), (1e-5, None))

        theta_init = np.mean(portfolio)

        # Initial guesses for theta, mu, sigma
        initial_guess = np.array((theta_init, 100, 100))

        result = so.minimize(self._compute_log_likelihood, initial_guess,
                             args=(portfolio, self.delta_t), bounds=bounds)

        # Unpacking optimal values
        theta, mu, sigma_square = result.x

        # Undo negation
        max_log_likelihood = -result.fun

        return theta, mu, sigma_square, max_log_likelihood

    def check_fit(self):
        """
        Shows the parameters of the OU model fitted to the data and parameters achieved
        from the simulated process with the fitted coefficients for further assessment. The main
        incentive is to have simulated max log-likelihood (mll) function close to a fitted one.
        If the simulated mll is much greater it means that data provided to
        the model is not good enough to be modeled by an OU process.

        Note: The check can occasionally return numbers significantly different from the fitted mll
        due to the random nature of the simulated process and the possibility of outliers. It is advised
        to perform the check multiple times and if it's consistently showing very different values of mll
        you might suspect the unsuitability of your data.

        :return: (pd.Dataframe) Values for fitted and simulated OU model parameters
        """
        # Calculating portfolio values to calculate maximized average log-likelihood
        portfolio = self._fit_data(self.training_period[0], self.training_period[1]).transpose()

        # Simulate OU process with the same coefficients as fitted module's
        simulation_data = self.ou_model_simulation(portfolio.shape[0])

        # Fitting optimal coefficients to simulated model
        simulation_parameters = self.optimal_coefficients(simulation_data)

        # Finding an average simulated max log-likelihood function value
        mll_simulated = np.array([self.optimal_coefficients(self.ou_model_simulation(portfolio.shape[0]))[3]
                                  for i in range(100)]).mean()

        # Creating the output DataFrame
        output = pd.DataFrame({'Theta': [self.theta, simulation_parameters[0]],
                               'Mu': [self.mu, simulation_parameters[1]],
                               'Sigma': [np.sqrt(self.sigma_square), np.sqrt(simulation_parameters[2])],
                               'Max log-likelihood': [self.mll, mll_simulated]},
                              index=['empirical', 'simulated'])

        return output

    @staticmethod
    def _compute_log_likelihood(params, *args):
        """
        Computes the average Log Likelihood. (p.13)

        :param params: (tuple) A tuple of three elements representing theta, mu and sigma_squared.
        :param args: (tuple) All other values that to be passed to self._compute_log_likelihood()
        :return: (float) The average log likelihood from given parameters.
        """

        # Setting given parameters
        theta, mu, sigma_squared = params
        X, dt = args
        n = len(X)

        # Calculating log likelihood
        sigma_tilde_squared = sigma_squared * (1 - np.exp(-2 * mu * dt)) / (2 * mu)

        summation_term = sum((X[1:] - X[:-1] * np.exp(-mu * dt) - theta * (1 - np.exp(-mu * dt))) ** 2)

        summation_term = -summation_term / (2 * n * sigma_tilde_squared)

        log_likelihood = (-np.log(2 * np.pi) / 2) \
                         + (-np.log(np.sqrt(sigma_tilde_squared))) \
                         + summation_term

        return -log_likelihood

    def description(self):
        """

        Returns all the general parameters of the model, training interval timestamps if provided, the goodness of fit,
        allocated trading costs and discount rates, stop-loss level, beta, which stands for the optimal ratio between
        two assets in the created portfolio, and optimal levels calculated. If the stop-loss level was given optimal
        levels that account for stop-loss would be added to the list.

        :return: (pd.Series) Summary data for all model parameters and optimal levels.
        """
        # Calling _fit_data to create a helper variable
        portfolio = self._fit_data(self.training_period[0], self.training_period[1]).transpose()

        # Calculate the average simulated max log-likelihood
        simulated_mll = np.array(
            [self.optimal_coefficients(self.ou_model_simulation(portfolio.shape[0]))[3] for i in range(100)])

        # Calculating the default data values
        data = [self.training_period, self.theta, self.mu, np.sqrt(self.sigma_square), self.mll - simulated_mll.mean(),
                self.r, self.c, self.L, self.B_value,
                self.optimal_entry_level(), self.optimal_liquidation_level()]
        # Setting the names for the data indexes
        index = ['training period', 'long-term mean', 'speed of reversion', 'volatility', 'fitting error',
                 'discount rates', 'transaction costs', 'stop-loss level', 'beta',
                 'optimal entry level', 'optimal liquidation level']

        # If stop-loss level is set - account for additional values
        if self.L is not None:
            data.extend([self.optimal_entry_interval_stop_loss(), self.optimal_liquidation_level()])
            index.extend(['optimal entry interval [sl]', 'optimal liquidation level [sl]'])

        # Combine data and indexes into the pandas Series
        output = pd.Series(data=data, index=index)

        return output

    def plot_levels(self, data, stop_loss=False):
        """
        Plots the found optimal exit and entry levels on the graph
        alongside with the given data.

        :param data: (np.array/pd.DataFrame) Time series of portfolio prices /
            time series of of two assets prices of size (n x 2).
        :param stop_loss: (bool) A flag whether to take stop-loss level into account.
            when showcasing the results.
        :return: (plt.Figure) Figure with optimal exit and entry levels.
        """

        # Constructing proper input

        if len(data.shape) == 1:  # If using portfolio prices
            portfolio = data
        elif data.shape[1] == 2:  # If using two assets prices
            # Checking the data type before preprocessing
            if isinstance(data, pd.DataFrame):
                # Transforming pd.Dataframe into a numpy array
                data = data.to_numpy().transpose()
            else:
                data = data.transpose()
            # Creating a portfolio with previously found optimal ratio
            portfolio = self.portfolio_from_prices(prices=data, b_variable=self.B_value)
        else:
            raise Exception("The number of dimensions for input data is incorrect. "
                            "Please provide a 1 or 2-dimensional array or dataframe.")

        if stop_loss:
            # Plotting entry and exit levels calculated by default
            fig, graph = plt.subplots(2, 1)
            graph[0].plot(portfolio, label='portfolio price')
            graph[0].axhline(self.optimal_liquidation_level(), label="optimal liquidation level",
                             linestyle='--', color='red')
            graph[0].axhline(self.optimal_entry_level(), label="optimal entry level",
                             linestyle=':', color='green')
            graph[0].legend()
            graph[0].set_title('Default optimal levels')

            # Plotting the entry level and exit interval calculated taking into account the stop-loss level,
            # and the stop-loss level itself
            graph[1].plot(portfolio, label='portfolio price')
            graph[1].axhline(self.optimal_liquidation_level_stop_loss(), label="optimal liquidation level",
                             linestyle='--', color='red')
            graph[1].axhline(self.optimal_entry_interval_stop_loss()[0], label="optimal entry level",
                             linestyle=':', color='green')
            graph[1].axhline(self.optimal_entry_interval_stop_loss()[1],
                             linestyle=':', color='green')
            graph[1].axhline(self.L, label='stop-loss',
                             linestyle='-', color='black')
            graph[1].axhspan(self.optimal_entry_interval_stop_loss()[0], self.optimal_entry_interval_stop_loss()[1],
                             facecolor='#C5C9A4', alpha=0.2)
            graph[1].legend()
            graph[1].set_title('Optimal levels calculated with respect to stop-loss level')

        else:
            fig = plt.figure()
            plt.plot(portfolio, label='portfolio price')
            plt.axhline(self.optimal_liquidation_level(), label="optimal liquidation level",
                        linestyle='--', color='red')
            plt.axhline(self.optimal_entry_level(), label="optimal entry level",
                        linestyle=':', color='green')
            plt.xlabel('t')
            plt.ylabel('Price')
            plt.legend()

        return fig

    def _F(self, price, rate):
        """
        Calculates helper function to further define the exit/enter level. (p.18)

        :param price: (float) Portfolio price.
        :param rate: (float) Discounting rate.
        :return: (float) Value of F function.
        """

        # Function to integrate
        f_func = lambda u: (pow(u, (rate / self.mu - 1))
                            * np.exp(np.sqrt(2 * self.mu / self.sigma_square)
                                     * (price - self.theta) * u - u ** 2 / 2))

        # Integrating over positive values
        with warnings.catch_warnings():  # Silencing specific IntegrationWarning
            warnings.filterwarnings('ignore', r'The algorithm does not converge')
            calculated_f = quad(f_func, 0, np.inf)[0]

        return calculated_f

    def _F_derivative(self, price, rate, h=1e-4):
        """
        Calculates a derivative with respect to price of a helper function
        to further define the exit/enter level.

        :param price: (float) Portfolio price.
        :param rate: (float) Discounting rate.
        :param h: (float) Delta step to use to calculate derivative.
        :return: (float) Value of F derivative function.
        """

        # Numerically calculating the derivative
        calculated_f_d = (self._F(price + h, rate) - self._F(price, rate)) / h

        return calculated_f_d

    def _G(self, price, rate):
        """
        Calculates helper function to further define the exit/enter level. (p.18)

        :param price: (float) Portfolio price.
        :param rate: (float) Discounting rate.
        :return: (float) Value of G function.
        """

        # Function to integrate
        g_func = lambda u: (pow(u, (rate / self.mu - 1))
                            * np.exp(np.sqrt(2 * self.mu / self.sigma_square)
                                     * (self.theta - price) * u - u ** 2 / 2))

        # Integrating over positive values
        with warnings.catch_warnings():  # Silencing specific IntegrationWarning
            warnings.filterwarnings('ignore', r'The algorithm does not converge')
            calculated_g = quad(g_func, 0, np.inf)[0]

        return calculated_g

    def _G_derivative(self, price, rate, h=1e-4):
        """
        Calculate a derivative with respect to price to a helper function to
        further define the exit/enter level.

        :param price: (float) Portfolio price.
        :param rate: (float) Discounting rate.
        :param h: (float) Delta step to use to calculate derivative.
        :return: (float) Value of G derivative function.
        """

        # Numerically calculating the derivative
        calculated_g_d = (self._G(price + h, rate) - self._G(price, rate)) / h

        return calculated_g_d

    def V(self, price):
        """
        Calculates the expected discounted value of liquidation of the position. (p.23)

        :param price: (float) Portfolio value.
        :return: (float) Expected discounted liquidation value.
        """

        # Getting optimal liquidation level
        liquidation_level = self.optimal_liquidation_level()

        # Value of the V function
        if price < liquidation_level:
            output = ((liquidation_level - self.c[0])
                      * self._F(price, self.r[0])
                      / self._F(liquidation_level, self.r[0]))
        else:
            output = price - self.c[0]

        return output

    def _V_derivative(self, price, h=1e-4):
        """
        Calculates the derivative of the expected discounted value of
        liquidation of the position.

        :param price: (float) Portfolio value.
        :param h: (float) Delta step to use to calculate derivative.
        :return: (float) Value of V derivative function.
        """

        # Numerically calculating the derivative
        output = (self.V(price + h) - self.V(price)) / h

        return output

    def optimal_liquidation_level(self):
        """
        Calculates the optimal liquidation portfolio level. (p.23)

        :return: (float) Optimal liquidation portfolio level.
        """

        # If the liquidation level wasn't calculated before, setting it
        if self.liquidation_level[0] is None:

            equation = lambda price: (self._F(price, self.r[0]) - (price - self.c[0])
                                      * self._F_derivative(price, self.r[0]))

            bracket = [self.theta - 6 * np.sqrt(self.sigma_square),
                       self.theta + 6 * np.sqrt(self.sigma_square)]

            sol = root_scalar(equation, bracket=bracket)

            output = sol.root

            self.liquidation_level[0] = output

        # If was pre-calculated, using it
        else:

            output = self.liquidation_level[0]

        return output

    def optimal_entry_level(self):
        """
        Calculates the optimal entry portfolio level. (p.27)

        :return: (float) Optimal entry portfolio level.
        """

        # If the entry level wasn't calculated before, setting it
        if self.entry_level[0] is None:

            equation = lambda price: (self._G(price, self.r[1])
                                      * (self._V_derivative(price) - 1)
                                      - self._G_derivative(price, self.r[1])
                                      * (self.V(price) - price - self.c[1]))

            bracket = [self.theta - 6 * np.sqrt(self.sigma_square),
                       self.theta + 6 * np.sqrt(self.sigma_square)]

            sol = root_scalar(equation, bracket=bracket)

            output = sol.root

            self.entry_level[0] = output

        # If was pre-calculated, using it
        else:

            output = self.entry_level[0]

        return output

    def _C(self):
        """
        Calculates helper function to further define the exit/enter
        level with a stop-loss level. (p.31)

        :return: (float) Value of C function.
        """

        # Setting liquidation level with respect to stop-loss
        liquidation_sl = self.optimal_liquidation_level_stop_loss()

        # Calculating C function value
        calculated_c = (((liquidation_sl - self.c[0])
                         * self._G(self.L, self.r[0])
                         - (self.L - self.c[0])
                         * self._G(liquidation_sl, self.r[0]))
                        / (self._F(liquidation_sl, self.r[0])
                           * self._G(self.L, self.r[0])
                           - self._F(self.L, self.r[0])
                           * self._G(liquidation_sl, self.r[0])))

        return calculated_c

    def _D(self):
        """
        Calculates helper function to further define the exit/enter level
        with a stop-loss level. (p.31)

        :return: (float) Value of D function.
        """

        # Setting liquidation level with respect to stop-loss
        liquidation_sl = self.optimal_liquidation_level_stop_loss()

        # Calculating D function value
        calculated_d = (((self.L - self.c[0])
                         * self._F(liquidation_sl, self.r[0])
                         - (liquidation_sl - self.c[0])
                         * self._F(self.L, self.r[0]))
                        / (self._F(liquidation_sl, self.r[0])
                           * self._G(self.L, self.r[0])
                           - self._F(self.L, self.r[0])
                           * self._G(liquidation_sl, self.r[0])))

        return calculated_d

    def V_sl(self, price):
        """
        Calculates the expected discounted value of liquidation of the position
        considering the stop-loss level. (p. 31)

        :param price: (float) Portfolio value.
        :return: (float) Expected discounted value of liquidating the position
            considering the stop-loss level.
        """

        # Checking if the sl level was allocated
        if self.L is None:
            raise Exception("To use this function stop-loss level must be allocated.")

        liquidation_sl = self.optimal_liquidation_level_stop_loss()

        # V_sl function value
        if (self.L < price < liquidation_sl):
            output = self._C() * self._F(price, self.r[0]) \
                     + self._D() * self._G(price, self.r[0])
        else:
            output = price - self.c[0]

        return output

    def _V_sl_derivative(self, price, h=1e-4):
        """
        Calculates the derivative of the expected discounted value of liquidation
        of the position considering the stop-loss level.

        :param price: (float) Portfolio value.
        :param h: (float) Delta step to use to calculate derivative.
        :return: (float) Expected discounted value of liquidating the position
            considering the stop-loss level.
        """

        # Numerically calculating the derivative
        output = (self.V_sl(price + h) - self.V_sl(price)) / h

        return output

    def optimal_liquidation_level_stop_loss(self):
        """
        Calculates the optimal liquidation portfolio level considering the stop-loss level. (p.31)

        :return: (float) Optimal liquidation portfolio level considering the stop-loss.
        """

        # Checking if the sl level was allocated
        if self.L is None:
            raise Exception("To use this function stop-loss level must be allocated.")

        # If the liquidation level wasn't calculated before, set it
        if self.liquidation_level[1] is None:

            # Calculating three sub-parts of the equation
            a_var = lambda price: ((self.L - self.c[0])
                                   * self._G(price, self.r[0])
                                   - (price - self.c[0])
                                   * self._G(self.L, self.r[0])) \
                                  * self._F_derivative(price, self.r[0])

            b_var = lambda price: ((price - self.c[0])
                                   * self._F(self.L, self.r[0])
                                   - (self.L - self.c[0])
                                   * self._F(price, self.r[0])) \
                                  * self._G_derivative(price, self.r[0])

            c_var = lambda price: (self._G(price, self.r[0])
                                   * self._F(self.L, self.r[0])
                                   - self._G(self.L, self.r[0])
                                   * self._F(price, self.r[0]))

            # Final equation
            equation = lambda price: a_var(price) + b_var(price) - c_var(price)

            bracket = [self.theta - 6 * np.sqrt(self.sigma_square),
                       self.theta + 6 * np.sqrt(self.sigma_square)]

            sol = root_scalar(equation, bracket=bracket)

            # The root is the optimal liquidation level considering the stop-loss level
            output = sol.root

            self.liquidation_level[1] = output

        else:

            output = self.liquidation_level[1]

        return output

    def optimal_entry_interval_stop_loss(self):
        """
        Calculates the optimal entry portfolio interval considering the stop-loss level. (p.35)

        :return: (tuple) Optimal entry portfolio interval considering the stop-loss.
        """

        # Checking if the sl level was allocated
        if self.L is None:
            raise Exception("To use this function stop-loss level must be allocated.")

        # Checking for the necessary condition
        if not self._parameter_check():
            raise Exception("Please adjust your stop-loss level")

        # If the entry level wasn't calculated before, set it
        if self.entry_level[1] is None:

            equation1 = lambda price: (self._F(price, self.r[1]) *
                                       (self._V_sl_derivative(price) - 1)
                                       - self._F_derivative(price, self.r[1])
                                       * (self.V_sl(price) - price - self.c[1]))

            equation2 = lambda price: (self._G(price, self.r[1])
                                       * (self._V_sl_derivative(price) - 1)
                                       - self._G_derivative(price, self.r[1])
                                       * (self.V_sl(price) - price - self.c[1]))

            # Set the liquidation level to previously calculated value
            b = self.liquidation_level[1]

            bracket = [self.L, b]

            # Solving the equations
            sol2 = root_scalar(equation2, bracket=bracket)

            sol1 = root_scalar(equation1, bracket=[self.L, sol2.root])

            output = [round(sol1.root, 5), round(sol2.root, 5)]

            self.entry_level[1] = output

        else:

            output = self.entry_level[1]

        return output

    def _parameter_check(self):
        """
        Checks if fitted parameters satisfy the necessary condition to calculate
        optimal entry level accounting for stop-loss. (p.34)

        Condition:
        sup_x{V_L(x) - x - cb} > 0

        :return: (bool) The result of the check.
        """

        # Setting bounds for the x variable
        bounds = ((None, None),)

        # Setting the negated value of goal function because scipy.optimize
        # doesn't have maximization function
        func = lambda x: -(self.V_sl(x) - x - self.c[1])

        # Initial guesses for value of x
        initial_guess = (self.theta - np.sqrt(self.sigma_square))

        # Minimization of the negated function to maximize goal function
        result = so.minimize(func, initial_guess, bounds=bounds)

        # Testing the condition
        output = -result.fun > 0

        return output
