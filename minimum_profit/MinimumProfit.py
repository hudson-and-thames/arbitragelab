import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from mlfinlab.statistical_arbitrage import JohansenPortfolio, EngleGrangerPortfolio


class MinimumProfit:
    """
    This is a class that optimizes the upper and lower bounds for mean-reversion cointegration pair trading.
    The model assumes the cointegration error follows an AR(1) process and utilizes mean first-passage time to
    determine the optimal levels to initiate trades. The trade will be closed when cointegration error reverts to
    its mean.

    Methods:
        train_test_split(date_cutoff, num_cutoff): A method that splits the price data into a training set and
            a test set according to the cutoff. In-sample simulation can be done when both cutoffs are set to None.
        fit(train_df): Derive the cointegration coefficient, cointegration error, AR(1) cofficient and the fitted
            residual of the AR(1) process.
        optimize(ar_coeff, epsilon_t, ar_resid, horizon, granularity):
            Optimize the upper bound for U-trade by optimizing minimum trade profit
        trade(self, trade_df, upper_bound, minimum_profit, beta, epsilon_t, **kwarg):
            Simulate trades applying the optimal upper bound and lower bound for U-trade and L-trade, respectively.
    """

    def __init__(self, price_s1, price_s2, s1_name="Share S1", s2_name="Share S2"):
        """
        Constructor of the cointegration pair trading optimization class.

        Build a pd.DataFrame of both series for cointegration error calculation.
        Initialize a built-in trade book with position, entry price, P&L, and number of trades information.

        :param price_s1: (pd.Series or np.array) Share S1 price series
        :param price_s2: (pd.Series or np.array) Share S2 price series
        :param s1_name: (str) Share S1 name
        :param s2_name: (str) Share S2 name
        """
        if isinstance(price_s1, np.ndarray) and isinstance(price_s2, np.ndarray):
            # If the inputs are Numpy arrays, reshape and then convert into a DataFrame
            price_s1 = price_s1.reshape(-1, 1)
            price_s2 = price_s2.reshape(-1, 1)
            self.price_df = pd.DataFrame(np.hstack((price_s1, price_s2)))
        elif isinstance(price_s1, pd.Series) and isinstance(price_s2, pd.Series):
            # If the inputs are pandas Series, then just concatenate into a Dataframe
            self.price_df = pd.concat([price_s1, price_s2], axis=1)
        else:
            raise TypeError("Price series have to be numpy.array or pd.Series; "
                            "both series have to be of the same type.")

        # Store the ticker name and rename the columns
        self.price_df.columns = [s1_name, s2_name]
        self._s1_name = s1_name
        self._s2_name = s2_name

        # Record the position and P&L
        self._position = np.array([0., 0.])
        self._entry_price = np.array([0., 0.])
        self._pnl = 0.

        # Record the U-trade and L-trade count
        self._trade_count = np.zeros((2,))

    def train_test_split(self, date_cutoff=None, num_cutoff=None):
        """
        Split the price series into a training set to calculate the cointegration coefficient, beta,
        and a test set to simulate the trades to check trade frequency and PnL.

        Set both cutoff to none to perform an in-sample test.

        :param date_cutoff: (pd.Timestamp) If the price series has a date index then this will be used for split
        :param num_cutoff: (int) Number of periods to include in the training set.
            Could be used for any type of index
        :return:
        train_series: (pd.DataFrame) Training set price series.
        test_series: (pd.DataFrame) Test set price series.
        """
        # If num_cutoff is not None, then date_cutoff should be ignored
        if num_cutoff is not None:
            warnings.warn("Already defined the number of data points included in training set. Date cutoff will be "
                          "ignored.")
            train_series = self.price_df.iloc[:num_cutoff, :]
            test_series = self.price_df.iloc[num_cutoff:, :]
            return train_series, test_series

        # Both cutoff is None, do in-sample test. So training set and test set are the same.
        if date_cutoff is None:
            return self.price_df, self.price_df

        # Verify the index is indeed pd.DatetimeIndex
        assert self.price_df.index.is_all_dates, "Index is not of pd.DatetimeIndex type."

        # Make sure the split point is in between the time range of the data
        min_date = self.price_df.index.min()
        max_date = self.price_df.index.max()

        assert min_date < date_cutoff < max_date, "Date split point is not within time range of the data."

        train_series = self.price_df.loc[:date_cutoff]
        test_series = self.price_df.loc[date_cutoff:]

        return train_series, test_series

    def fit(self, train_df, use_johansen=False):
        """
        Find the cointegration coefficient, beta, and the AR(1) coefficient for cointegration error
        :param train_df: (pd.DataFrame) Training set price series generated by train_test_split function.
        :param use_johansen: (bool) If True, use Johansen to calculate beta; if False, use Engle-Granger.
        :return:
        beta: (float) Cointegration coefficient
        epsilon_t: (pd.Series) Cointegration error
        ar_coeff: (float) AR(1) coefficient
        ar_resid: (np.array) AR(1) fit residual on cointegration error
        """
        # Step 0, calculate hedge ratio and cointegration error
        if use_johansen:
            # Use Johansen test to find the hedge ratio
            jo_portfolio = JohansenPortfolio()
            jo_portfolio.fit(train_df, det_order=0)

            # Retrieve beta
            coint_vec = jo_portfolio.cointegration_vectors.loc[0]

            # Normalize based on the first asset
            coint_vec = coint_vec / coint_vec[0]
            beta = coint_vec[1]

        else:
            # Use Engle-Granger test to find the hedge ratio
            eg_portfolio = EngleGrangerPortfolio()
            eg_portfolio.fit(train_df, add_constant=True)

            # Retrieve beta
            coint_vec = eg_portfolio.cointegration_vectors
            beta = coint_vec[self._s2_name].values[0]

        # Calculate the cointegration error, epsilon_t
        epsilon_t = train_df[self._s1_name] + beta * train_df[self._s2_name]

        # Fit an AR(1) model to find the AR(1) coefficient
        ar_fit = sm.tsa.ARMA(epsilon_t, (1, 0)).fit(trend='c', disp=0)
        _, ar_coeff = ar_fit.params

        return beta, epsilon_t, ar_coeff, ar_fit.resid

    @staticmethod
    def _gaussian_kernel(ar_coeff, integrate_grid, ar_resid):
        """
        Calculate the Gaussian kernel (K(u_i, u_j)) matrix for mean passage time calculation.
        :param ar_coeff: The fitted AR(1) coefficient.
        :param integrate_grid: (np.array) The integration grid with equal separation.
        :param ar_resid: (np.array) The residual obtained from AR(1) fit on cointegration error.
        :return: kernel: (np.array) The Gaussian kernel (K(u_i, u_j)) matrix.
        """
        # Variable integrate_grid is evenly spaced, use np.diff to derive the interval
        grid_h = np.diff(integrate_grid)[0]

        # Generate the weight vector
        len_grid = integrate_grid.shape[0]
        weights = np.repeat(2, len_grid)

        # The start and the end weights 1, not 2
        weights[0] = 1
        weights[-1] = 1

        # Now derive the standard deviation of AR(1) residual, sigma_ksi
        sigma_ksi = ar_resid.std()
        # sigma_ksi = np.sqrt(1 - ar_coeff ** 2) * sigma_epsilon

        # Vectorize the term (u_j - phi * u_i) in the exponential
        exp_term1 = np.tile(integrate_grid, (len_grid, 1))
        exponent = exp_term1 - ar_coeff * integrate_grid.reshape(-1, 1)

        # Calculate the kernel
        kernel = grid_h / (2. * np.sqrt(2 * np.pi) * sigma_ksi) * np.exp(-0.5 / (sigma_ksi ** 2) * np.square(exponent))

        # Multiply the weights
        kernel = np.multiply(kernel, weights.reshape(1, -1))

        return kernel

    def _mean_passage_time(self, lower, upper, ar_coeff, ar_resid, granularity):
        """
        Compute E(\\Tau_{a, b}(y0)), where lower = a, upper = b.
        :param lower: (int) Interval lower bound
        :param upper: (int) Interval upper bound
        :param ar_coeff: (float) AR(1) coefficient
        :param ar_resid: (np.array) The residual obtained from AR(1) fit on cointegration error.
        :param granularity: (float) Summation interval for integration.
        :return: Mean first-passage time over interval [a,b] of an AR(1) process, starting at y0
        """
        # Build the grid for summation
        grid = granularity * np.arange(lower, upper)

        # Calculate the gaussian kernel
        gaussian = self._gaussian_kernel(ar_coeff, grid, ar_resid)

        # Calculate the mean passage time at each grid point
        passage_time = np.linalg.solve(np.eye(gaussian.shape[0]) - gaussian, np.ones(gaussian.shape[0]))

        # Return a pandas.Series indexed by grid points for easy retrieval
        passage_time_df = pd.Series(passage_time, index=grid)
        return passage_time_df

    def optimize(self, ar_coeff, epsilon_t, ar_resid, horizon, granularity=0.01):
        """
        Optimize the upper bound following the optimization procedure in paper.

        :param ar_coeff: (float) AR(1) coefficient of the cointegrated spread
        :param epsilon_t: (pd.Series) Cointegration error
        :param ar_resid: (np.array) AR(1) fit residual on cointegration error.
        :param horizon: (int) Test trading period
        :param granularity: (float) Integration discretization interval, default to 0.01.
        :return:
        optimal_ub: (float) Optimal upper bound
        optimal_td: (float) Optimal trade duration
        optimal_iti: (float) Optimal inter-trades interval
        optimal_mtp: (float) Optimal minimum trade profit
        optimal_num_trades: (float) Optimal number of trades
        """
        minimum_trade_profit = []

        # Use 5 times of the standard deviation of cointegration error as an approximation of infinity
        infinity = np.floor(epsilon_t.std() * 5 / granularity + 1)

        # Generate a sequence of pre-set upper-bounds
        upper_bounds = granularity * np.arange(0, infinity)

        # For trade duration calculation, the integration is on fixed interval [0, inf].
        # Only calculate once to be efficient.
        trade_durations = self._mean_passage_time(0, infinity, ar_coeff, ar_resid, granularity)

        # For each upper bound, calculate minimum total profit
        for ub in upper_bounds:
            # Calculate trade duration
            td = trade_durations.loc[ub]

            # Calculate inter-trade interval.
            # Need to calculate every time as the upper bound is floating
            inter_trade_interval = self._mean_passage_time(-infinity, np.floor(ub / 0.01 + 1),
                                                           ar_coeff, ar_resid, granularity)

            # Retrieve the data at initial state = 0
            iti = inter_trade_interval.loc[0.]

            # Number of trades
            num_trades = horizon / (td + iti) - 1

            # Calculate the minimum trade profit
            mtp = ub * num_trades

            minimum_trade_profit.append((td, iti, mtp, num_trades))

        # Find the optimal upper bound
        minimum_trade_profit = np.array(minimum_trade_profit)

        # According to construction, the mtp is the variable we want to maximize and it was stored in 3rd column
        max_idx = minimum_trade_profit[:, 2].argmax()

        # Retrieve optimal trade duration (TD), optimal inter-trades interval (ITI),
        # optimal minimum trade profit (MTP), and optimal number of trades
        optimal_td, optimal_iti, optimal_mtp, optimal_num_trades = minimum_trade_profit[max_idx, :]

        # Retrieve optimal upper bound
        optimal_ub = upper_bounds[max_idx]

        return optimal_ub, optimal_td, optimal_iti, optimal_mtp, optimal_num_trades

    def trade(self, trade_df, upper_bound, minimum_profit, beta, epsilon_t,
              dollar_constraint=np.Inf, allow_ltrade=True, verbose=False):
        """
        Simulate the trade on the test set after upper bound optimization.

        :param trade_df: (pd.DataFrame) Price series of the two cointegrated assets
        :param upper_bound: (float) Optimized upper bound based on mean passage time optimization
        :param minimum_profit: (float) Optimized minimum profit based on mean passage time optimization
        :param beta: (float) Fitted cointegration coefficient, beta
        :param epsilon_t: (np.array) Cointegration error obtained from training set
        :param dollar_constraint: (float) Available capital for trade
        :param allow_ltrade: (bool) If True, allow the trade simualtor to open the trade when the cointegrated
            spread fall below -upper_bound; otherwise, only fade the spread when the spread price exceeds upper_bound
        :param verbose: (bool) If True, output trade log. Otherwise, output nothing.
        :return:
        trade_count: Number of U-trades and L-trades, in this specific order
        pnl: P&L of all trades
        """
        # Starting a new trading account
        self._trade_count = np.zeros((2,))
        self._pnl = 0

        # Closing condition, which is the mean of the epsilon_t
        closing_cond = epsilon_t.mean()

        # Overbought level to fade the spread, corresponds to U-trades
        overbought = closing_cond + upper_bound

        # Oversold level to fade the spread, corresponds to L-trades
        oversold = closing_cond - upper_bound

        # Step 2, choose integer n > K * abs(beta) / (a-b)
        share_s2_count = np.ceil(minimum_profit * np.abs(beta) / upper_bound)

        # Now calculate the cointegration error for the trade_df
        trade_epsilon_t = trade_df[self._s1_name] + beta * trade_df[self._s2_name]
        trade_df_with_cond = trade_df.assign(coint_error=trade_epsilon_t)

        # U-trade triggers
        trade_df_with_cond = trade_df_with_cond.assign(otc_U=trade_df_with_cond['coint_error'] >= overbought)
        trade_df_with_cond = trade_df_with_cond.assign(ctc_U=trade_df_with_cond['coint_error'] <= closing_cond)

        # L-trade triggers
        trade_df_with_cond = trade_df_with_cond.assign(otc_L=trade_df_with_cond['coint_error'] <= oversold)
        trade_df_with_cond = trade_df_with_cond.assign(ctc_L=trade_df_with_cond['coint_error'] >= closing_cond)

        # Trading periods in the trade_df
        period = trade_df.shape[0]

        # Add a flag to let the simulator know if a U-trade is currently open or a L-trade
        # No open position = 0
        # U-trade = 1
        # L-trade = -1
        current_trade = 0

        # Verbose message output formatting dictionary to indicate trade types
        trade_type = {
            1: "U",
            -1: "L"
        }

        # Start trading
        for i in range(period):
            current_price = trade_df_with_cond[[self._s1_name, self._s2_name]].iloc[i, :].values
            if current_trade == 0:
                # No position, and the opening trade condition is satisfied
                share_s1_count = np.ceil(share_s2_count / abs(beta))

                # Before opening the trade, check if the dollar constraint allows us to open
                position = np.array([share_s1_count, share_s2_count])

                capital_req = np.dot(current_price, position)

                # Capital requirement satisfied, open the position
                if capital_req <= dollar_constraint:
                    # Record the entry price.
                    self._entry_price = current_price

                    # Do we open a U-trade or L-trade?
                    if trade_df_with_cond['otc_U'].iloc[i]:

                        # U-trade, short share S1, long share S2
                        self._position = position * np.array([-1, 1])
                        current_trade = 1

                    elif allow_ltrade and trade_df_with_cond['otc_L'].iloc[i]:

                        # L-trade, long share S1, short share S2
                        self._position = position * np.array([1, -1])
                        current_trade = -1
                    else:
                        # No opening condition met, forward to next day
                        continue

                    if verbose:
                        print("{}-Trade opened! "
                              "Entry: "
                              "{}: {} shares @ ${}, {}: {} shares @ ${}".format(trade_type[current_trade],
                                                                                self._s1_name,
                                                                                self._position[0],
                                                                                self._entry_price[0],
                                                                                self._s2_name,
                                                                                self._position[1],
                                                                                self._entry_price[1]))

                # Make sure the trade will not be closed on the same day (using elif)

            else:
                # We have a trade on
                if current_trade == 1 and trade_df_with_cond['ctc_U'].iloc[i]:
                    # The open trade is a U-trade
                    self._trade_count[0] += 1

                elif current_trade == -1 and trade_df_with_cond['ctc_L'].iloc[i]:
                    # The open trade is a L-trade
                    self._trade_count[1] += 1

                else:
                    # No condition triggered, just forward to next day
                    continue

                # Calculate the PnL
                trade_pnl = np.dot(current_price - self._entry_price, self._position)

                # Accumulate the PnL
                self._pnl += trade_pnl

                # Report
                if verbose:
                    print("Trade closed! "
                          "Exit: "
                          "{}: {} shares @ ${}, {}: {} shares @ ${}\n"
                          "P&L: {}".format(self._s1_name,
                                           -self._position[0],
                                           current_price[0],
                                           self._s2_name,
                                           -self._position[1],
                                           current_price[1],
                                           trade_pnl))
                    print('=' * 30)

                # Clear the trade book.
                self._entry_price = np.zeros((2,))
                self._position = np.zeros((2,))
                current_trade = 0

                # No re-entry the same day (using elif)

        # Final trading result
        if verbose:
            print("Total Trades: {}, P&L: {}".format(self._trade_count, self._pnl))
        return self._trade_count, self._pnl
