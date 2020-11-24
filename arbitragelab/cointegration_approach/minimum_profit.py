# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=invalid-name, too-many-arguments
"""
This module optimizes the upper and lower bounds for mean-reversion cointegration pair trading
and generates the corresponding trading signal.
"""

import sys
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from arbitragelab.cointegration_approach.engle_granger import EngleGrangerPortfolio
from arbitragelab.cointegration_approach.johansen import JohansenPortfolio


class MinimumProfit:
    """
    This is a class that optimizes the upper and lower bounds for mean-reversion cointegration
    pair trading.

    The model assumes the cointegration error follows an AR(1) process and utilizes
    mean first-passage time to determine the optimal levels to initiate trades.
    The trade will be closed when cointegration error reverts to its mean.
    """

    def __init__(self, price_df: pd.DataFrame):
        """
        Constructor of the cointegration pair trading optimization class.

        :param price_df: (pd.DataFrame) Price series dataframe which contains both series.
        """

        # Store the ticker name and rename the columns
        if price_df.shape[1] != 2:
            raise Exception("Data Format Error. Should only contain two price series.")
        self.price_df = price_df

        # Store the train and trade dataset for future use
        self._train_df = None
        self._trade_df = None

    def train_test_split(self, date_cutoff: Optional[pd.Timestamp] = None,
                         num_cutoff: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the price series into a training set to calculate the cointegration coefficient, beta,
        and a test set to simulate the trades to check trade frequency and PnL.

        Set both cutoff to none to perform an in-sample test.

        :param date_cutoff: (pd.Timestamp) If the price series has a date index then this will be used for split.
        :param num_cutoff: (int) Number of periods to include in the training set (could be used for any type of index).
        :return: (pd.DataFrame, pd.DataFrame) Training set price series; test set price series.
        """
        # If num_cutoff is not None, then date_cutoff should be ignored
        if num_cutoff is not None:
            warnings.warn(
                "Already defined the number of data points included in training set. Date cutoff will be ignored.")
            train_series = self.price_df.iloc[:num_cutoff, :]
            test_series = self.price_df.iloc[num_cutoff:, :]

            self._train_df, self._trade_df = train_series, test_series
            return train_series, test_series

        # Both cutoff is None, do in-sample test. So training set and test set are the same.
        if date_cutoff is None:
            self._train_df, self._trade_df = self.price_df, self.price_df
            return self.price_df, self.price_df

        # Verify the index is indeed pd.DatetimeIndex
        assert self.price_df.index.is_all_dates, "Index is not of pd.DatetimeIndex type."

        # Make sure the split point is in between the time range of the data
        min_date = self.price_df.index.min()
        max_date = self.price_df.index.max()

        assert min_date < date_cutoff < max_date, "Date split point is not within time range of the data."

        train_series = self.price_df.loc[:date_cutoff]
        test_series = self.price_df.loc[date_cutoff:]

        self._train_df, self._trade_df = train_series, test_series
        return train_series, test_series

    def fit(self, sig_level: str = "95%",
            use_johansen: bool = False) -> Tuple[float, pd.Series, float, np.array]:
        """
        Find the cointegration coefficient, beta, and the AR(1) coefficient for cointegration error.

        .. note::

            Cointegration of the price series is crucial to the success of the optimization. The strategy performance
            could be affected if the price series are not cointegrated at 95% level.

        :param sig_level: (str) Cointegration test significance level. Possible options are "90%", "95%", and "99%".
        :param use_johansen: (bool) If True, use Johansen to calculate beta;
            if False, use Engle-Granger.
        :return: (float, pd.Series, float, np.array) Cointegration coefficient, beta;
            Cointegration error, epsilon_t; AR(1) coefficient;
            AR(1) fit residual on cointegration error.
        """
        if sig_level not in ['90%', '95%', '99%']:
            raise ValueError("Significance level can only be the following:\n "
                             "90%, 95%, or 99%.\n Please check the input.")

        # Calculate hedge ratio and cointegration error
        if use_johansen:
            # Use Johansen test to find the hedge ratio
            jo_portfolio = JohansenPortfolio()
            jo_portfolio.fit(self._train_df, det_order=0)

            # Check eigenvalue and trace statistics to see if the pairs are cointegrated at 90% level.
            eigen_stats = jo_portfolio.johansen_eigen_statistic
            trace_stats = jo_portfolio.johansen_trace_statistic

            eigen_not_coint = (eigen_stats.loc['eigen_value'] < eigen_stats.loc[sig_level]).all()
            trace_not_coint = (trace_stats.loc['trace_statistic'] < trace_stats.loc[sig_level]).all()

            if eigen_not_coint or trace_not_coint:
                warnings.warn("The asset pair is not cointegrated at "
                              "{} level based on eigenvalue or trace statistics.".format(sig_level))

            # Retrieve beta
            coint_vec = jo_portfolio.cointegration_vectors.loc[0]

            # Normalize based on the first asset
            coint_vec = coint_vec / coint_vec[0]
            beta = coint_vec[1]

        else:
            # Use Engle-Granger test to find the hedge ratio
            eg_portfolio = EngleGrangerPortfolio()
            eg_portfolio.fit(self._train_df, add_constant=True)

            # Check ADF statistics to see if the pairs are cointegrated at 90% level.
            adf_stats = eg_portfolio.adf_statistics

            # ADF stats are negative. The largest one is the least significant.
            if (adf_stats.loc['statistic_value'] > adf_stats.loc[sig_level]).all():
                warnings.warn("The asset pair is not cointegrated at {} level "
                              "based on ADF statistics.".format(sig_level))

            # Retrieve beta
            coint_vec = eg_portfolio.cointegration_vectors
            beta = coint_vec.iloc[:, 1].values[0]

        # Calculate the cointegration error, epsilon_t
        epsilon_t = self._train_df.iloc[:, 0] + beta * self._train_df.iloc[:, 1]

        # Fit an AR(1) model to find the AR(1) coefficient
        with warnings.catch_warnings():
            # Silencing specific statsmodels ValueWarning
            warnings.filterwarnings('ignore', r'A date index has been provided,')
            ar_fit = sm.tsa.ARMA(epsilon_t, (1, 0)).fit(trend='c', disp=0)
            _, ar_coeff = ar_fit.params

        return beta, epsilon_t, ar_coeff, ar_fit.resid

    @staticmethod
    def _gaussian_kernel(ar_coeff: float, integrate_grid: np.array, ar_resid: np.array) -> np.array:
        """
        Calculate the Gaussian kernel :math:`K(u_i, u_j)` matrix for mean passage time calculation.

        :param ar_coeff: (float) The fitted AR(1) coefficient.
        :param integrate_grid: (np.array) The integration grid with equal separation.
        :param ar_resid: (np.array) The residual obtained from AR(1) fit on cointegration error.
        :return: (np.array) The Gaussian kernel (K(u_i, u_j)) matrix.
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

        # Vectorize the term (u_j - phi * u_i) in the exponential
        exp_term1 = np.tile(integrate_grid, (len_grid, 1))
        exponent = exp_term1 - ar_coeff * integrate_grid.reshape(-1, 1)

        # Calculate the kernel
        kernel = grid_h / (2. * np.sqrt(2 * np.pi) * sigma_ksi) * np.exp(-0.5 / (sigma_ksi ** 2) * np.square(exponent))

        # Multiply the weights
        kernel = np.multiply(kernel, weights.reshape(1, -1))

        return kernel

    def _mean_passage_time(self, lower: int, upper: int, ar_coeff: float, ar_resid: np.array,
                           granularity: float) -> pd.Series:
        """
        Compute E(\\Tau_{a, b}(y0)), where lower = a, upper = b.

        :param lower: (int) Interval lower bound.
        :param upper: (int) Interval upper bound.
        :param ar_coeff: (float) AR(1) coefficient.
        :param ar_resid: (np.array) The residual obtained from AR(1) fit on cointegration error.
        :param granularity: (float) Summation interval for integration.
        :return: (pd.Series) Mean first-passage time over interval [a,b] of an AR(1) process,
            starting at y0.
        """
        # Build the grid for summation
        grid = granularity * np.arange(lower, upper)

        # Calculate the gaussian kernel
        gaussian = self._gaussian_kernel(ar_coeff, grid, ar_resid)

        # Calculate the mean passage time at each grid point
        k_dim = gaussian.shape[0]
        passage_time = np.linalg.solve(np.eye(k_dim) - gaussian, np.ones(k_dim))

        # Return a pandas.Series indexed by grid points for easy retrieval
        passage_time_df = pd.Series(passage_time, index=grid)
        return passage_time_df

    @staticmethod
    def _print_progress(iteration, max_iterations, prefix='', suffix='', decimals=1, bar_length=50):
        # pylint: disable=expression-not-assigned
        """
        Calls in a loop to create a terminal progress bar.
        https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
        :param iteration: (int) Current iteration.
        :param max_iterations: (int) Maximum number of iterations.
        :param prefix: (str) Prefix string.
        :param suffix: (str) Suffix string.
        :param decimals: (int) Positive number of decimals in percent completed.
        :param bar_length: (int) Character length of the bar.
        """
        str_format = "{0:." + str(decimals) + "f}"
        # Calculate the percent completed.
        percents = str_format.format(100 * (iteration / float(max_iterations)))
        # Calculate the length of bar.
        filled_length = int(round(bar_length * iteration / float(max_iterations)))
        # Fill the bar.
        block = '█' * filled_length + '-' * (bar_length - filled_length)
        # Print new line.
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, block, percents, '%', suffix)),

        if iteration == max_iterations:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def optimize(self, ar_coeff: float, epsilon_t: pd.Series, ar_resid: np.array,
                 horizon: int, granularity: float = 0.01) -> Tuple[float, ...]:
        """
        Optimize the upper bound following the optimization procedure in paper.

        .. note::

            The Nyström method used to estimate the inter-trade interval is computationally intensive and could take
            considerable amount of time to yield the final result, especially when cointegration error has a large
            standard deviation.

        :param ar_coeff: (float) AR(1) coefficient of the cointegrated spread.
        :param epsilon_t: (pd.Series) Cointegration error.
        :param ar_resid: (np.array) AR(1) fit residual on cointegration error.
        :param horizon: (int) Test trading period.
        :param granularity: (float) Integration discretization interval, default to 0.01.
        :return: (float, float, float, float, float) Optimal upper bound; optimal trade duration;
            optimal inter-trades interval; optimal minimum trade profit; optimal number of trades.
        """
        minimum_trade_profit = []

        # Use 5 times the standard deviation of cointegration error as an approximation of infinity
        infinity = np.floor(epsilon_t.std() * 5 / granularity + 1)

        # Generate a sequence of pre-set upper-bounds
        upper_bounds = granularity * np.arange(0, infinity)

        # For trade duration calculation, the integration is on fixed interval [0, inf].
        # Only calculate once to be efficient.
        trade_durations = self._mean_passage_time(0, infinity, ar_coeff, ar_resid, granularity)

        # For each upper bound, calculate minimum total profit
        for ub in tqdm(upper_bounds):
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

        # According to construction, the mtp is the variable we want to maximize (3rd column)
        max_idx = minimum_trade_profit[:, 2].argmax()

        # Retrieve optimal parameter set
        return (upper_bounds[max_idx], *minimum_trade_profit[max_idx, :])

    def trade_signal(self, upper_bound: float, minimum_profit: float,
                     beta: float, epsilon_t: np.array) -> Tuple[pd.DataFrame, np.array, np.array]:
        """
        Generate the trade signal and calculate the number of shares to trade.

        :param upper_bound: (float) Optimized upper bound based on mean passage time optimization.
        :param minimum_profit: (float) Optimized minimum profit based on mean passage time
            optimization.
        :param beta: (float) Fitted cointegration coefficient, beta.
        :param epsilon_t: (np.array) Cointegration error obtained from training set.
        :return: (pd.DataFrame, np.array, np.array) Dataframe with trading conditions;
            number of shares to trade for each leg in the cointegration pair;
            exact values of cointegration error for initiating and closing trades.
        """
        # According to the paper, trading one unit of the cointegrated pair yields a minimum profit of upper_bound
        # Therefore, minimum_profit cannot be less than upper_bound
        if minimum_profit < upper_bound:
            raise Exception("The minimum profit should be greater than the upper bound!")

        # Closing condition, which is the mean of the epsilon_t
        closing_cond = epsilon_t.mean()

        # Overbought level to fade the spread, corresponds to U-trades
        overbought = closing_cond + upper_bound

        # Oversold level to fade the spread, corresponds to L-trades
        oversold = closing_cond - upper_bound

        # Step 2, choose integer n > K * abs(beta) / (a-b), which is the number of share S2 to trade
        # Calculate the number of share S1 to trade as well
        share_s2_count = np.ceil(minimum_profit * np.abs(beta) / upper_bound)
        share_s1_count = np.ceil(share_s2_count / abs(beta))

        # Now calculate the cointegration error for the trade_df
        trade_epsilon_t = self._trade_df.iloc[:, 0] + beta * self._trade_df.iloc[:, 1]
        trade_df_with_cond = self._trade_df.assign(coint_error=trade_epsilon_t)

        # U-trade triggers
        trade_df_with_cond = trade_df_with_cond.assign(otc_U=trade_df_with_cond['coint_error'] >= overbought)
        trade_df_with_cond = trade_df_with_cond.assign(ctc_U=trade_df_with_cond['coint_error'] <= closing_cond)

        # L-trade triggers
        trade_df_with_cond = trade_df_with_cond.assign(otc_L=trade_df_with_cond['coint_error'] <= oversold)
        trade_df_with_cond = trade_df_with_cond.assign(ctc_L=trade_df_with_cond['coint_error'] >= closing_cond)

        # Record number of shares to trade
        shares = np.array([share_s1_count, share_s2_count])

        # Record the exact values of initiating/closing trades for plotting purposes
        cond_lines = np.array([oversold, closing_cond, overbought])

        return trade_df_with_cond, shares, cond_lines