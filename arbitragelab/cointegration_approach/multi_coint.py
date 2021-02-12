# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module generates trading signals of three or more cointegrated assets.
"""

import warnings
from copy import deepcopy
from typing import Tuple, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arbitragelab.cointegration_approach.johansen import JohansenPortfolio


class MultivariateCointegration:
    """
    This class generates the trading signals for a daily rebalancing strategy, calculate the returns of the strategy,
    and plot the equity curves and cointegration vector time evolution.
    """

    def __init__(self, asset_df: pd.DataFrame, trade_df: pd.DataFrame):
        """
        Constructor of the multivariate cointegration trading signal class.

        The log price dataframe and the cointegration vectors are stored for repeating use.

        :param asset_df: (pd.DataFrame) Raw in-sample price dataframe of the assets.
        :param trade_df: (pd.DataFrame) Raw out-of-sample price dataframe of the assets. Use None if only in-sample
            properties are desired.
        """

        self.__asset_df = asset_df
        self.__trade_df = trade_df
        self.__coint_vec = None

    @staticmethod
    def _missing_impute(price_df: pd.DataFrame, nan_method: str = 'ffill', order: int = 3) -> pd.DataFrame:
        """
        Fill the missing values of the asset prices with two options: front-fill or cubic spline.

        :param price_df: (pd.DataFrame) Dataframe that contains the raw asset price.
        :param nan_method: (str) Missing value imputation method. If "ffill" then use front-fill;
            if "spline" then use cubic spline.
        :param order: (int) Polynomial order for spline function.
        :return: (pd.DataFrame) Imputed dataframe.
        """

        if nan_method == "ffill":
            # Use front-fill to impute
            price_df = price_df.fillna(method='ffill')

        elif nan_method == "spline":
            # Use cubic spline to impute
            price_df = price_df.interpolate(method='spline', order=order)

        else:
            raise ValueError("The value of argument nan_method must be 'ffill' or 'spline'.")

        return price_df

    def fillna_inplace(self, nan_method: str = 'ffill', order: int = 3):
        """
        Replace the class attribute dataframes with imputed training dataframe and trading dataframe.

        :param nan_method: (str) Missing value imputation method. If "ffill" then use front-fill;
            if "spline" then use cubic spline.
        :param order: (int) Polynomial order for spline function.
        """

        total_df = self._missing_impute(pd.concat([self.__asset_df, self.__trade_df]),
                                        nan_method=nan_method, order=order)
        self.__asset_df = total_df.loc[self.__asset_df.index]
        self.__trade_df = total_df.loc[self.__trade_df.index]

    @staticmethod
    def calc_log_price(price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the log price of each asset for cointegration coefficient calculation.

        :param price_df: (pd.DataFrame) Dataframe that contains the raw asset price.
        :return: (pd.DataFrame) Log prices of the assets.
        """

        # Return log price
        return price_df.apply(np.log)

    @staticmethod
    def calc_price_diff(price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the price difference of day t and day t-1 of each asset for P&L calculation.

        :param price_df: (pd.DataFrame) Dataframe that contains the raw asset price.
        :return: (pd.DataFrame) Log prices of the assets.
        """

        # Drop first row of NA and return the price difference
        return price_df.diff().dropna()

    @property
    def asset_df(self) -> pd.DataFrame:
        """
        Property that gives read-only access to the in-sample asset price dataframe.

        :return: (pd.DataFrame) Dataframe of asset prices.
        """

        return self.__asset_df

    @property
    def trade_df(self) -> pd.DataFrame:
        """
        Property that gives read-only access to the out-of-sample asset price dataframe.

        :return: (pd.DataFrame) Dataframe of log asset prices.
        """

        return self.__trade_df

    def fit(self, log_price: pd.DataFrame, sig_level: str = "95%", rolling_window_size: Optional[int] = 1500,
            suppress_warnings: bool = False) -> np.array:
        """
        Use Johansen test to retrieve the cointegration vector.

        :param log_price: (pd.DataFrame) Log price dataframe used to derive cointegration vector.
        :param sig_level: (str) Cointegration test significance level. Possible options are "90%", "95%", and "99%".
        :param rolling_window_size: (int) Number of data points used for training with rolling window. If None,
            then use cumulative window, i.e. the entire dataset.
        :param suppress_warnings: (bool) Boolean flag to suppress the cointegration warning message.
        :return: (np.array) The cointegration vector, b.
        """

        # Checking the significance of a test
        if sig_level not in ['90%', '95%', '99%']:
            raise ValueError("Significance level can only be the following:\n "
                             "90%, 95%, or 99%.\n Please check the input.")

        # Calculate the cointegration vector with Johansen test
        jo_portfolio = JohansenPortfolio()

        # Select if applying rolling window
        if rolling_window_size is None:
            jo_portfolio.fit(log_price, det_order=0)
        else:
            jo_portfolio.fit(log_price.iloc[-rolling_window_size:], det_order=0)

        # Check statistics to see if the pairs are cointegrated at the specified significance level
        eigen_stats = jo_portfolio.johansen_eigen_statistic
        trace_stats = jo_portfolio.johansen_trace_statistic
        eigen_not_coint = (eigen_stats.loc['eigen_value'] < eigen_stats.loc[sig_level]).all()
        trace_not_coint = (trace_stats.loc['trace_statistic'] < trace_stats.loc[sig_level]).all()

        # If not cointegrated then warn the users that the performance might be affected
        if not suppress_warnings and (eigen_not_coint or trace_not_coint):
            warnings.warn("The asset pair is not cointegrated at "
                          "{} level based on eigenvalue or trace statistics.".format(sig_level))

        # Retrieve the cointegration vector and store it
        coint_vec = jo_portfolio.cointegration_vectors.loc[0]
        self.__coint_vec = coint_vec

        return coint_vec

    def num_of_shares(self, log_price: pd.DataFrame, last_price: pd.Series, nlags: int = 30,
                      dollar_invest: float = 1.e7) -> Tuple[np.array, ...]:
        """
        Calculate the number of shares that needs to be traded given a notional value.

        :param log_price: (pd.DataFrame) Dataframe of log prices of training data.
        :param last_price: (pd.Series) Last price for trading signal generation.
        :param nlags: (int) Amount of lags for cointegrated returns sum, corresponding to the parameter P in the paper.
        :param dollar_invest: (float) The value of long-short positions, corresponding to the parameter C in the paper.
        :return: (np.array, np.array, np.array, np.array) The number of shares to trade;
            the notional values of positions.
        """

        # Calculate the cointegration error Y_t, recover the date index
        coint_error = np.dot(log_price, self.__coint_vec)
        coint_error_df = pd.DataFrame(coint_error)
        coint_error_df.index = log_price.index

        # Calculate the return Z_t by taking the difference. Drop the NaN of the first data point
        realization = coint_error_df.diff().dropna()

        # Calculate the direction of the trade
        sign = np.sign(realization.iloc[-nlags:].sum()).values[0]

        # Classify the assets into positive cointegration coefficient (CC) group and negative CC group
        pos_coef_asset = self.__coint_vec[self.__coint_vec >= 0]
        neg_coef_asset = self.__coint_vec[self.__coint_vec < 0]

        # Calculate notional values
        pos_notional = pos_coef_asset * sign * dollar_invest / pos_coef_asset.sum()
        neg_notional = neg_coef_asset * sign * dollar_invest / neg_coef_asset.sum()

        # Calculate number of shares
        pos_shares = pos_notional / last_price[pos_coef_asset.index]
        neg_shares = neg_notional / last_price[neg_coef_asset.index]

        # Calculate actual notional values due to rounding
        pos_notional = np.floor(pos_shares) * last_price[pos_coef_asset.index]
        neg_notional = np.floor(neg_shares) * last_price[neg_coef_asset.index]

        # Assign the correct sign to the number of shares according to the sign of CC
        return -1. * np.floor(pos_shares), np.floor(neg_shares), -1. * pos_notional, neg_notional

    @staticmethod
    def _rebal_pnl(signal: pd.Series, price_diff: pd.Series) -> Tuple[float, float]:
        """
        Calculate the P&L of one day's trade.

        Suppose the current time is T. The signals generated are based on price history up to time T-1. We open the
        positions at T and exit the positions at T+1, and calculate the P&L for long positions and short positions,
        respectively.

        By construction, both the variables `signal` and `price_diff` have the asset names

        :param signal: (pd.Series) Trade signal of the most recent day.
        :param price_diff: (pd.Series) The price difference of the assets between T and T+1.
        :return: (float, float) Long P&L; Short P&L.
        """

        # Join the trading signal and price difference by asset names
        day_pnl_df = pd.concat([signal, price_diff], axis=1)

        # Rename the columns
        day_pnl_df.columns = ["Shares", "Price Diff"]

        # Retrieve the long positions and short positions
        long_df = day_pnl_df[day_pnl_df['Shares'] >= 0]
        short_df = day_pnl_df[day_pnl_df['Shares'] < 0]

        # Calculate the long PnL and short PnL
        long_pnl = (long_df['Shares'] * long_df['Price Diff']).sum()
        short_pnl = (short_df['Shares'] * short_df['Price Diff']).sum()

        return long_pnl, short_pnl

    # pylint: disable=invalid-name, too-many-locals
    def trading_signal(self, nlags: int, dollar_invest: float = 1.e7, rolling_window_size: Optional[int] = None,
                       update_freq: int = 22, insample: bool = False) -> Tuple[pd.DataFrame, ...]:
        """
        Generate trading signal, i.e. the number of shares to trade each day.

        :param nlags: (int) Amount of lags for cointegrated returns sum, corresponding to the parameter P in the paper.
        :param dollar_invest: (float) The value of long/short positions, corresponding to the parameter C in the paper.
        :param rolling_window_size: (int) Number of data points used for training with rolling window. If None,
            then use cumulative window, i.e. the entire dataset.
        :param update_freq: (int) Frequency to update the cointegration vector for out-of-sample test. Default is
            monthly (22 trading days).
        :param insample: (bool) If True, run an in-sample test where the cointegration vector will be estimated using
            all available data and will not be recalculated monthly.
        :return: (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame) Trading signal dataframe;
            trading signal notional dataframe;  cointegration vector time evolution dataframe; daily returns dataframe.
        """

        # Signal DF, cointegration vector evolution DF, and portfolio value DF for PnL and returns calculation
        signals, signals_notional, coint_vec_evo, returns_df = [], [], [], []

        # Create a copy of the original training DF so we can preserve the data because we will update the original
        # training data with incoming trading data
        train_df = deepcopy(self.__asset_df)

        # Get trading period and daily price difference
        price_diff = self.calc_price_diff(self.__trade_df)
        trading_period = price_diff.shape[0]

        # If in sample, calculate the cointegration vector first as it will not change anymore
        if insample:
            all_data = self.calc_log_price(pd.concat([self.__asset_df, self.__trade_df]))
            self.fit(all_data, rolling_window_size=rolling_window_size, suppress_warnings=True)

        # Generate the trading signal for each day using a loop
        for t in range(trading_period):
            # Calculate log prices
            log_train_df = self.calc_log_price(train_df)

            # Update the cointegration vector if out-of-sample
            if not insample and t % update_freq == 0:
                self.fit(log_train_df, rolling_window_size=rolling_window_size, suppress_warnings=True)

            # Calculate number of shares to trade
            pos_shares, neg_shares, pos_ntn, neg_ntn = self.num_of_shares(log_train_df, self.__trade_df.iloc[t],
                                                                          nlags=nlags, dollar_invest=dollar_invest)

            # Calculate the PnL for one day's trade and daily percentage returns
            long_pnl, short_pnl = self._rebal_pnl(pd.concat([pos_shares, neg_shares]), price_diff.iloc[t])
            returns = (long_pnl + short_pnl) / (2 * dollar_invest)

            # Bookkeeping: Record the signals and the cointegration vector time evolution
            signals.append(pd.concat([pos_shares, neg_shares]))
            signals_notional.append(pd.concat([pos_ntn, neg_ntn]))
            coint_vec_evo.append(self.__coint_vec)
            returns_df.append(returns)

            # Update the training dataframe
            train_df = train_df.append(self.__trade_df.iloc[t])

        # Concatenate the signals and convert the signal into a dataframe
        signals_df = pd.concat(signals, axis=1).T
        signals_ntn_df = pd.concat(signals_notional, axis=1).T
        signals_df.index = price_diff.index
        signals_ntn_df.index = price_diff.index

        # Concatenate the time evolution of cointegration vectors and convert it into a dataframe
        coint_vec_evo_df = pd.concat(coint_vec_evo, axis=1).T
        coint_vec_evo_df.index = self.__trade_df.index[:-1]

        # Generate the returns DF
        returns_df = pd.DataFrame(returns_df)
        returns_df.index = signals_df.index

        return signals_df, signals_ntn_df, coint_vec_evo_df, returns_df

    @staticmethod
    def summary(returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Statistics of the trading strategy returns.

        The statistics include: mean, standard deviation, skewness, kurtosis, Sharpe ratio, Sortino ratio,
        final cumulative returns, percentage of up days and down days, max returns, and min returns.

        :param returns_df: (pd.DataFrame) Daily percentage returns dataframe.
        :return: (pd.DataFrame) Trading strategy returns statistics dataframe.
        """

        # Get the mean, standard deviation, and total trading days
        basic_stats = returns_df.describe()
        rt_mean = basic_stats.loc['mean'].squeeze()
        rt_std = basic_stats.loc['std'].squeeze()
        total_days = basic_stats.loc['count'].squeeze()

        # Get the max returns and min returns
        rt_max = basic_stats.loc['max'].squeeze()
        rt_min = basic_stats.loc['min'].squeeze()

        # Get skewness and kurtosis
        rt_skew = returns_df.skew().squeeze()
        rt_kurt = returns_df.kurt().squeeze()

        # Calculate the Sharpe ratio
        rt_sharpe = rt_mean / rt_std * np.sqrt(252)

        # Calculate the percentage of up days and down days
        down_returns = returns_df[returns_df[0] < 0]
        up_returns = returns_df[returns_df[0] >= 0]
        rt_down_pct = len(down_returns) / total_days
        rt_up_pct = len(up_returns) / total_days

        # Calculate Sortino ratio
        rt_sortino = rt_mean / down_returns.std().squeeze() * np.sqrt(252)

        # Calculate cumulative returns
        cumul_rt_df = (1. + returns_df).cumprod().squeeze() - 1.
        rt_cumul_return = cumul_rt_df.iloc[-1].squeeze()

        # Summarize everything into one single dataframe.
        summary_dict = {
            "Cumulative Return": rt_cumul_return,
            "Returns Mean": rt_mean,
            "Returns Standard Deviation": rt_std,
            "Returns Skewness": rt_skew,
            "Returns Kurtosis": rt_kurt,
            "Max Return": rt_max,
            "Min Return": rt_min,
            "Sharpe ratio": rt_sharpe,
            "Sortino ratio": rt_sortino,
            "Percentage of Up Days": rt_up_pct,
            "Percentage of Down Days": rt_down_pct
        }

        return pd.Series(summary_dict).to_frame()

    @staticmethod
    def plot_all(signals: pd.DataFrame, signals_ntn: pd.DataFrame, coint_vec_evo: pd.DataFrame, returns: pd.DataFrame,
                 figw: float = 15., figh: float = 15., title: str = "", use_weights: bool = False,
                 start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None) -> plt.Figure:
        """
        Plot the equity curve, the cointegration vector time evolution, and the trading signals.

        :param signals: (pd.DataFrame) Trading signal dataframe.
        :param signals_ntn: (pd.DataFrame) Trading signal dataframe in notional values.
        :param coint_vec_evo: (pd.DataFrame) Cointegration vector time evolution dataframe.
        :param returns: (pd.DataFrame) Daily returns dataframe.
        :param figw: (float) Figure width.
        :param figh: (float) Figure height.
        :param title: (str) Figure title details.
        :param use_weights: (bool) If True, use percentage of notional value of each position to represent signals;
            otherwise, use number of shares to represent signals.
        :param start_date: (pd.Timestamp) Start point of the plot.
        :param end_date: (pd.Timestamp) End point of the plot.
        :return: (plt.Figure) The full plot including 3 subplots: equity curve, cointegration vector, and trading
            signals.
        """

        # Define the ticks on the x-axis
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        # Set up the grid
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(figw, figh), gridspec_kw={'height_ratios': [2.5, 1.5, 1]})

        # Equity curve
        cumul_rt = (1. + returns).cumprod().squeeze() - 1.
        axes[0].plot(cumul_rt, label='Cumulative Returns')
        axes[0].legend(loc='upper left', fontsize=12)
        axes[0].tick_params(axis='y', labelsize=14)
        axes[0].set_ylabel("Cumul. Returns", fontsize=14)

        # Cointegration vector time evolution
        lines = axes[1].plot(coint_vec_evo)
        axes[1].legend(iter(lines), list(coint_vec_evo.columns), loc='lower center', fontsize=12,
                       ncol=4, bbox_to_anchor=(0.5, -0.2))
        axes[1].tick_params(axis='y', labelsize=14)
        axes[1].set_ylabel("Coint. Vector", fontsize=14)

        # Signal lines
        if use_weights:
            weights = signals_ntn.apply(lambda x: x / x.abs().sum(), axis=1)
            axes[2].plot(weights)
            axes[2].set_ylabel("Portfolio Weights", fontsize=14)
        else:
            axes[2].plot(signals)
            axes[2].set_ylabel("Num. of Shares", fontsize=14)
        axes[2].tick_params(axis='y', labelsize=14)

        # Formatting the tick labels
        axes[0].xaxis.set_major_locator(years)
        axes[0].xaxis.set_major_formatter(years_fmt)
        axes[0].xaxis.set_minor_locator(months)
        axes[0].tick_params(axis='x', labelsize=14)
        axes[0].tick_params(axis='y', labelsize=14)

        # Define the date range of the plot
        if start_date is not None and end_date is not None:
            axes[2].set_xlim((start_date, end_date))

        # Set up a title for the entire plot
        fig.suptitle("Returns, Cointegration Vector Time-Evolution, and Trading Signals\n"
                     "{}".format(title), fontsize=20)
        fig.subplots_adjust(top=0.92)
        return fig

    @staticmethod
    def plot_returns(returns: pd.DataFrame, figw: float = 15., figh: float = 15., title: str = "Returns",
                     start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None) -> plt.Figure:
        """
        Plot the equity curve only.

        :param returns: (pd.DataFrame) Daily returns dataframe.
        :param figw: (float) Figure width.
        :param figh: (float) Figure height.
        :param title: (str) Figure title.
        :param start_date: (pd.Timestamp) Start point of the plot.
        :param end_date: (pd.Timestamp) End point of the plot.
        :return: (plt.Figure) A single equity curve plot.
        """

        # Define the ticks on the x-axis
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        # Setup the grid
        fig, ax = plt.subplots(1, 1, figsize=(figw, figh))

        # Plot the equity curve
        cumul_rt = (1. + returns).cumprod().squeeze() - 1.
        ax.plot(cumul_rt, label='Cumulative Returns')
        ax.legend(loc='upper left', fontsize=12)
        ax.tick_params(axis='y', labelsize=14)

        # Formatting the tick labels
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylabel("Cumul. Returns", fontsize=14)

        # Define the date range of the plot
        if start_date is not None and end_date is not None:
            ax.set_xlim((start_date, end_date))

        # Set up a title for the entire plot
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=0.95)
        return fig
