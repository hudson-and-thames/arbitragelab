# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module generates trading signals of three or more cointegrated assets.
"""

import warnings
from copy import deepcopy
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from arbitragelab.cointegration_approach import JohansenPortfolio


class MultivariateCointegration:
    """
    This class generates the trading signals for a daily rebalancing strategy, calculate the returns of the strategy,
    and plot the equity curves and cointegration vector time evolution.
    """

    def __init__(self, asset_df: pd.DataFrame, trade_df: Optional[pd.DataFrame]):
        """
        Constructor of the multivariate cointegration trading signal class.

        The log price dataframe and the cointegration vectors are stored for repeating use.

        :param asset_df: (pd.Dataframe) Raw in-sample price dataframe of the assets.
        :param trade_df: (pd.Dataframe) Raw out-of-sample price dataframe of the assets. Use None if only in-sample
            properties are desired.
        """
        self.__asset_df = asset_df
        self.__trade_df = trade_df
        self.__coint_vec = None


    @staticmethod
    def missing_impute(price_df: pd.DataFrame, nan_method: str = 'ffill', order: int = 3):
        """
        Fill the missing values of the asset prices with two options: front-fill or cubic spline.

        :param price_df: (pd.DataFrame) Dataframe that contains the raw asset price.
        :param nan_method: (str) Missing value imputation method. If "ffill" then use front-fill;
            if "spline" then use cubic spline.
        :param order: (int) Polynomial order for spline function.
        :return: (pd.DataFrame) Imputed dataframe.
        """
        if nan_method == "ffill":
            # Use front-fill to impute.
            price_df = price_df.fillna(method='ffill')

        elif nan_method == "spline":
            # Use cubic spline to impute.
            price_df = price_df.interpolate(method='spline', order=order)

        else:
            raise ValueError("The value of argument nan_method must be 'ffill' or 'spline'.")

        return price_df

    def calc_log_price(self, price_df: pd.DataFrame, nan_method: str = 'ffill', order: int = 3):
        """
        Calculate the log price of each asset for cointegration coefficient calculation.

        :param price_df: (pd.DataFrame) Dataframe that contains the raw asset price.
        :param nan_method: (str) Missing value imputation method. If "ffill" then use front-fill;
            if "spline" then use cubic spline.
        :param order: (int) Polynomial order for spline function.
        :return: (pd.DataFrame) Log prices of the assets.
        """

        # Impute missing data
        price_df = self.missing_impute(price_df, nan_method=nan_method, order=order)

        # Return log price.
        return price_df.apply(np.log)

    def calc_price_diff(self, price_df: pd.DataFrame, nan_method: str = 'ffill', order: int = 3):
        """
        Calculate the price difference of day t and day t-1 of each asset for P&L calculation.

        :param price_df: (pd.DataFrame) Dataframe that contains the raw asset price.
        :param nan_method: (str) Missing value imputation method. If "ffill" then use front-fill;
            if "spline" then use cubic spline.
        :param order: (int) Polynomial order for spline function.
        :return: (pd.DataFrame) Log prices of the assets.
        """

        # Impute missing data
        price_df = self.missing_impute(price_df, nan_method=nan_method, order=order)

        # Drop first row of NA and return the price difference.
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

    def fit(self, log_price: pd.DataFrame, nan_method: str = "ffill", sig_level: str = "95%",
            rolling_window_size: Optional[int] = 1500, suppress_warnings: bool = False,
            spline_order: int = 3) -> np.array:
        """
        Use Johansen test to retrieve the cointegration vector.

        :param log_price: (pd.DataFrame) Log price dataframe used to derive cointegration vector.
        :param nan_method: (str) Missing value imputation method. If "ffill" then use front-fill;
            if "spline" then use cubic spline.
        :param sig_level: (str) Cointegration test significance level. Possible options are "90%", "95%", and "99%".
        :param rolling_window_size: (int) Number of data points used for training with rolling window. If None,
            then use cumulative window, i.e. the entire dataset.
        :param suppress_warnings: (bool) Boolean flag to suppress the cointegration warning message.
        :param spline_order: (int) Polynomial order for spline function.
        :return: (np.array) The cointegration vector, b.
        """
        # Checking the significance of a test.
        if sig_level not in ['90%', '95%', '99%']:
            raise ValueError("Significance level can only be the following:\n "
                             "90%, 95%, or 99%.\n Please check the input.")

        # Calculate the cointegration vector with Johansen test.
        jo_portfolio = JohansenPortfolio()

        # Select if applying rolling window.
        if rolling_window_size is None:
            jo_portfolio.fit(log_price, det_order=0)
        else:
            jo_portfolio.fit(log_price.iloc[-rolling_window_size:], det_order=0)

        # Check statistics to see if the pairs are cointegrated at the specified significance level.
        eigen_stats = jo_portfolio.johansen_eigen_statistic
        trace_stats = jo_portfolio.johansen_trace_statistic
        eigen_not_coint = (eigen_stats.loc['eigen_value'] < eigen_stats.loc[sig_level]).all()
        trace_not_coint = (trace_stats.loc['trace_statistic'] < trace_stats.loc[sig_level]).all()

        # If not cointegrated then warn the users that the performance might be affected.
        if not suppress_warnings and (eigen_not_coint or trace_not_coint):
            warnings.warn("The asset pair is not cointegrated at "
                          "{} level based on eigenvalue or trace statistics.".format(sig_level))

        # Retrieve the cointegration vector and store it.
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
        :return: (np.array, np.array) The number of shares to trade.
        """

        # Calculate the cointegration error Y_t, recover the date index.
        coint_error = np.dot(log_price, self.__coint_vec)
        coint_error_df = pd.DataFrame(coint_error)
        coint_error_df.index = log_price.index

        # Calculate the return Z_t by taking the difference. Drop the NaN of the first data point.
        realization = coint_error_df.diff().dropna()

        # Calculate the direction of the trade.
        sign = np.sign(realization.iloc[-nlags:].sum()).values[0]

        # Classify the assets into positive cointegration coefficient (CC) group and negative CC group.
        pos_coef_asset = self.__coint_vec[self.__coint_vec >= 0]
        neg_coef_asset = self.__coint_vec[self.__coint_vec < 0]

        # Calculate number of shares.
        pos_shares = pos_coef_asset * sign * dollar_invest / pos_coef_asset.sum() / last_price[pos_coef_asset.index]
        neg_shares = neg_coef_asset * sign * dollar_invest / neg_coef_asset.sum() / last_price[neg_coef_asset.index]

        # Assign the correct sign to the number of shares according to the sign of CC.
        return -1. * np.floor(pos_shares), np.floor(neg_shares)

    @staticmethod
    def rebal_pnl(signal: pd.Series, price_diff: pd.Series) -> Tuple[float, float]:
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

        # Join the trading signal and price difference by asset names.
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

    # pylint: disable=invalid-name
    def trading_signal(self, nlags: int, dollar_invest: float = 1.e7, rolling_window_size: Optional[int] = None,
                       update_freq: int = 22) -> Tuple[pd.DataFrame, ...]:
        """
        Generate trading signal, i.e. the number of shares to trade each day.

        :param nlags: (int) Amount of lags for cointegrated returns sum, corresponding to the parameter P in the paper.
        :param dollar_invest: (float) The value of long/short positions, corresponding to the parameter C in the paper.
        :param rolling_window_size: (int) Number of data points used for training with rolling window. If None,
            then use cumulative window, i.e. the entire dataset.
        :param update_freq: (int) Frequency to update the cointegration vector for out-of-sample test. Default is
            monthly (22 trading days).
        :return: (pd.DataFrame, pd.DataFrame) Trading signal dataframe; cointegration vector time evolution dataframe.
        """

        # Signal DF, cointegration vector evolution DF, and portfolio value DF for PnL and returns calculation.
        signals, coint_vec_evo, returns_df = [], [], []

        # Create a copy of the original training DF so we can preserve the data because we will update the original
        # training data with incoming trading data.
        train_df = deepcopy(self.__asset_df)

        # Get trading period and daily price difference
        price_diff = self.calc_price_diff(self.__trade_df)
        trading_period = price_diff.shape[0]

        # Assign the notional value of the portfolio
        returns_df.append(0.)

        # Generate the trading signal for each day using a loop.
        for t in range(trading_period):
            # Update the cointegration vector according to the update frequency.
            if t % update_freq == 0:
                log_train_df = self.calc_log_price(train_df)
                self.fit(log_train_df, rolling_window_size=rolling_window_size, suppress_warnings=True)

            # Calculate number of shares to trade.
            pos_shares, neg_shares = self.num_of_shares(log_train_df, self.__trade_df.iloc[t], nlags=nlags,
                                                        dollar_invest=dollar_invest)

            # Calculate the PnL for one day's trade
            long_pnl, short_pnl = self.rebal_pnl(pd.concat([pos_shares, neg_shares]), price_diff.iloc[t])

            # Bookkeeping: Record the signals and the cointegration vector time evolution.
            signals.append(pd.concat([pos_shares, neg_shares]))
            coint_vec_evo.append(self.__coint_vec)
            returns = (long_pnl + short_pnl) / (2 * dollar_invest)
            returns_df.append(returns)

            # Update the training dataframe.
            train_df = train_df.append(self.__trade_df.iloc[t])

        # Concatenate the signals and convert the signal into a dataframe.
        signals_df = pd.concat(signals, axis=1).T
        signals_df.index = price_diff.index

        # Concatenate the time evolution of cointegration vectors and convert it into a dataframe.
        coint_vec_evo_df = pd.concat(coint_vec_evo, axis=1).T
        coint_vec_evo_df.index = self.__trade_df.index[:-1]

        # Generate the returns DF
        returns_df = pd.DataFrame(returns_df)

        return signals_df, coint_vec_evo_df, returns_df
