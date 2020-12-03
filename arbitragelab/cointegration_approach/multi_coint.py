# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module generates trading signals of three or more cointegrated assets.
"""

import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from arbitragelab.cointegration_approach import EngleGrangerPortfolio, JohansenPortfolio


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
        self.__log_asset_df = None
        self.__trade_df = trade_df
        self.__coint_vec = None

    def calc_log_price(self, nan_method: str = 'ffill', order: int = 3) -> pd.DataFrame:
        """
        Calculate the log price of each asset for cointegration coefficient calculation.
        Fill missing value with two options: front-fill or cubic spline.

        :param nan_method: (str) Missing value imputation method. If "ffill" then use front-fill;
            if "spline" then use cubic spline.
        :param order: (int) Polynomial order for spline function.
        :return: (pd.DataFrame) Log prices of the assets.
        """
        if nan_method == "ffill":
            # Use front-fill to impute.
            self.__asset_df = self.__asset_df.fillna(method='ffill')

        elif nan_method == "spline":
            # Use cubic spline to impute.
            self.__asset_df = self.__asset_df.interpolate(method='spline', order=order)

        else:
            raise ValueError("The value of argument nan_method must be 'ffill' or 'spline'")

        self.__log_asset_df = self.__asset_df.apply(np.log)
        return self.__log_asset_df

    @property
    def asset_df(self) -> pd.DataFrame:
        """
        Property that gives read-only access to the asset price dataframe.

        :return: (pd.DataFrame) Dataframe of asset prices.
        """
        return self.__asset_df

    @property
    def log_asset_df(self) -> pd.DataFrame:
        """
        Property that gives read-only access to the log asset price dataframe.

        :return: (pd.DataFrame) Dataframe of log asset prices.
        """
        return self.__log_asset_df

    def fit(self, nan_method: str = 'ffill', order: int = 3, sig_level: str = "95%",
            rolling_window_size: Optional[int] = 1500, suppress_warnings=False) -> np.array:
        """
        Use Johansen test to retrieve the cointegration vector.

        :param nan_method: (str) Missing value imputation method. If "ffill" then use front-fill;
            if "spline" then use cubic spline.
        :param order: (int) Polynomial order for spline function.
        :param sig_level: (str) Cointegration test significance level. Possible options are "90%", "95%", and "99%".
        :param rolling_window_size: (int) Number of data points used for training with rolling window. If None,
            then use cumulative window, i.e. the entire dataset.
        :param suppress_warnings: (bool) Boolean flag to suppress the cointegration warning message.
        :return: (np.array) The cointegration vector, b.
        """
        # Checking the significance of a test.
        if sig_level not in ['90%', '95%', '99%']:
            raise ValueError("Significance level can only be the following:\n "
                             "90%, 95%, or 99%.\n Please check the input.")

        # Calculate the log price.
        log_price_df = self.calc_log_price(nan_method=nan_method, order=order)

        # Calculate the cointegration vector with Johansen test.
        jo_portfolio = JohansenPortfolio()

        # Select if applying rolling window.
        if rolling_window_size is None:
            jo_portfolio.fit(log_price_df, det_order=0)
        else:
            jo_portfolio.fit(log_price_df.iloc[-rolling_window_size:], det_order=0)

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

    def num_of_shares(self, nlags: int = 30, dollar_invest: float = 1.e7) -> Tuple[np.array, ...]:
        """
        Calculate the number of shares that needs to be traded.

        :param nlags: (int) Amount of lags for cointegrated returns sum, corresponding to the parameter P in the paper.
        :param dollar_invest: (float) The value of long-short positions, corresponding to the parameter C in the paper.
        :return: (np.array, np.array) The number of shares to trade.
        """
        # Calculate the cointegration error Y_t, recover the date index.
        coint_error = np.dot(self.__log_asset_df, self.__coint_vec)
        coint_error_df = pd.DataFrame(coint_error)
        coint_error_df.index = self.__log_asset_df.index

        # Calculate the return Z_t by taking the difference. Drop the NaN of the first data point.
        realization = coint_error_df.diff().dropna()

        # Calculate the direction of the trade
        sign = np.sign(realization.iloc[-nlags:].sum()).values[0]

        # Classify the assets into positive cointegration coefficient (CC) group and negative CC group.
        pos_coef_asset = self.__coint_vec[self.__coint_vec >= 0]
        neg_coef_asset = self.__coint_vec[self.__coint_vec < 0]

        # Retrieve asset price
        last_price = self.__asset_df.iloc[-1]

        # Calculate number of shares.
        pos_shares = pos_coef_asset * sign * dollar_invest / pos_coef_asset.sum() / last_price[pos_coef_asset.index]
        neg_shares = neg_coef_asset * sign * dollar_invest / neg_coef_asset.sum() / last_price[neg_coef_asset.index]

        # Assign the correct sign to the number of shares according to the sign of CC.
        return -1. * np.floor(pos_shares), np.floor(neg_shares)

    def trading_signal(self, nlags: int, nan_method: str = "ffill", dollar_invest: float = 1.e7,
                       rolling_window_size: Optional[int] = None, update_freq: int = 22,
                       spline_order: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate trading signal, i.e. the number of shares to trade each day.

        :param nlags: (int) Amount of lags for cointegrated returns sum, corresponding to the parameter P in the paper.
        :param nan_method: (str) Missing value imputation method. If "ffill" then use front-fill;
            if "spline" then use cubic spline.
        :param dollar_invest: (float) The value of long-short positions, corresponding to the parameter C in the paper.
        :param rolling_window_size: (int) Number of data points used for training with rolling window. If None,
            then use cumulative window, i.e. the entire dataset.
        :param update_freq: (int) Frequency to update the cointegration vector for out-of-sample test. Default is
            monthly (22 trading days).
        :param spline_order: (int) Polynomial order for spline function.
        :return: (pd.DataFrame, pd.DataFrame) Trading signal dataframe; cointegration vector time evolution dataframe.
        """

        # Signal DF
        signals = []

        # Cointegration vector evolution DF
        coint_vec_evo = []

        # Get the out-of-sample testing period length
        trading_period = self.__trade_df.shape[0]

        # Generate the trading signal for each day using a loop.
        for t in range(trading_period):
            # Update the cointegration vector according to the update frequency
            if t % update_freq == 0:
                self.fit(nan_method=nan_method, order=spline_order, rolling_window_size=rolling_window_size,
                         suppress_warnings=True)

            # Calculate number of shares to trade
            pos_shares, neg_shares = self.num_of_shares(nlags=nlags, dollar_invest=dollar_invest)
            signals.append(pd.concat([pos_shares, neg_shares]))

            # Update the new datapoint to the training dataframe
            self.__asset_df = self.__asset_df.append(self.__trade_df.iloc[t])

            # Record the cointegration vector evolution
            coint_vec_evo.append(self.__coint_vec)

        signals_df = pd.concat(signals, axis=1).T
        signals_df.index = self.__trade_df.index

        coint_vec_evo_df = pd.concat(coint_vec_evo, axis=1).T
        coint_vec_evo_df.index = self.__trade_df.index

        return signals_df, coint_vec_evo_df
