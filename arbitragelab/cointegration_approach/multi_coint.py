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

from arbitragelab.cointegration_approach import JohansenPortfolio


class MultivariateCointegration:
    """
    Calculate trading signal.
    """

    def __init__(self, asset_df: pd.DataFrame):
        """
        Constructor of the multivariate cointegration trading signal class.

        The log price dataframe and the cointegration vectors are stored for repeating use.

        :param asset_df: (pd.Dataframe) Raw price dataframe of the assets.
        """

        self.__asset_df = asset_df
        self.__log_df = None
        self.__coint_vec = None

    def calc_log_price(self, nan_method: str = 'spline', order: int = 3) -> pd.DataFrame:
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
            self.__asset_df.fillna(method='ffill', inplace=True)

        elif nan_method == "spline":
            # Use cubic spline to impute.
            self.__asset_df.interpolate(method='spline', order=order, inplace=True)

        else:
            raise ValueError("The value of argument nan_method must be 'ffill' or 'spline'")

        self.__log_df = self.__asset_df.apply(np.log)

        return self.__log_df

    @property
    def asset_df(self) -> pd.DataFrame:
        """
        Property that gives read-only access to the asset price dataframe.

        :return: (pd.DataFrame) Dataframe of asset prices.
        """

        return self.__asset_df

    @property
    def log_df(self) -> pd.DataFrame:
        """
        Property that gives read-only access to the log asset price dataframe.

        :return: (pd.DataFrame) Dataframe of log asset prices.
        """

        return self.__log_df

    def fit(self, nan_method: str = 'spline', order: int = 3, sig_level: str = "95%",
            rolling_window_size: Optional[int] = 1500) -> np.array:
        """
        Use the Johansen test to retrieve the cointegration vector.

        :param nan_method: (str) Missing value imputation method. If "ffill" then use front-fill;
            if "spline" then use cubic spline.
        :param order: (int) Polynomial order for spline function.
        :param sig_level: (str) Cointegration test significance level. Possible options are "90%", "95%", and "99%".
        :param rolling_window_size: (int) Number of data points used for training with rolling window. If None,
            then use cumulative window, i.e. the entire dataset
        :return: (np.array) The cointegration vector, b.
        """

        # Checking the significance of a test.
        if sig_level not in ['90%', '95%', '99%']:
            raise ValueError("Significance level can only be the following:\n "
                             "90%, 95%, or 99%.\n Please check the input.")

        # Calculate the cointegration vector.
        jo_portfolio = JohansenPortfolio()
        log_price_df = self.calc_log_price(nan_method=nan_method, order=order)

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
        if eigen_not_coint or trace_not_coint:
            warnings.warn("The asset pair is not cointegrated at "
                          "{} level based on eigenvalue or trace statistics.".format(sig_level))

        # Retrieve beta and store it.
        coint_vec = jo_portfolio.cointegration_vectors.loc[0]
        self.__coint_vec = coint_vec

        return coint_vec

    def trading_signal(self, nlags: int = 30, dollar_invest: float = 1.e7) -> Tuple[np.array, ...]:
        """
        Calculate the daily trading signal as the number of shares that need to be traded.

        :param nlags: Number of lag for cointegrated returns sum, corresponding to the parameter P in the paper.
        :param dollar_invest: The value of long-short positions, corresponding to the parameter C in the paper.
        :return (np.array, np.array): Arrays with numbers of shares to sell and buy.
        """

        # Calculate the cointegration error Y_t, recover the date index.
        coint_error = np.dot(self.__log_df, self.__coint_vec)
        coint_error_df = pd.DataFrame(coint_error)
        coint_error_df.index = self.__log_df.index

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

        # Round down. Sell the rip for positive shares, buy the dip for negative shares.
        return -1. * np.floor(pos_shares), np.floor(neg_shares)
