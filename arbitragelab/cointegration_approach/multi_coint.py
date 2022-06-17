# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module generates a cointegration vector for mean-reversion trading of three or more cointegrated assets.
"""

import warnings
from typing import Tuple, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arbitragelab.cointegration_approach.johansen import JohansenPortfolio
from arbitragelab.util import segment


class MultivariateCointegration:
    """
    This class optimizes bounds for mean-reversion trading of a spread consisting of three and more assets.

    The implementation is based on the method described by Galenko, A., Popova, E. and Popova, I. in
    `"Trading in the presence of cointegration" <http://www.ntuzov.com/Nik_Site/Niks_files/Research/papers/stat_arb/Galenko_2007.pdf>`_
    """

    def __init__(self):
        """
        Constructor of the multivariate cointegration trading signal class.

        The log price dataframe and the cointegration vectors are stored for repeating use.
        """

        self.__asset_df = None
        self.__coint_vec = None

        segment.track('MultivariateCointegration')

    def set_train_dataset(self, price_df: pd.DataFrame):
        """
        Provide price series for model to calculate the cointegration coefficient and beta.

        :param price_df: (pd.DataFrame) Price series dataframe which contains both series.
        """

        self.__asset_df = price_df

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
        Replace the class attribute dataframes with imputed training dataframe.

        :param nan_method: (str) Missing value imputation method. If "ffill" then use front-fill;
            if "spline" then use cubic spline.
        :param order: (int) Polynomial order for spline function.
        """

        total_df = self._missing_impute(self.__asset_df, nan_method=nan_method, order=order)
        self.__asset_df = total_df

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
        Calculate the price difference of day t and day t-1 of each asset.

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

    def fit(self, log_price: pd.DataFrame, sig_level: str = "95%", suppress_warnings: bool = False) -> np.array:
        """
        Use Johansen test to retrieve the cointegration vector.

        :param log_price: (pd.DataFrame) Log price dataframe used to derive cointegration vector.
        :param sig_level: (str) Cointegration test significance level. Possible options are "90%", "95%", and "99%".
        :param suppress_warnings: (bool) Boolean flag to suppress the cointegration warning message.
        :return: (np.array) The cointegration vector, b.
        """

        # Checking the significance of a test
        if sig_level not in ['90%', '95%', '99%']:
            raise ValueError("Significance level can only be the following:\n "
                             "90%, 95%, or 99%.\n Please check the input.")

        # Calculate the cointegration vector with Johansen test
        jo_portfolio = JohansenPortfolio()

        # Fit portfolio
        jo_portfolio.fit(log_price, det_order=0)

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

    # pylint: disable=invalid-name, too-many-locals
    def get_coint_vec(self) -> Tuple[pd.DataFrame, ...]:
        """
        Generate contegration vector to generate trading signals.

        :return: (np.array) The cointegration vector, b.
        """

        # As it is in sample, calculate the cointegration vector first as it will not change anymore
        all_data = self.calc_log_price(self.__asset_df)
        coint_vec = self.fit(all_data, suppress_warnings=True)

        return coint_vec

    @staticmethod
    def summary(returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Statistics of the trading strategy returns.

        The statistics include: mean, standard deviation, skewness, kurtosis, Sharpe ratio, Sortino ratio,
        final cumulative returns, percentage of up days and down days, max returns, and min returns.

        :param returns_df: (pd.DataFrame) Daily percentage returns dataframe.
        :return: (pd.DataFrame) Trading strategy returns statistics dataframe.
        """

        # Reset column name
        returns_df.columns = [0]

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
