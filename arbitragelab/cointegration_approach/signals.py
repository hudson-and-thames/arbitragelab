# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Module for signal generation for Cointegration Approach.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from arbitragelab.util import segment


def get_half_life_of_mean_reversion(data: pd.Series) -> float:
    """
    Get half-life of mean-reversion under the assumption that data follows the Ornstein-Uhlenbeck process.

    :param data: (np.array) Data points
    :return: (float) Half-life of mean reversion
    """

    segment.track('get_half_life_of_mean_reversion')

    reg = LinearRegression(fit_intercept=True)

    training_data = data.shift(1).dropna().values.reshape(-1, 1)
    target_values = data.diff().dropna()
    reg.fit(X=training_data, y=target_values)

    half_life = -np.log(2) / reg.coef_[0]

    return half_life


def linear_trading_strategy(portfolio_series: pd.Series, sma_window: int = None,
                            std_window: int = None) -> pd.DataFrame:
    """
    This function implements a simple linear mean-reverting strategy from the book
    by E.P Chan: `"Algorithmic Trading: Winning Strategies and Their Rationale"
    <https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146>`_,
    page 59. The strategy uses mean-reverting portfolio series representing the dollar value of portfolio
    formed by multiplying hedge vector by asset prices
    (hedge vector can be formed by either OLS regression or cointegration vector, Kalman filter).

    :param portfolio_series: (pd.Series) Value of a portfolio used to build trading signals.
    :param sma_window: (int) Window for SMA (Simple Moving Average).
    :param std_window: (int) Window for SMD (Simple Moving st. Deviation).
    :return: (pd.DataFrame) Mean-reverting portfolio series and target allocation on each day.
    """

    segment.track('linear_trading_strategy')

    if sma_window is None:
        # The book suggests to use window = speed of reversion
        sma_window = int(get_half_life_of_mean_reversion(portfolio_series))

    if std_window is None:
        std_window = int(get_half_life_of_mean_reversion(portfolio_series))

    z_score_series = (portfolio_series - portfolio_series.rolling(window=sma_window).mean()) / \
                     portfolio_series.rolling(window=std_window).std()

    results_df = pd.DataFrame(index=portfolio_series.index)

    results_df['portfolio_series'] = portfolio_series
    results_df['z_score'] = z_score_series
    results_df['target_quantity'] = -z_score_series

    return results_df


def bollinger_bands_trading_strategy(portfolio_series: pd.Series, sma_window: int = None, std_window: int = None,
                                     entry_z_score: float = 3, exit_z_score: float = -3) -> pd.DataFrame:
    """
    This function implements Bollinger Bands strategy from the book
    by E.P Chan: `"Algorithmic Trading: Winning Strategies and Their Rationale"
    <https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146>`_,
    page 70. The strategy is the extension of Simple Linear Strategy (page.59) by trading only when
    ``|z_score| >= entry_z_score`` and exit from a position when
    ``|z_score| <= exit_z_score``.

    :param portfolio_series: (pd.Series) Value of a portfolio used build trading signals.
    :param sma_window: (int) Window for SMA (Simple Moving Average).
    :param std_window: (int) Window for SMD (Simple Moving st. Deviation).
    :param entry_z_score: (float) Z-score value to enter (long or short) the position.
    :param exit_z_score: (float) Z-score value to exit (long or short) the position.
    :return: (pd.DataFrame) Mean-reverting portfolio series and target allocation on each day.
    """

    segment.track('bollinger_bands_trading_strategy')

    if exit_z_score >= entry_z_score:
        raise ValueError('Exit Z-score can not be bigger than entry Z-Score.')

    results_df = linear_trading_strategy(portfolio_series, sma_window, std_window)

    long_entry_index = results_df[results_df['z_score'] < -entry_z_score].index
    long_exit_index = results_df[results_df['z_score'] >= -exit_z_score].index

    short_entry_index = results_df[results_df['z_score'] > entry_z_score].index
    short_exit_index = results_df[results_df['z_score'] <= exit_z_score].index

    results_df['long_units'] = np.nan
    results_df['short_units'] = np.nan
    results_df.iloc[0, results_df.columns.get_loc('long_units')] = 0
    results_df.iloc[0, results_df.columns.get_loc('short_units')] = 0

    results_df.loc[long_entry_index, 'long_units'] = 1
    results_df.loc[long_exit_index, 'long_units'] = 0
    results_df.loc[short_entry_index, 'short_units'] = -1
    results_df.loc[short_exit_index, 'short_units'] = 0

    results_df.fillna(method='pad', inplace=True)
    results_df['target_quantity'] = results_df['long_units'] + results_df['short_units']

    return results_df[['portfolio_series', 'z_score', 'target_quantity', 'long_units', 'short_units']]
