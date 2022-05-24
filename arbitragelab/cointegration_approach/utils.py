# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Various utility functions used in cointegration/mean-reversion trading.
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


def get_hurst_exponent(data: np.array, max_lags: int = 100) -> float:
    """
    Hurst Exponent Calculation.

    :param data: (np.array) Time Series that is going to be analyzed.
    :param max_lags: (int) Maximum amount of lags to be used calculating tau.
    :return: (float) Hurst exponent.
    """

    segment.track('get_hurst_exponent')
    lags = range(2, max_lags)
    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag])))
           for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    return poly[0] * 2.0
