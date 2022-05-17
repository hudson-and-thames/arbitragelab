"""
The module implements Hurst exponent calculations.
"""

import numpy as np


def get_hurst_exponent(data: np.array, max_lags: int = 100) -> float:
    """
    Hurst Exponent Calculation.

    :param data: (np.array) Time Series that is going to be analyzed.
    :param max_lags: (int) Maximum amount of lags to be used calculating tau.
    :return: (float) Hurst exponent.
    """

    lags = range(2, max_lags)
    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag])))
           for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    return poly[0] * 2.0
