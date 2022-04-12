"""
Johansen hedge ratio calculation.
"""

from typing import Tuple

import pandas as pd

from arbitragelab.cointegration_approach import JohansenPortfolio
from arbitragelab.hedge_ratios.spread_construction import construct_spread


# pylint: disable=invalid-name

def get_johansen_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[
    dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get hedge ratio from Johansen test eigen vector..

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and y and OLS fit residuals.
    """
    port = JohansenPortfolio()
    port.fit(price_data)

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()
    # Convert to a format expected by `construct_spread` function and normalize such that dependent has a hedge ratio 1.
    hedge_ratios = port.cointegration_vectors.iloc[0].to_dict()
    for ticker, h in hedge_ratios.items():
        if ticker != dependent_variable:
            hedge_ratios[ticker] = -h / hedge_ratios[dependent_variable]
    hedge_ratios[dependent_variable] = 1.0

    residuals = construct_spread(price_data, hedge_ratios=hedge_ratios, dependent_variable=dependent_variable)

    # Normalize Johansen cointegration vectors such that dependent variable has a hedge ratio of 1.
    return hedge_ratios, X, y, residuals
