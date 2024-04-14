"""
Johansen hedge ratio calculation.
"""
# pylint: disable=invalid-name

from typing import Tuple

import pandas as pd

from arbitragelab.cointegration_approach import JohansenPortfolio
from arbitragelab.hedge_ratios.spread_construction import construct_spread
from arbitragelab.util import segment


def get_johansen_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[
    dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get hedge ratio from Johansen test eigenvector.

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and y and OLS fit residuals.
    """

    segment.track('get_johansen_hedge_ratio')

    # Construct a Johansen portfolio
    port = JohansenPortfolio()
    port.fit(price_data, dependent_variable)

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()

    # Convert to a format expected by `construct_spread` function and normalize such that dependent has a hedge ratio 1.
    hedge_ratios = port.hedge_ratios.iloc[0].to_dict()

    residuals = construct_spread(price_data, hedge_ratios=hedge_ratios, dependent_variable=dependent_variable)

    # Normalize Johansen cointegration vectors such that dependent variable has a hedge ratio of 1.
    return hedge_ratios, X, y, residuals
