# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Utility functions used to construct spreads.
"""

import pandas as pd


def construct_spread(price_data: pd.DataFrame, hedge_ratios: pd.Series, dependent_variable: str = None) -> pd.Series:
    """
    Construct spread from `price_data` and `hedge_ratios`. If a user sets `dependent_variable` it means that a
    spread will be: hedge_ratio_dependent_variable * dependent_variable - sum(hedge_ratios * other variables).
    Otherwise, spread is:  hedge_ratio_0 * variable_0 - sum(hedge ratios * variables[1:]).

    :param price_data: (pd.DataFrame) Asset prices data frame.
    :param hedge_ratios: (pd.Series) Hedge ratios series (index-tickers, values-hedge ratios).
    :param dependent_variable: (str) Dependent variable to use. Set None for dependent variable being equal to 0 column.
    :return: (pd.Series) Spread series.
    """

    weighted_prices = price_data * hedge_ratios  # price * hedge
    if dependent_variable is not None:
        non_dependent_variables = [x for x in weighted_prices.columns if x != dependent_variable]
        return weighted_prices[dependent_variable] - weighted_prices[non_dependent_variables].sum(axis=1)

    return weighted_prices[weighted_prices.columns[0]] - weighted_prices[weighted_prices.columns[1:]].sum(axis=1)
