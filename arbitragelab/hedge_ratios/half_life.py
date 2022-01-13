# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Module which implements Minimum Half-Life Hedge Ratio detection algorithm.
"""
# pylint: disable=invalid-name

from typing import Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from arbitragelab.cointegration_approach.signals import get_half_life_of_mean_reversion


def _min_hl_function(beta: np.array, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Fitness function to minimize in Minimum Half-Life Hedge Ratio algorithm.

    :param beta: (np.array) Array of hedge ratios.
    :param X: (pd.DataFrame) DataFrame of dependent variables. We hold `beta` units of X assets.
    :param y: (pd.Series) Series of target variable. For this asset we hold 1 unit.
    :return: (float) Half-life of mean-reversion.
    """

    spread = y - (beta * X).sum(axis=1)

    return abs(get_half_life_of_mean_reversion(spread))


def get_minimum_hl_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> \
        Tuple[dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get hedge ratio by minimizing spread half-life of mean reversion.

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and y and OLS fit residuals.
    """

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()
    initial_guess = (y[0] / X).mean().values
    result = minimize(_min_hl_function, x0=initial_guess, method='BFGS', tol=1e-5, args=(X, y))
    residuals = y - (result.x * X).sum(axis=1)

    hedge_ratios = result.x
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))

    return hedge_ratios_dict, X, y, residuals
