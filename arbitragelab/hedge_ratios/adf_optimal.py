# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Implementation of finding ADF optimal hedge ratio.
"""
# pylint: disable=invalid-name
# pylint: disable=protected-access

from typing import Tuple
import warnings
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from arbitragelab.cointegration_approach.engle_granger import EngleGrangerPortfolio


def _min_adf_stat(beta: np.array, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Fitness function to minimize in ADF test statistic algorithm.

    :param beta: (np.array) Array of hedge ratios.
    :param X: (pd.DataFrame) DataFrame of dependent variables. We hold `beta` units of X assets.
    :param y: (pd.Series) Series of target variable. For this asset we hold 1 unit.
    :return: (float) Half-life of mean-reversion.
    """

    # Performing Engle-Granger test on spread
    portfolio = EngleGrangerPortfolio()
    spread = y - (beta * X).sum(axis=1)
    portfolio._perform_eg_test(spread)

    return portfolio.adf_statistics.loc['statistic_value'].iloc[0]


def get_adf_optimal_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> \
        Tuple[dict, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Get hedge ratio by minimizing ADF test statistic.

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and y, OLS fit residuals and optimization object.
    """

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()
    initial_guess = (y[0] / X).mean().values
    result = minimize(_min_adf_stat, x0=initial_guess, method='BFGS', tol=1e-5, args=(X, y))
    residuals = y - (result.x * X).sum(axis=1)

    hedge_ratios = result.x
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))
    if result.status != 0:
        warnings.warn('Optimization failed to converge. Please check output hedge ratio! The result can be unstable!')

    return hedge_ratios_dict, X, y, residuals, result
