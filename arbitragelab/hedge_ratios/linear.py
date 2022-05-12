# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
The module implements OLS (Ordinary Least Squares) and TLS (Total Least Squares) hedge ratio calculations.
"""
# pylint: disable=invalid-name

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.odr import ODR, Model, RealData


def get_ols_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str, add_constant: bool = False) -> \
        Tuple[dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get OLS hedge ratio: y = beta*X.

    :param price_data: (pd.DataFrame) Data Frame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :param add_constant: (bool) Boolean flag to add constant in regression setting.
    :return: (Tuple) Hedge ratios, X, and y and OLS fit residuals.
    """

    ols_model = LinearRegression(fit_intercept=add_constant)

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)
    exogenous_variables = X.columns.tolist()
    if X.shape[1] == 1:
        X = X.values.reshape(-1, 1)

    y = price_data[dependent_variable].copy()

    ols_model.fit(X, y)
    residuals = y - ols_model.predict(X)

    hedge_ratios = ols_model.coef_
    hedge_ratios_dict = dict(zip([dependent_variable] + exogenous_variables, np.insert(hedge_ratios, 0, 1.0)))

    return hedge_ratios_dict, X, y, residuals


def _linear_f_no_constant(beta: np.array, x_variable: np.array) -> np.array:
    """
    This is the helper linear model that is used in the Orthogonal Regression.

    :param beta: (np.array) Model beta coefficient.
    :param x_variable: (np.array) Model X vector.
    :return: (np.array) Vector result of equation calculation.
    """

    _, b = beta[0], beta[1:]
    b.shape = (b.shape[0], 1)

    return (x_variable * b).sum(axis=0)


def _linear_f_constant(beta: np.array, x_variable: np.array) -> np.array:
    """
    This is the helper linear model that is used in the Orthogonal Regression.

    :param beta: (np.array) Model beta coefficient.
    :param x_variable: (np.array) Model X vector.
    :return: (np.array) Vector result of equation calculation.
    """

    a, b = beta[0], beta[1:]
    b.shape = (b.shape[0], 1)

    return a + (x_variable * b).sum(axis=0)


def get_tls_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str, add_constant: bool = False) -> \
        Tuple[dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get Total Least Squares (TLS) hedge ratio using Orthogonal Regression.

    :param price_data: (pd.DataFrame) Data Frame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :param add_constant: (bool) Boolean flag to add constant in regression setting.
    :return: (Tuple) Hedge ratios dict, X, and y and fit residuals.
    """

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)
    y = price_data[dependent_variable].copy()

    linear = Model(_linear_f_constant) if add_constant is True else Model(_linear_f_no_constant)
    mydata = RealData(X.T, y)
    myodr = ODR(mydata, linear, beta0=np.ones(X.shape[1] + 1))
    res_co = myodr.run()

    hedge_ratios = res_co.beta[1:]  # We don't need constant
    residuals = y - res_co.beta[0] - (X * hedge_ratios).sum(axis=1) if add_constant is True else y - (
            X * hedge_ratios).sum(axis=1)
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))

    return hedge_ratios_dict, X, y, residuals
