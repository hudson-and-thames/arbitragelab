"""
The module implements OLS (Ordinary Least Squares) and TLS (Total Least Squares) hedge ratio calculations.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.odr import ODR, Model, RealData


# pylint: disable=invalid-name
def get_ols_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str, add_constant: bool = False) -> \
        Tuple[object, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get OLS hedge ratio: y = beta*X.

    :param price_data: (pd.DataFrame) Data Frame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :param add_constant: (bool) Boolean flag to add constant in regression setting.
    :return: (Tuple) Fit OLS, X, and y and OLS fit residuals.
    """
    ols_model = LinearRegression(fit_intercept=add_constant)

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)
    if X.shape[1] == 1:
        X = X.values.reshape(-1, 1)

    y = price_data[dependent_variable].copy()

    ols_model.fit(X, y)
    residuals = y - ols_model.predict(X)
    return ols_model, X, y, residuals


# pylint: disable=invalid-name
def _linear_f(beta: np.array, x_variable: np.array) -> np.array:
    """
    This is the helper linear model that is going to be used in the Orthogonal Regression.
    :param beta: (np.array) Model beta coefficient.
    :param x_variable: (np.array) Model X vector.
    :return: (np.array) Vector result of equation calculation.
    """

    return beta[0] * x_variable


# pylint: disable=invalid-name
def get_tls_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> \
        Tuple[object, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get Total Least Squares (TLS) hedge ratio using Orthogonal Regression.

    :param price_data: (pd.DataFrame) Data Frame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Fit TLS object, X, and y and fit residuals.
    """
    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)
    y = price_data[dependent_variable].copy()

    linear = Model(_linear_f)
    mydata = RealData(X.squeeze(), y)
    myodr = ODR(mydata, linear, beta0=[1.0])
    res_co = myodr.run()
    residuals = y - X.squeeze()*res_co.beta[0]

    return res_co, X, y, residuals
