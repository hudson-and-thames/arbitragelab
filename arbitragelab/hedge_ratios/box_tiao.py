# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Hedge ratio estimation using Box-Tiao canonical decomposition on the assets dataframe.
"""
# pylint: disable=invalid-name

from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from arbitragelab.hedge_ratios.spread_construction import construct_spread


def _least_square_VAR_fit(demeaned_price_data: pd.DataFrame) -> np.array:
    """
    Calculate the least square estimate of the VAR(1) matrix.

    :param demeaned_price_data: (pd.DataFrame) Demeaned price data.
    :return: (np.array) Least square estimate of VAR(1) matrix.
    """

    # Fit VAR(1) model
    var_model = sm.tsa.VAR(demeaned_price_data)

    # The statsmodels package will give the least square estimate
    least_sq_est = np.squeeze(var_model.fit(1).coefs, axis=0)

    return least_sq_est


def get_box_tiao_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> \
        Tuple[dict, pd.DataFrame, None, pd.Series]:
    """
    Perform Box-Tiao canonical decomposition on the assets dataframe.

    The resulting ratios are the weightings of each asset in the portfolio. There are N decompositions for N assets,
    where each column vector corresponds to one portfolio. The order of the weightings corresponds to the
    descending order of the eigenvalues.

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and fit residuals.
    """

    X = price_data.copy()
    X = X[[dependent_variable] + [x for x in X.columns if x != dependent_variable]]

    demeaned = X - X.mean()  # Subtract mean columns

    # Calculate the least square estimate of the price with VAR(1) model
    least_sq_est = _least_square_VAR_fit(demeaned)

    # Construct the matrix from which the eigenvectors need to be computed
    covar = demeaned.cov()
    box_tiao_matrix = np.linalg.inv(covar) @ least_sq_est @ covar @ least_sq_est.T

    # Calculate the eigenvectors and sort by eigenvalue
    eigvals, eigvecs = np.linalg.eig(box_tiao_matrix)

    # Sort the eigenvectors by eigenvalues by descending order
    bt_eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
    hedge_ratios = dict(zip(X.columns, bt_eigvecs[:, -1]))

    # Convert to a format expected by `construct_spread` function and normalize such that dependent has a hedge ratio 1
    for ticker, h in hedge_ratios.items():
        if ticker != dependent_variable:
            hedge_ratios[ticker] = -h / hedge_ratios[dependent_variable]
    hedge_ratios[dependent_variable] = 1.0

    residuals = construct_spread(price_data, hedge_ratios=hedge_ratios, dependent_variable=dependent_variable)

    # Return the weights
    return hedge_ratios, X, None, residuals
