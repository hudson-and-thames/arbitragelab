# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
This module implements Engle-Granger cointegration approach.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

from arbitragelab.cointegration_approach.base import CointegratedPortfolio
from arbitragelab.util import segment


class EngleGrangerPortfolio(CointegratedPortfolio):
    """
    The class implements the construction of a mean-reverting portfolio using the two-step Engle-Granger method.
    It also tests model residuals for unit-root (presence of cointegration).
    """

    # pylint: disable=invalid-name
    def __init__(self):
        """
        Class constructor method.
        """

        self.price_data = None  # pd.DataFrame with price data used to fit the model.
        self.residuals = None  # OLS model residuals.
        self.dependent_variable = None  # Column name for dependent variable used in OLS estimation.
        self.cointegration_vectors = None  # Regression coefficients used as hedge-ratios.
        self.adf_statistics = None  # ADF statistics.

        segment.track('EngleGrangerPortfolio')

    def perform_eg_test(self, residuals: pd.Series):
        """
        Perform Engle-Granger test on model residuals and generate test statistics and p values.

        :param residuals: (pd.Series) OLS residuals.
        """
        test_result = adfuller(residuals)
        critical_values = test_result[4]
        self.adf_statistics = pd.DataFrame(index=['99%', '95%', '90%'], data=critical_values.values())
        self.adf_statistics.loc['statistic_value', 0] = test_result[0]

    def fit(self, price_data: pd.DataFrame, add_constant: bool = False):
        """
        Finds hedge-ratios using a two-step Engle-Granger method to form a mean-reverting portfolio.
        By default, the first column of price data is used as a dependent variable in OLS estimation.

        This method was originally described in `"Co-integration and Error Correction: Representation,
        Estimation, and Testing," Econometrica, Econometric Society, vol. 55(2), pages 251-276, March 1987
        <https://www.jstor.org/stable/1913236>`_ by Engle, Robert F and Granger, Clive W J.

        :param price_data: (pd.DataFrame) Price data with columns containing asset prices.
        :param add_constant: (bool) A flag to add a constant term in linear regression.
        """

        self.price_data = price_data
        self.dependent_variable = price_data.columns[0]

        # Fit the regression
        hedge_ratios, _, _, residuals = self.get_ols_hedge_ratio(price_data=price_data,
                                                                 dependent_variable=self.dependent_variable,
                                                                 add_constant=add_constant)
        self.cointegration_vectors = pd.DataFrame([np.append(1, -1 * np.array(
            [hedge for ticker, hedge in hedge_ratios.items() if ticker != self.dependent_variable]))],
                                                  columns=price_data.columns)

        # Get model residuals
        self.residuals = residuals
        self.perform_eg_test(self.residuals)

    @staticmethod
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
