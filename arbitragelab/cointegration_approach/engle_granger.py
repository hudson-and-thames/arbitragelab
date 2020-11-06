# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
This module implements Engle-Granger cointegration approach for statistical arbitrage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from mlfinlab.statistical_arbitrage.base import CointegratedPortfolio


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
        self.ols_model = None  # OLS model used to estimate coefficients.
        self.residuals = None  # OLS model residuals.
        self.dependent_variable = None  # Column name for dependent variable used in OLS estimation.
        self.cointegration_vectors = None  # Regression coefficients used as hedge-ratios.
        self.adf_statistics = None  # ADF statistics.

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
        self.ols_model = LinearRegression(fit_intercept=add_constant)

        X = price_data.copy()
        X.drop(columns=self.dependent_variable, axis=1, inplace=True)
        if X.shape[1] == 1:
            X = X.values.reshape(-1, 1)

        y = price_data[self.dependent_variable].copy()

        self.ols_model.fit(X, y)
        self.cointegration_vectors = pd.DataFrame([np.append(1, -1 * self.ols_model.coef_)],
                                                  columns=price_data.columns)

        # Get model residuals
        self.residuals = y - self.ols_model.predict(X)
        test_result = adfuller(self.residuals)
        critical_values = test_result[4]
        self.adf_statistics = pd.DataFrame(index=['99%', '95%', '90%'], data=critical_values.values())
        self.adf_statistics.loc['statistic_value', 0] = test_result[0]
