# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This module implements the TAR model by Ender and Granger (1998).
"""

import numpy as np
import pandas as pd 
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults

from arbitragelab.cointegration_approach.johansen import JohansenPortfolio

class TAR(JohansenPortfolio):
    """
    Threshold AutoRegressive Model.
    """
    
    def __init__(self, price_data: pd.DataFrame, calculate_spread: bool = True):
        """
        Init function.

        :param price_data: (pd.DataFrame) Collection of time series to
            construct to spread from.
        """
        
        if calculate_spread:
            super().fit(price_data)
            self.spread = super().construct_mean_reverting_portfolio(price_data)
        else:
            self.spread = price_data
            
            
    def _tag_regime(self, series: pd.Series) -> pd.DataFrame:
        """
        Tags up/down swings in different vectors.

        :param series: (pd.Series) Time series to tag.
        :return: (pd.DataFrame) Original series with two new columns
            with values [0,1] indicating down/up swings.
        """

        tagged_df = series.copy().to_frame()
        tagged_df.columns = ['y_{t-1}']
        tagged_df['I_{1}'] = 0
        tagged_df['I_{0}'] = 0
        tagged_df.loc[tagged_df['y_{t-1}'] >= 0, 'I_{1}'] = 1
        tagged_df.loc[tagged_df['y_{t-1}'] < 0, 'I_{0}'] = 1

        return tagged_df.dropna()
    
    def fit(self) -> RegressionResults:
        """
        Fits the OLS model.

        :return: (RegressionResults) 
        """

        jspread = pd.DataFrame(self.spread.values)
        jspread.columns = ['spread']
        jspread['rets'] = jspread['spread']
        jspread['rets'] = jspread['rets'].diff()
        jspread['spread_lag1'] = jspread['spread'].shift(1)
        jspread.dropna(inplace=True)

        Y = jspread['rets']

        X = jspread['spread_lag1']

        tagged_Y = self._tag_regime(X)

        regime_one = tagged_Y['y_{t-1}'] * tagged_Y['I_{1}']
        regime_two = tagged_Y['y_{t-1}'] * tagged_Y['I_{0}']
        
        regime_X = pd.concat([regime_one, regime_two], axis=1)

        regime_X.columns = ['p_1', 'p_2']

        model = sm.OLS(Y.values, regime_X)
        results = model.fit()
        self.results = results

        return results
    
    def summary(self) -> pd.DataFrame:
        """
        Returns summary as in paper.

        :return: (pd.DataFrame) Summary of results.
        """

        coefficient_1 = self.results.params.loc['p_1']
        pvalue_1 = self.results.wald_test('p_1 = 0').pvalue


        coefficient_2 = self.results.params.loc['p_2']
        pvalue_2 = self.results.wald_test('p_2 = 0').pvalue

        equiv_fvalue = self.results.wald_test('p_1 = p_2').fvalue
        equiv_pvalue = self.results.wald_test('p_1 = p_2').pvalue


        tuple_frame = [(coefficient_1, None, pvalue_1),
                       (coefficient_2, None, pvalue_2),
                       (None, equiv_fvalue[0][0], equiv_pvalue)]

        result_frame = pd.DataFrame(tuple_frame).T
        result_frame.columns = ['p_1', 'p_2', 'p_1 = p_2']
        result_frame['index'] = ['Coefficient', 'F-stat', 'p-value']
        result_frame.set_index('index', inplace=True)

        return result_frame.astype(float)