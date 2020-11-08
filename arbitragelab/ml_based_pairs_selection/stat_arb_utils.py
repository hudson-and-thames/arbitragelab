"""
This module houses utility functions used by the PairsSelector
"""

import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import adfuller
from scipy.odr import ODR, Model, RealData

def _outer_ou_loop(spreads_df: pd.DataFrame, test_period: str, cross_overs_per_delta: int, molecule: list) -> pd.DataFrame:
    """
    This function gets mean reversion calculations (half life and number of mean cross overs) for each pair in the molecule.
    Uses the linear regression method to get the half life, which is much lighter computationally wise compared to the version
    using the OrnsteinUhlenbeck class.

    :param spreads_df: (pd.DataFrame) Spreads Universe
    :param test_period: (str) Time delta format, to be used as the time period where the mean crossovers will be calculated
    :param cross_overs_per_delta: (int) Crossovers per time delta selected
    :param molecule: (list) Indices of pairs
    :return: (pd.DataFrame) Mean Reversion statistics
    """

    ou_results = []

    for pair in molecule:

        spread = spreads_df.loc[:, str(pair)]

        if len(spread.shape) > 1:
            spread = pd.Series(np.squeeze(spread.values))

        lagged_spread = spread.shift(1).dropna(0)
        lagged_spread_c = sm.add_constant(lagged_spread)

        delta_y_t = np.diff(spread)

        model = sm.OLS(delta_y_t, lagged_spread_c)
        res = model.fit()

        # Note that when mean reversion is expected, λ / SE has a negative value.
        # res.params[0] / res.bse[0]

        # This result implies that the expected duration of mean reversion λ is
        # inversely proportional to the absolute value of λ

        # h = np.log(2) / abs(res.params[0])

        # split the spread in two; the train_df is going to be used to get the long term mean
        # and test_df is going to be used to get the number of mean cross overs

        test_df = spread.last(test_period)

        train_df = spread.iloc[: -len(test_df)]

        long_term_mean = np.mean(train_df)

        centered_series = test_df - long_term_mean

        cross_over_indices = np.where(np.diff(np.sign(centered_series)))[0]

        cross_overs_dates = spreads_df.index[cross_over_indices]

        cross_overs_counts = cross_overs_dates.to_frame().resample('Y').count()

        cross_overs_counts.columns = ['counts']

        if cross_overs_counts.empty:
            cross_overs = False
        else:
            cross_overs = len(cross_overs_counts[cross_overs_counts['counts'] > cross_overs_per_delta]) > 0

        ou_results.append([np.log(2) / abs(res.params[0]), cross_overs])

    return pd.DataFrame(ou_results, index=molecule, columns=['hl', 'crossovers'])

def linear_f(beta, x_variable):
    """
    This is the helper linear model that is going to be used in the Orthogonal Regression.

    :param beta: (np.array) Model beta coefficient
    :param x_variable: (np.array) Model X vector
    :return: (np.array)
    """
    return beta[0]*x_variable + beta[1]

def _outer_cointegration_loop(prices_df: pd.DataFrame, molecule: list) -> pd.DataFrame:
    """
    This function calculates the Engle-Granger test for each pair in the molecule. Uses the Total
    Least Squares approach to take into consideration the variance of both price series.

    :param prices_df: (pd.DataFrame) Price Universe
    :param molecule: (list) Indices of pairs
    :return: (pd.DataFrame) Cointegration statistics
    """

    cointegration_results = []

    for pair in molecule:
        maxlag = None
        autolag = "aic"
        trend = "c"

        linear = Model(linear_f)
        mydata = RealData(prices_df.loc[:, pair[0]], prices_df.loc[:, pair[1]])
        myodr = ODR(mydata, linear, beta0=[1., 2.])
        res_co = myodr.run()

        res_adf = adfuller(
            res_co.delta - res_co.eps, maxlag=maxlag, autolag=autolag, regression="nc"
        )

        pval_asy = mackinnonp(res_adf[0], regression=trend)

        cointegration_results.append(
            (res_adf[0], pval_asy, res_co.beta[0], res_co.beta[1]))

    return pd.DataFrame(cointegration_results, index=molecule, columns=['coint_t', 'pvalue', 'hedge_ratio', 'constant'])
