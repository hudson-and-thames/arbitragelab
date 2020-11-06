# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
This module implements the Johansen cointegration approach for statistical arbitrage.
"""

import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from mlfinlab.statistical_arbitrage.base import CointegratedPortfolio


class JohansenPortfolio(CointegratedPortfolio):
    """
    The class implements the construction of a mean-reverting portfolio using eigenvectors from
    the Johansen cointegration test. It also checks Johansen (eigenvalue and trace statistic) tests
    for the presence of cointegration for a given set of assets.
    """

    def __init__(self):
        """
        Class constructor.
        """

        self.price_data = None  # pd.DataFrame with price data used to fit the model.
        self.cointegration_vectors = None  # Johansen eigenvectors used to form mean-reverting portfolios.
        self.johansen_trace_statistic = None  # Trace statistic data frame for each asset used to test for cointegration.
        self.johansen_eigen_statistic = None  # Eigenvalue statistic data frame for each asset used to test for cointeg.

    def fit(self, price_data: pd.DataFrame, det_order: int = 0, n_lags: int = 1):
        """
        Finds cointegration vectors from the Johansen test used to form a mean-reverting portfolio.

        Note: Johansen test yields several linear combinations that may yield mean-reverting portfolios. The function
        stores all of them in decreasing order of eigenvalue meaning that the first linear combination forms the most
        mean-reverting portfolio which is used in trading. However, researchers may use other stored cointegration vectors
        to check other portfolios.

        This function will calculate and set johansen_trace_statistic and johansen_eigen_statistic only if
        the number of variables in the input dataframe is <=12. Otherwise it will generate a warning.

        A more detailed description of this method can be found on p. 54-58 of
        `"Algorithmic Trading: Winning Strategies and Their Rationale" by Ernie Chan
        <https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146>`_.

        This function is a wrapper around the coint_johansen function from the statsmodels.tsa module. Detailed
        descriptions of this function are available in the
        `statsmodels documentation
        <https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html>`_.

        :param price_data: (pd.DataFrame) Price data with columns containing asset prices.
        :param det_order: (int) -1 for no deterministic term in Johansen test, 0 - for constant term, 1 - for linear trend.
        :param n_lags: (int) Number of lags used in the Johansen test. The practitioners use 1 as the default base value.
        """

        self.price_data = price_data

        test_res = coint_johansen(price_data, det_order=det_order, k_ar_diff=n_lags)

        # Store eigenvectors in decreasing order of eigenvalues
        self.cointegration_vectors = pd.DataFrame(test_res.evec[:, test_res.ind].T, columns=price_data.columns)

        # Test critical values are available only if number of variables <= 12
        if price_data.shape[1] <= 12:
            # Eigenvalue test
            self.johansen_eigen_statistic = pd.DataFrame(test_res.max_eig_stat_crit_vals.T,
                                                         columns=price_data.columns, index=['90%', '95%', '99%'])
            self.johansen_eigen_statistic.loc['eigen_value'] = test_res.max_eig_stat.T
            self.johansen_eigen_statistic.sort_index(ascending=False)

            # Trace statistic
            self.johansen_trace_statistic = pd.DataFrame(test_res.trace_stat_crit_vals.T,
                                                         columns=price_data.columns, index=['90%', '95%', '99%'])
            self.johansen_trace_statistic.loc['trace_statistic'] = test_res.trace_stat.T
            self.johansen_trace_statistic.sort_index(ascending=False)
