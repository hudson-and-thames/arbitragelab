# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Base class for cointegration approach in statistical arbitrage.
"""

from abc import ABC
import pandas as pd


class CointegratedPortfolio(ABC):
    """
    Class for portfolios formed using the cointegration method (Johansen test, Engle-Granger test).
    """

    def construct_mean_reverting_portfolio(self, price_data: pd.DataFrame,
                                           cointegration_vector: pd.Series = None) -> pd.Series:
        """
        When cointegration vector was formed, this function is used to multiply asset prices by cointegration vector
        to form mean-reverting portfolio which is analyzed for possible trade signals.

        :param price_data: (pd.DataFrame) Price data with columns containing asset prices.
        :param cointegration_vector: (pd.Series) Cointegration vector used to form a mean-reverting portfolio.
            If None, a cointegration vector with maximum eigenvalue from fit() method is used.
        :return: (pd.Series) Cointegrated portfolio dollar value.
        """

        if cointegration_vector is None:
            cointegration_vector = self.cointegration_vectors.iloc[0]  # Use eigenvector with biggest eigenvalue.

        return (cointegration_vector * price_data).sum(axis=1)
