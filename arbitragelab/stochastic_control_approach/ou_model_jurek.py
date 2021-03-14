# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module implements the optimal dynamic portfolio strategy for arbitrageurs with a finite horizon and
non-myopic preferences facing a mean-reverting arbitrage opportunity using OU process.

This module is a realization of the methodology in the following paper:
`Jurek, J.W. and Yang, H., 2007, April. Dynamic portfolio selection in arbitrage. In EFA 2006 Meetings Paper.
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=882536>`__
"""

import numpy as np
import pandas as pd

class StochasticControlJurek:
    def __init__(self):

        self.ticker_A = None
        self.ticker_B = None
        self.spread = None
        self.time_array = None

        self.delta_t = 1 / 251   #TODO : Is this value correct for this paper?
        self.sigma = None
        self.mu = None
        self.k = None

        self.gamma = None

    def fit(self, data: pd.DataFrame):

        self.time_array = np.arange(0, len(data)) / self.delta_t
        self.ticker_A, self.ticker_B = data.columns[0], data.columns[1]

        returns_df = data.pct_change()
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().dropna()

        total_return_indices = data.copy()
        total_return_indices.iloc[0, :] = 1
        total_return_indices.iloc[1:, :] = pd.DataFrame.cumprod(1 + returns_df, axis=0)

        # df = data.iloc[:round(len(data) * 0.1), :].mean(axis=0)
        # mispricing_percent = df[self.ticker_B] / df[self.ticker_A]
        # print(mispricing_percent)
        # total_return_indices[self.ticker_A]  *= mispricing_percent

        #TODO : Need to figure out weights for return indices for spread calcuation

        self.spread = total_return_indices[self.ticker_A] - total_return_indices[self.ticker_B]

        self._estimate_params()

    def _estimate_params(self):

        N = len(self.spread)

        self.mu = self.spread.mean()

        self.k = (-1 / self.delta_t) * np.log(np.multiply(self.spread[1:] - self.mu, self.spread[:-1] - self.mu).sum()
                                              / np.power(self.spread[1:] - self.mu, 2).sum())

        sigma_calc_sum = np.power(self.spread[1:] - self.mu - np.exp(-self.k*self.delta_t)*(self.spread[:-1] - self.mu)
                                  / np.exp(-self.k*self.delta_t), 2).sum()
        #TODO : Check for error in paper for this summation?

        self.sigma = np.sqrt(2*self.k*sigma_calc_sum/((np.exp(2*self.k*self.delta_t) - 1)*(N - 2)))
