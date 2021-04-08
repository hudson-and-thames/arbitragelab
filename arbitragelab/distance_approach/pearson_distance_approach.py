# Copyright 2021, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Implementation of the statistical arbitrage distance approach proposed by
Gatev, E., Goetzmann, W. N., and Rouwenhorst, K. G. in
"Pairs trading:  Performance of a relative-value arbitrage rule." (2006)
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615.
"""

import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from arbitragelab.util import devadarsh


class PearsonStrategy:
    """
    Class for creation of portfolios following the strategy by Chen, H., Chen, S. J., and Li, F.
    in "Empirical Investigation of an Equity Pairs Trading Strategy." (2012)
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1361293.
    """

    def __init__(self):
        """
        Initialize Pearson strategy.
        """

        # Internal Parameters
        self.monthly_return = None  # Monthly return dataset in formation period
        self.risk_free = None  # Risk free rate dataset for calculating return differences
        self.beta_dict = None  # Regression coefficients for each stock in the formation period
        self.pairs_dict = None  # Top n pairs selected during the formation period
        self.short_stocks = None  # Stocks having lowest values of return differences
        self.long_stocks = None  # Stocks having highest values of return differences
        self.return_diff_sorted = None  # Sorted dictionary of return difference in the test period
        self.normalized_prices = None  # Normalized prices of the train data

        devadarsh.track('PearsonStrategy')

    def form_portfolio(self, train_data, risk_free=None, num_pairs=50, weight='equal'):
        """
        Forms portfolio based on the input train data.

        For each stock i in year t+1(or the last month of year t), this method computes the Pearson correlation
        coefficients between the returns of stock i and all other stocks in the given train data for all periods in
        the train data. Usually, the formation period is set to be 4 years but it may be changed upon the user’s needs.

        Then the method finds top n stocks with the highest correlations to stock i as its pairs in the formation
        period. For each month in year t+1, this method computes the pairs portfolio return as the equal-weighted
        average return of the n pairs stocks from the previous month.

        The hypothesis in this method is that if a stock i’s return deviates more from its pairs portfolio returns
        than usual, this divergence should be reversed in the next month and expecting abnormally higher returns than
        other stocks.

        In this method, for stock i, this method uses a new variable, return difference, which captures the return
        divergence between i’s stock return and its pairs-portfolio return.

        After calculating the “return differences” of all stocks for the last month of the formation period,
        this method constructs decile portfolios where stocks with high return differences have higher subsequent
        returns. Therefore after all stocks are sorted in descending order based on their previous month’s return
        divergence, decile 10 is “long stocks” and decile 1 is “short stocks”.

        :param train_data: (pd.DataFrame) Daily price data with date in its index and stocks in its columns.
        :param risk_free: (pd.Series) Daily risk free rate data with date in its index.
        :param num_pairs: (int) Number of top pairs to use for portfolio formation.
        :param weight: (str) Weighting Scheme for portfolio returns [``equal`` by default, ``correlation``].
        """

        # Preprocess data to get monthly return from daily price data
        self.data_preprocess(train_data, risk_free)

        # Calculating beta and getting pairs in the formation period
        self.beta_pairs_formation(num_pairs, weight)

        # Build portfolio based on the return difference in the last period
        last_month = self.monthly_return.iloc[-1, :]

        # Calculate the return differences of all stocks in the train data
        return_diff_dict = self.calculate_return_diff(last_month)

        # Sort the dictionary based on the return differences value
        self.return_diff_sorted = {k: v for k, v in sorted(return_diff_dict.items(), key=lambda item: item[1])}

        # Get the number of stocks to get two different portfolios
        num_stocks = len(self.return_diff_sorted)

        # Stocks with bottom 10% value of the return differences should be shorted
        self.short_stocks = dict(itertools.islice(self.return_diff_sorted.items(), 0, int(num_stocks * 0.1)))

        # Stocks with top 10% value of the return differences should be longed
        self.long_stocks = dict(itertools.islice(self.return_diff_sorted.items(), int(num_stocks * 0.9), num_stocks))

    def trade_pairs(self, test_data, methods):


    def get_short_stocks(self):
        """
        Outputs generated stocks in decile 1 which have low return differences.

        :return: (dict) A dictionary with stocks in its key and the value of return differences in its value.
        """

        return self.short_stocks

    def get_long_stocks(self):
        """
        Outputs generated stocks in decile 10 which have high return differences.

        :return: (dict) A dictionary with stocks in its key and the value of return differences in its value.
        """

        return self.long_stocks

    def get_return_diff(self):
        """
        Outputs the whole dictionary of return difference calculated in the last month of the formation period
        :return: (dict) A dictionary with stocks in its key and the value of return differences in its value.
        """

        return self.return_diff_sorted

    def _data_preprocess(self, train_data, risk_free):
        """
        Preprocess train data and risk free data.

        As monthly return data is used to calculate the beta and Pearson correlation in the formation period,
        it is needed to preprocess the train data and risk free data.

        :param train_data: (pd.DataFrame) Daily price data with date in its index and stocks in its columns.
        :param risk_free: (pd.Series) Daily risk free rate data with date in its index.
        """

        # Calculate normalized prices with mean and standard deviation
        data_copy = train_data.copy()
        self.normalized_prices = (data_copy - data_copy.mean()) / (data_copy.std())

        # Change the daily price data into daily return by calculating the percent change
        daily_return = train_data.pct_change()

        # Change the index of the train data into string to group by months later
        daily_return.index = daily_return.index.map(lambda x: pd.to_datetime(str(x)))

        # Add 1 to the values to calculate the compound return
        daily_return = daily_return + 1

        # Calculate monthly return for the given data
        monthly_return = daily_return.groupby([daily_return.index.year, daily_return.index.month]).prod()

        self.monthly_return = monthly_return

        if risk_free is not None:
            # Get risk free rate
            risk_free = risk_free.rename('risk_free')
            risk_free.index = risk_free.index.map(lambda x: pd.to_datetime(str(x)))
            risk_free = risk_free.groupby([risk_free.index.year, risk_free.index.month]).mean()

            self.risk_free = risk_free

    def _beta_pairs_formation(self, num_pairs, weight):
        """

        :param num_pairs:
        :param weight: (str) Weighting Scheme for portfolio returns [``equal`` by default, ``correlation``].        :return:
        """

        # Make empty dictionaries for beta value and pairs for each stock
        beta_dict = {}
        pairs_dict = {}

        # Get beta for each of the stocks in the train data
        for stock in self.monthly_return.columns:
            # Make a correlation matrix to get top 50 pairs for a given stock
            corr_matrix = self.monthly_return.corr()

            # Find pairs based on the Pearson correlation
            pairs = corr_matrix.loc[stock].sort_values(ascending=False)[1:num_pairs + 1].index.to_list()

            # Save the pairs in a dictionary
            pairs_dict[stock] = pairs

            # Get stock return to derive regression coefficient
            stock_return = self.monthly_return.loc[:, stock].values

            # Equal weighting
            if weight == 'equal':

                # Calculate equal weighted portfolio returns with pairs
                portfolio_return = self.monthly_return.loc[:, pairs].mean(axis=1).values - 1
                portfolio_return = portfolio_return.reshape((-1, 1))

            # Correlation based weighting
            else:

                # Find correlation values for pairs
                corr_values = corr_matrix.loc[stock].loc[pairs].values
                corr_values = corr_values/corr_values.sum()

                # Get pair return values
                pairs_return = self.monthly_return.loc[:, pairs].values
                portfolio_return = np.matmul(pairs_return, corr_values)

            # Use linear regression to get regression coefficient of two returns
            model = LinearRegression().fit(portfolio_return, stock_return)
            stock_coefficient = model.coef_[0]

            # Save the beta in a dictionary
            beta_dict[stock] = stock_coefficient

        self.beta_dict = beta_dict
        self.pairs_dict = pairs_dict

    def _calculate_return_diff(self, last_month):
        """

        :param last_month:
        :return:
        """

        # Make a dictionary for return difference
        return_diff_dict = {}

        for stock in last_month.index:
            # Risk free rate for the last month
            risk_free_test = self.risk_free[last_month.name]

            # Get pairs and beta from the dictionaries
            pairs = self.pairs_dict[stock]
            beta = self.beta_dict[stock]

            # Get portfolio and stock return for the last month
            portfolio_return = last_month[pairs].mean()
            stock_return = last_month[stock]

            # Calculate return difference value
            return_diff = beta * (portfolio_return - risk_free_test) - (stock_return - risk_free_test)

            # Save the return difference in the dictionary
            return_diff_dict[stock] = return_diff

        return return_diff_dict
