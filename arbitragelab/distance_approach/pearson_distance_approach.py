# Copyright 2021, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Implementation of the statistical arbitrage distance approach proposed by
Chen, H., Chen, S. J., and Li, F.
in "Empirical Investigation of an Equity Pairs Trading Strategy." (2012)
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1361293.
"""

import itertools
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from arbitragelab.util import segment


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
        self.risk_free = None  # Risk free rate dataset for calculating return differences in train period
        self.test_risk_free = None  # Risk free rate dataset for calculating return differences in test period
        self.beta_dict = None  # Regression coefficients for each stock in the formation period
        self.pairs_dict = None  # Top n pairs selected during the formation period
        self.last_month = None  # Returns from the last month of training data
        self.test_monthly_return = None  # Monthly return dataset in test period
        self.trading_signal = None  # Trading signal dataframe
        self.long_pct = 0.1  # Percentage of long stocks in the sorted return divergence
        self.short_pct = 0.1  # Percentage of short stocks in the sorted return divergence

        segment.track('PearsonStrategy')

    def form_portfolio(self, train_data, risk_free=0.0, num_pairs=50, weight='equal'):
        """
        Forms portfolio based on the input train data.

        For each stock i in year t+1(or the last month of year t), this method computes the Pearson correlation
        coefficients between the returns of stock i and returns of all other stocks in the given train data set.
        Usually, the formation period is set to 4 years but it may be changed upon the user’s needs.

        Then the method finds top n stocks with the highest correlations to stock i as its pairs in the formation
        period. For each month in year t+1, this method computes the pairs portfolio return as the equal-weighted
        average return of the n pairs stocks from the previous month.

        The hypothesis in this approach is that if a stock’s return deviates from its pairs portfolio returns more
        than usual, this divergence is expected to be reversed in the next month. And the returns of this stock are
        expected to be abnormally high/low in comparison to other stocks.

        In this method, for stock i, this method uses a new variable, return difference, which captures the return
        divergence between i’s stock return and its pairs-portfolio return.

        :param train_data: (pd.DataFrame) Daily price data with date in its index and stocks in its columns.
        :param risk_free: (pd.Series/float) Daily risk-free rate data as a series or a float number.
        :param num_pairs: (int) Number of top pairs to use for portfolio formation.
        :param weight: (str) Weighting Scheme for portfolio returns [``equal`` by default, ``correlation``].
        """

        # Preprocess data to get monthly return from daily price data
        self._data_preprocess(train_data, risk_free)

        # Calculating beta and getting pairs in the formation period
        self._beta_pairs_formation(num_pairs, weight)

        # Save the last month return for the signal generation step
        self.last_month = self.monthly_return.iloc[-1, :]

    def trade_portfolio(self, test_data=None, test_risk_free=0.0, long_pct=0.1, short_pct=0.1):
        """
        Trade portfolios by generating trading signals in the test data.

        In each month in the test period, all stocks are sorted in descending order based on their previous month’s
        return divergence from its pairs portfolio created in the formation period. Then a long-short portfolio is
        constructed with top p % of the stocks are “long stocks” and bottom q % of stocks are “short stocks”.

        If the test data is not given in this method, it automatically results in signals from the last month of the
        training data.

        :param test_data: (pd.DataFrame) Daily price data with date in its index and stocks in its columns.
        :param test_risk_free: (pd.Series/float) Daily risk-free rate data as a series or a float number.
        :param long_pct: (float) Percentage of long stocks in the sorted return divergence.
        :param short_pct: (float) Percentage of short stocks in the sorted return divergence.
        """

        # Set long and short percentage when trading
        self.long_pct, self.short_pct = long_pct, short_pct

        if test_data is None:
            self.trading_signal = self._find_trading_signals(self.last_month, single_period=True)

        else:
            # Preprocess test data
            self._data_preprocess(test_data, test_risk_free, phase='test')

            # Get trading signals for the test data
            self.trading_signal = self._find_trading_signals(self.test_monthly_return)

    def _find_trading_signals(self, monthly_return, single_period=False):
        """
        A helper function for finding trading signals.

        This method comprises two steps: First, by copying the monthly return dataframe in the test period,
        make sure that the trading signal is having the same index with monthly return dataframe. Next,
        by calculating the previous month’s return divergence for each of the stock, decide its position in the next
        month.

        :param monthly_return: (pd.DataFrame) A monthly return dataframe.
        :param single_period: (bool) Whether finding trading signal for a single period ahead.
        :return: (pd.DataFrame) Generated trading signals with multi index of year and month.
        """

        # Check if test data is not given and generating trading signal for only a single period
        if single_period:

            # Get long and short stock for the given data
            long_stocks, short_stocks = self._get_long_short(monthly_return, self.long_pct, self.short_pct)

            # Assign 1 for long stocks, -1 for short stocks, and 0 for others
            trading_signal_values = [1 if stock in long_stocks.keys()
                                     else -1 if stock in short_stocks.keys() else 0 for stock in monthly_return.index]

            # Create a series of trading signal
            trading_signal = pd.Series(trading_signal_values, index=monthly_return.index)

        else:
            # Calculate the trading signals in the test period
            for i in range(len(monthly_return)):

                if i == 0:
                    # Make a copy of monthly return to generate trading signal
                    trading_signal = monthly_return.copy()

                    # Decide the trading signal for the first month of the test period
                    trading_month = trading_signal.index[0]

                    # Using the last month from the training dataset as the previous month
                    prev_month_return = self.last_month

                else:

                    prev_month = trading_signal.index[i - 1]

                    trading_month = trading_signal.index[i]

                    prev_month_return = monthly_return.loc[prev_month, :]

                long_stocks, short_stocks = self._get_long_short(prev_month_return, self.long_pct, self.short_pct)

                for stock in trading_signal.columns:

                    if stock in short_stocks.keys():
                        trading_signal.loc[trading_month, stock] = -1

                    elif stock in long_stocks.keys():
                        trading_signal.loc[trading_month, stock] = 1

                    else:
                        trading_signal.loc[trading_month, stock] = 0

        return trading_signal

    def _get_long_short(self, prev_month_return, long_pct=0.1, short_pct=0.1):
        """
        Derive long and short stocks by calculating return divergence and form a portfolio based on the values.

        :param prev_month_return: (pd.Series) A series of monthly return to calculate the return divergence.
        :param long_pct: (float) Percentage of long stocks in the sorted return divergence.
        :param short_pct: (float) Percentage of short stocks in the sorted return divergence.
        :return: (dict) Long and short stocks with its corresponding return divergence values on its items.
        """

        return_diff_dict = self._calculate_return_diff(prev_month_return)

        # Sort the dictionary based on the return differences value
        return_diff_sorted = dict(sorted(return_diff_dict.items(), key=lambda item: item[1]))

        # Get the number of stocks to get two different portfolios
        num_stocks = len(return_diff_sorted)

        # Stocks with bottom 10% value of the return differences should be shorted
        short_stocks = dict(itertools.islice(return_diff_sorted.items(), 0, math.ceil(num_stocks * short_pct)))

        # Stocks with top 10% value of the return differences should be longed
        long_stocks = dict(itertools.islice(return_diff_sorted.items(), math.ceil(num_stocks * (1 - long_pct)),
                                            num_stocks))

        return long_stocks, short_stocks

    def get_trading_signal(self):
        """
        Outputs trading signal in monthly basis. 1 for a long position, -1 for a short position and 0 for closed
        position.

        :return: (pd.DataFrame) A dataframe with multi index of year and month for given test period.
        """

        return self.trading_signal

    def get_beta_dict(self):
        """
        Outputs beta, a regression coefficients for each stock, in the formation period.

        :return: (dict) A dictionary with stock in its key and beta in its value
        """

        return self.beta_dict

    def get_pairs_dict(self):
        """
        Outputs top n pairs selected during the formation period for each of the stock.

        :return: (dict) A dictionary with stock in its key and pairs in its value
        """

        return self.pairs_dict

    def _data_preprocess(self, price_data, risk_free, phase='train'):
        """
        Preprocess train data and risk free data.

        As monthly return data is used to calculate the beta and Pearson correlation in the formation period,
        it is needed to preprocess the train data and risk free data.

        :param price_data: (pd.DataFrame) Daily price data with date in its index and stocks in its columns.
        :param risk_free: (pd.Series/float) Daily risk free rate data as a series or a float number.
        :param phase: (str) Phase indicating training or testing, [``train`` by default, ``test``].
        """

        # Calculate normalized prices with mean and standard deviation
        data_copy = price_data.copy()

        # Change the daily price data into daily return by calculating the percent change
        daily_return = data_copy.pct_change()

        # Change the index of the train data into datetime to group by months later
        daily_return.index = pd.to_datetime(daily_return.index)

        # Add 1 to the values to calculate the compound return
        daily_return = daily_return + 1

        # Calculate monthly return for the given data
        monthly_return = daily_return.groupby([daily_return.index.year, daily_return.index.month]).prod()

        # Rescale the monthly return
        monthly_return = monthly_return - 1

        # Rename the multi index
        monthly_return.index.names = ["Year", "Month"]

        if isinstance(risk_free, float):

            # If risk free data is given as float, construct pd series
            risk_free = pd.Series(data=[risk_free for _ in range(len(monthly_return.index))], name='risk_free',
                                  index=monthly_return.index)

        else:

            # Get risk free rate
            risk_free = risk_free.rename('risk_free')
            risk_free.index = risk_free.index.map(lambda x: pd.to_datetime(str(x)))
            risk_free = risk_free.groupby([risk_free.index.year, risk_free.index.month]).mean()

        if phase == 'train':
            self.monthly_return = monthly_return
            self.risk_free = risk_free

        else:
            self.test_monthly_return = monthly_return
            self.risk_free = self.risk_free.append(risk_free)

    def _beta_pairs_formation(self, num_pairs, weight):
        """
        Calculate beta, a coefficient of measuring return divergence, in the formation period and form pairs.

        :param num_pairs: (int) Number of top pairs to use for portfolio formation.
        :param weight: (str) Weighting Scheme for portfolio returns [``equal`` by default, ``correlation``].
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
                corr_values = corr_matrix.loc[stock].loc[pairs].values - 1
                corr_values = corr_values / corr_values.sum()

                # Get pair return values
                pairs_return = self.monthly_return.loc[:, pairs].values
                portfolio_return = np.matmul(pairs_return, corr_values).reshape((-1, 1))

            # Use linear regression to get regression coefficient of two returns
            model = LinearRegression().fit(portfolio_return, stock_return)
            stock_coefficient = model.coef_[0]

            # Save the beta in a dictionary
            beta_dict[stock] = stock_coefficient

        self.beta_dict = beta_dict
        self.pairs_dict = pairs_dict

    def _calculate_return_diff(self, last_month):
        """
        Calculate return divergence of the stocks based on the beta calculated in the formation period.

        :param last_month: (pd.Series) A series of monthly return to calculate the return divergence.
        :return: (dict) A dictionary with stocks with its key and return divergence in its values.
        """

        # Make a dictionary for return difference
        return_diff_dict = {}

        for stock in last_month.index:
            # Risk free rate for the last month
            risk_free = self.risk_free[last_month.name]

            # Get pairs and beta from the dictionaries
            pairs = self.pairs_dict[stock]
            beta = self.beta_dict[stock]

            # Get portfolio and stock return for the last month
            portfolio_return = last_month[pairs].mean()
            stock_return = last_month[stock]

            # Calculate return difference value
            return_diff = beta * (portfolio_return - risk_free) - (stock_return - risk_free)

            # Save the return difference in the dictionary
            return_diff_dict[stock] = return_diff

        return return_diff_dict
