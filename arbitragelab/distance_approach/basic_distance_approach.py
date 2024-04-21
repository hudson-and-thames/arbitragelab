"""
Implementation of the statistical arbitrage distance approach proposed by
Gatev, E., Goetzmann, W. N., and Rouwenhorst, K. G. in
"Pairs trading:  Performance of a relative-value arbitrage rule." (2006)
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615.
"""
# pylint: disable=broad-exception-raised)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DistanceStrategy:
    """
    Class for creation of trading signals following the strategy by Gatev, E., Goetzmann, W. N., and Rouwenhorst, K. G.
    in "Pairs trading:  Performance of a relative-value arbitrage rule." (2006)
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615.
    """

    def __init__(self):
        """
        Initialize Distance strategy.
        """

        # Internal parameters
        self.min_normalize = None  # Minimum values for each price series used for normalization
        self.max_normalize = None  # Maximum values for each price series used for normalization
        self.pairs = None  # Created pairs after the form_pairs stage
        self.train_std = None  # Historical volatility for each chosen pair portfolio
        self.normalized_data = None  # Normalized test dataset
        self.portfolios = None  # Pair portfolios composed from test dataset
        self.train_portfolio = None  # Pair portfolios composed from train dataset
        self.trading_signals = None  # Final trading signals
        self.num_crossing = None  # Number of zero crossings from train dataset

    def form_pairs(self, train_data, method='standard', industry_dict=None, num_top=5, skip_top=0, selection_pool=50,
                   list_names=None):
        """
        Forms pairs based on input training data.

        This method includes procedures from the pairs formation step of the distance strategy.

        First, the input data is being normalized using max and min price values for each series:
        Normalized = (Price - Min(Price)) / (Max(Price) - Min(Price))

        Second, the normalized data is used to find a pair for each element - another series of
        prices that would have a minimum sum of square differences between normalized prices.
        Only unique pairs are picked in this step (pairs ('AA', 'BD') and ('BD', 'AA') are assumed
        to be one pair ('AA', 'BD')).
        During this step, if one decides to match pairs within the same industry group, with the
        industry dictionary given, the sum of square differences is calculated only for the pairs
        of prices within the same industry group.

        Third, based on the desired number of top pairs to chose and the pairs to skip, they are
        taken from the list of created pairs in the previous step. Pairs are sorted so that ones
        with a smaller sum of square distances are placed at the top of the list.

        Finally, the historical volatility for the portfolio of each chosen pair is calculated.
        Portfolio here is the difference of normalized prices of two elements in a pair.
        Historical volatility will later be used in the testing(trading) step of the
        distance strategy. The formula for calculating a portfolio price here:
        Portfolio_price = Normalized_price_A - Normalized_price_B

        Note: The input dataframe to this method should not contain missing values, as observations
        with missing values will be dropped (otherwise elements with fewer observations would
        have smaller distance to all other elements).

        :param train_data: (pd.DataFrame/np.array) Dataframe with training data used to create asset pairs.
        :param num_top: (int) Number of top pairs to use for portfolio formation.
        :param skip_top: (int) Number of first top pairs to skip. For example, use skip_top=10
            if you'd like to take num_top pairs starting from the 10th one.
        :param list_names: (list) List containing names of elements if Numpy array is used as input.
        :param method: (str) Methods to use for sorting pairs [``standard`` by default, ``variance``,
                             ``zero_crossing``].
        :param selection_pool: (int) Number of pairs to use before sorting them with the selection method.
        :param industry_dict: (dict) Dictionary matching ticker to industry group.
        """

        # If np.array given as an input
        if isinstance(train_data, np.ndarray):
            train_data = pd.DataFrame(train_data, columns=list_names)

        # Normalizing input data
        normalized, self.min_normalize, self.max_normalize = self.normalize_prices(train_data)

        # Dropping observations with missing values (for distance calculation)
        normalized = normalized.dropna(axis=0)

        # If industry dictionary is given, pairs are matched within the same industry group
        all_pairs = self.find_pair(normalized, industry_dict)

        # Choosing needed pairs to construct a portfolio
        self.pairs = self.sort_pairs(all_pairs, selection_pool)

        # Calculating historical volatility of pair portfolios (diffs of normalized prices)
        self.train_std = self.find_volatility(normalized, self.pairs)

        # Creating portfolios for pairs chosen in the pairs formation stage with train dataset
        self.train_portfolio = self.find_portfolios(normalized, self.pairs)

        # Calculating the number of zero crossings from the dataset
        self.num_crossing = self.count_number_crossing()

        # In case of a selection method other than standard or industry is used, sorting paris
        # based on the method
        self.selection_method(method, num_top, skip_top)

        # Storing only the necessary values for pairs selected in the above
        self.num_crossing = {pair: self.num_crossing[pair] for pair in self.pairs}
        self.train_std = {pair: self.train_std[pair] for pair in self.pairs}
        self.train_portfolio = self.train_portfolio[self.train_portfolio.columns
                                                        .intersection([str(pair) for pair in self.pairs])]

    def selection_method(self, method, num_top, skip_top):
        """
        Select pairs based on the method. This module helps sorting selected pairs for the given method
        in the formation period.

        :param method: (str) Methods to use for sorting pairs [``standard`` by default, ``variance``,
                             ``zero_crossing``].
        :param num_top: (int) Number of top pairs to use for portfolio formation.
        :param skip_top:(int) Number of first top pairs to skip. For example, use skip_top=10
            if you'd like to take num_top pairs starting from the 10th one.
        """

        if method not in ['standard', 'zero_crossing', 'variance']:
            # Raise an error if the given method is inappropriate.
            raise Exception("Please give an appropriate method for sorting pairs between ‘standard’, "
                            "‘zero_crossing’, or 'variance'")

        if method == 'standard':

            self.pairs = self.pairs[skip_top:(skip_top + num_top)]

        elif method == 'zero_crossing':

            # Sorting pairs from the dictionary by the number of zero crossings in a descending order
            sorted_pairs = sorted(self.num_crossing.items(), key=lambda x: x[1], reverse=True)

            # Picking top pairs
            pairs_selected = sorted_pairs[skip_top:(skip_top + num_top)]

            # Removing the number of crossings, so we have only tuples with elements
            pairs_selected = [x[0] for x in pairs_selected]

            self.pairs = pairs_selected

        else:

            # Sorting pairs from the dictionary by the size of variance in a descending order
            sorted_pairs = sorted(self.train_std.items(), key=lambda x: x[1], reverse=True)

            # Picking top pairs
            pairs_selected = sorted_pairs[skip_top:(skip_top + num_top)]

            # Removing the variance, so we have only tuples with elements
            pairs_selected = [x[0] for x in pairs_selected]

            self.pairs = pairs_selected

    def trade_pairs(self, test_data, divergence=2):
        """
        Generates trading signals for formed pairs based on new testing(trading) data.

        This method includes procedures from the trading step of the distance strategy.

        First, the input test data is being normalized with the min and max price values
        from the pairs formation step (so we're not using future data when creating signals).
        Normalized = (Test_Price - Min(Train_Price)) / (Max(Train_Price) - Min(Train_Price))

        Second, pair portfolios (differences of normalized price series) are constructed
        based on the chosen top pairs from the pairs formation step.

        Finally, for each pair portfolio trading signals are created. The logic of the trading
        strategy is the following: we open a position when the portfolio value (difference between
        prices) is bigger than divergence * historical_standard_deviation. And we close the
        position when the portfolio price changes sign (when normalized prices of elements cross).

        Positions are being opened in two ways. We open a long position on the first element
        from pair and a short position on the second element. The price of a portfolio is then:

        Portfolio_price = Normalized_price_A - Normalized_price_B

        If Portfolio_price > divergence * st_deviation, we open a short position on this portfolio.

        IF Portfolio_price < - divergence * st_deviation, we open a long position on this portfolio.

        Both these positions will be closed once Portfolio_price reaches zero.

        :param test_data: (pd.DataFrame/np.array) Dataframe with testing data used to create trading signals.
            This dataframe should contain the same columns as the dataframe used for pairs formation.
        :param divergence: (float) Number of standard deviations used to open a position in a strategy.
            In the original example, 2 standard deviations were used.
        """

        # If np.array given as an input
        if isinstance(test_data, np.ndarray):
            test_data = pd.DataFrame(test_data, columns=self.min_normalize.index)

        # If the pairs formation step wasn't performed
        if self.pairs is None:
            raise Exception("Pairs are not defined. Please perform the form_pairs() step first.")

        # Normalizing the testing data with min and max values obtained from the training data
        self.normalized_data, _, _ = self.normalize_prices(test_data, self.min_normalize, self.max_normalize)

        # Creating portfolios for pairs chosen in the pairs formation stage
        self.portfolios = self.find_portfolios(self.normalized_data, self.pairs)

        # Creating trade signals for pair portfolios
        self.trading_signals = self.signals(self.portfolios, self.train_std, divergence)

    def get_signals(self):
        """
        Outputs generated trading signals for pair portfolios.

        :return: (pd.DataFrame) Dataframe with trading signals for each pair.
            Trading signal here is the target quantity of portfolios to hold.
        """

        return self.trading_signals

    def get_portfolios(self):
        """
        Outputs pair portfolios used to generate trading signals.

        :return: (pd.DataFrame) Dataframe with portfolios for each pair.
        """

        return self.portfolios

    def get_scaling_parameters(self):
        """
        Outputs minimum and maximum values used for normalizing each price series.

        Formula used for normalization:
        Normalized = (Price - Min(Price)) / (Max(Price) - Min(Price))

        :return: (pd.DataFrame) Dataframe with columns 'min_value' and 'max_value' for each element.
        """

        scale = pd.DataFrame()

        scale['min_value'] = self.min_normalize
        scale['max_value'] = self.max_normalize

        return scale

    def get_pairs(self):
        """
        Outputs pairs that were created in the pairs formation step and sorted by the method.

        :return: (list) List containing tuples of two strings, for names of elements in a pair.
        """

        return self.pairs

    def get_num_crossing(self):
        """
        Outputs pairs that were created in the pairs formation step with its number of zero crossing.

        :return: (dict) Dictionary with keys as pairs and values as the number of zero
            crossings for pairs.
        """

        return self.num_crossing

    def count_number_crossing(self):
        """
        Calculate the number of zero crossings for the portfolio dataframe generated with train dataset.

        As the number of zero crossings in the formation period does have some usefulness in predicting
        future convergence, this method calculates the number of times the normalized spread crosses the
        value zero which measures the frequency of divergence and convergence between two securities.

        :return: (dict) Dictionary with keys as pairs and values as the number of zero
            crossings for pairs.
        """

        # Creating a dictionary for number of zero crossings
        num_zeros_dict = {}

        # Iterating through pairs
        for pair in self.train_portfolio:
            # Getting names of individual elements from dataframe column names
            pair_val = pair.strip('\')(\'').split('\', \'')
            pair_val = tuple(pair_val)

            # Check if portfolio price crossed zero
            portfolio = self.train_portfolio[pair].to_frame()
            pair_mult = portfolio * portfolio.shift(1)

            # Get the number of zero crossings for the portfolio
            num_zero_crossings = len(portfolio[pair_mult.iloc[:, 0] <= 0].index)

            # Adding the pair's number of zero crossings to the dictionary
            num_zeros_dict[pair_val] = num_zero_crossings

        return num_zeros_dict

    def plot_portfolio(self, num_pair):
        """
        Plots a pair portfolio (difference between element prices) and trading signals
        generated for it.

        :param num_pair: (int) Number of the pair from the list to use for plotting.
        :return: (plt.Figure) Figure with portfolio plot and trading signals plot.
        """

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 7))
        fig.suptitle('Distance Strategy results for portfolio' + self.trading_signals.columns[num_pair])

        axs[0].plot(self.portfolios[self.trading_signals.columns[num_pair]])
        axs[0].title.set_text('Portfolio value (the difference between element prices)')

        axs[1].plot(self.trading_signals[self.trading_signals.columns[num_pair]], '#b11a21')
        axs[1].title.set_text('Number of portfolio units to hold')

        return fig

    def plot_pair(self, num_pair):
        """
        Plots prices for a pair of elements and trading signals generated for their portfolio.

        :param num_pair: (int) Number of the pair from the list to use for plotting.
        :return: (plt.Figure) Figure with prices for pairs plot and trading signals plot.
        """

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 7))
        fig.suptitle('Distance Strategy results for pair' + self.trading_signals.columns[num_pair])

        pair_val = self.trading_signals.columns[num_pair].strip('\')(\'').split('\', \'')
        pair_val = tuple(pair_val)

        axs[0].plot(self.normalized_data[pair_val[0]], label="Long asset in a portfolio - " + pair_val[0])
        axs[0].plot(self.normalized_data[pair_val[1]], label="Short asset in a portfolio - " + pair_val[1])
        axs[0].legend()
        axs[0].title.set_text('Price of elements in a portfolio.')

        axs[1].plot(self.trading_signals[self.trading_signals.columns[num_pair]], '#b11a21')
        axs[1].title.set_text('Number of portfolio units to hold')

        return fig

    @staticmethod
    def normalize_prices(data, min_values=None, max_values=None):
        """
        Normalizes given dataframe of prices.

        Formula used:
        Normalized = (Price - Min(Price)) / (Max(Price) - Min(Price))

        :param data: (pd.DataFrame) Dataframe with prices.
        :param min_values: (pd.Series) Series with min values to use for price scaling.
            If None, will be calculated from the given dataset.
        :param max_values: (pd.Series) Series with max values to use for price scaling.
            If None, will be calculated from the given dataset.
        :return: (pd.DataFrame, pd.Series, pd.Series) Dataframe with normalized prices
            and series with minimum and maximum values used to normalize price series.
        """

        # If normalization parameters are not given, calculate
        if (max_values is None) or (min_values is None):
            max_values = data.max()
            min_values = data.min()

        # Normalizing the dataset
        data_copy = data.copy()
        normalized = (data_copy - min_values) / (max_values - min_values)

        return normalized, min_values, max_values

    @staticmethod
    def find_pair(data, industry_dict=None):
        """
        Finds the pairs with smallest distances in a given dataframe.

        Closeness measure here is the sum of squared differences in prices.
        Duplicate pairs are dropped, and elements in pairs are sorted in alphabetical
        order. So pairs ('AA', 'BC') and ('BC', 'AA') are treated as one pair ('AA', 'BC').

        :param data: (pd.DataFrame) Dataframe with normalized price series.
        :param industry_dict: (dictionary) Dictionary matching ticker to industry group.
        :return: (dict) Dictionary with keys as closest pairs and values as their distances.
        """

        # Creating a dictionary
        pairs = {}

        # Iterating through each element in dataframe
        for ticker in data:

            # Removing the chosen element from the dataframe
            data_excluded = data.drop([ticker], axis=1)

            # Removing tickers in different industry group if the industry dictionary is given
            if industry_dict is not None:
                # Getting the industry group for the ticker
                industry_group = industry_dict[ticker]
                # Getting the tickers within the same industry group
                tickers_same_industry = [ticker for ticker, industry in industry_dict.items()
                                         if industry == industry_group]
                # Removing other tickers in different industry group
                data_excluded = data_excluded.loc[:, data_excluded.columns.isin(tickers_same_industry)]

            # Calculating differences between prices
            data_diff = data_excluded.sub(data[ticker], axis=0)

            # Calculating the sum of square differences
            sum_sq_diff = (data_diff ** 2).sum()

            # Iterating through second elements
            for second_element in sum_sq_diff.index:
                # Adding all new pairs to the dictionary
                pairs[tuple(sorted((ticker, second_element)))] = sum_sq_diff[second_element]

        return pairs

    @staticmethod
    def sort_pairs(pairs, num_top=5, skip_top=0):
        """
        Sorts pairs of elements and returns top_num of closest ones.

        The skip_top parameter can be used to skip a number of first top portfolios.
        For example, if we'd like to pick pairs number 10-15 from the top list, we set
        num_top = 5, skip_top = 10.

        :param pairs: (dict) Dictionary with keys as pairs and values as distances
            between elements in a pair.
        :param num_top: (int) Number of closest pairs to take.
        :param skip_top: (int) Number of top closest pairs to skip.
        :return: (list) List containing sorted pairs as tuples of strings, representing
            elements in a pair.
        """

        # Sorting pairs from the dictionary by distances in an ascending order
        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=False)

        # Picking top pairs
        top_pairs = sorted_pairs[skip_top:(skip_top + num_top)]

        # Removing distance values, so we have only tuples with elements
        top_pairs = [x[0] for x in top_pairs]

        return top_pairs

    @staticmethod
    def find_volatility(data, pairs):
        """
        Calculates historical volatility of portfolios(differences of prices)
        for set of pairs.


        :param data: (pd.DataFrame) Dataframe with price series to use for calculation.
        :param pairs: (list) List of tuples with two elements to use for calculation.
        :return: (dict) Dictionary with keys as pairs of elements and values as their
            historical volatility.
        """

        # Creating a dictionary
        volatility_dict = {}

        # Iterating through pairs of elements
        for pair in pairs:
            # Getting two price series for elements in a pair
            par = data[list(pair)]

            # Differences between picked price series
            par_diff = par.iloc[:, 0] - par.iloc[:, 1]

            # Calculating standard deviation for difference series
            st_div = par_diff.std()

            # Adding pair and volatility to dictionary
            volatility_dict[pair] = st_div

        return volatility_dict

    @staticmethod
    def find_portfolios(data, pairs):
        """
        Calculates portfolios (difference of price series) based on given prices dataframe
        and set of pairs to use.

        When creating a portfolio, we long one share of the first element and short one share
        of the second element.

        :param data: (pd.DataFrame) Dataframe with price series for elements.
        :param pairs: (list) List of tuples with two str elements to use for calculation.
        :return: (pd.DataFrame) Dataframe with pairs as columns and their portfolio
            values as rows.
        """

        # Creating a dataframe
        portfolios = pd.DataFrame()

        # Iterating through pairs
        for pair in pairs:
            # Difference between price series - a portfolio
            par_diff = data.loc[:, pair[0]] - data.loc[:, pair[1]]
            portfolios[str(pair)] = par_diff

        return portfolios

    @staticmethod
    def signals(portfolios, variation, divergence):
        """
        Generates trading signals based on the idea described in the original paper.

        A position is being opened when the difference between prices (portfolio price)
        diverges by more than divergence (two in the original paper) historical standard
        deviations. This position is being closed once pair prices are crossing (portfolio
        price reaches zero).

        Positions are being opened in both buy and sell directions.

        :param portfolios: (pd.DataFrame) Dataframe with portfolio price series for pairs.
        :param variation: (dict) Dictionary with keys as pairs and values as the
            historical standard deviations of their pair portfolio.
        :param divergence: (float) Number of standard deviations used to open a position.
        :return: (pd.DataFrame) Dataframe with target quantity to hold for each portfolio.
        """

        # Creating a signals dataframe
        signals = pd.DataFrame()

        # Iterating through pairs
        for pair in portfolios:
            # Getting names of individual elements from dataframe column names
            pair_val = pair.strip('\')(\'').split('\', \'')
            pair_val = tuple(pair_val)

            # Historical standard deviation for a pair
            st_dev = variation[pair_val]

            # Check if portfolio price crossed zero
            portfolio = portfolios[pair].to_frame()
            pair_mult = portfolio * portfolio.shift(1)

            # Entering a short position when portfolio is higher than divergence * st_dev
            short_entry_index = portfolio[portfolio.iloc[:, 0] > divergence * st_dev].index
            short_exit_index = portfolio[pair_mult.iloc[:, 0] <= 0].index

            # Entering a long position in the opposite situation
            long_entry_index = portfolio[portfolio.iloc[:, 0] < -divergence * st_dev].index
            long_exit_index = portfolio[pair_mult.iloc[:, 0] <= 0].index

            # Transforming long and short trading signals into one signal - target quantity
            portfolio['long_units'] = np.nan
            portfolio['short_units'] = np.nan
            portfolio.iloc[0, portfolio.columns.get_loc('long_units')] = 0
            portfolio.iloc[0, portfolio.columns.get_loc('short_units')] = 0

            portfolio.loc[long_entry_index, 'long_units'] = 1
            portfolio.loc[long_exit_index, 'long_units'] = 0
            portfolio.loc[short_entry_index, 'short_units'] = -1
            portfolio.loc[short_exit_index, 'short_units'] = 0

            portfolio.fillna(method='pad', inplace=True)
            portfolio['target_quantity'] = portfolio['long_units'] + portfolio['short_units']

            # Adding target quantity to signals dataframe
            signals[str(pair)] = portfolio['target_quantity']

        return signals
