"""
Module for implementing partner selection approaches for vine copulas.
"""

# pylint: disable = invalid-name, broad-exception-raised
import functools
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from arbitragelab.copula_approach.vine_copula_partner_selection_utils import extremal_measure, get_quantiles_data ,\
    get_co_variance_matrix, get_sum_correlations_vectorized, diagonal_measure_vectorized, multivariate_rho_vectorized


class PartnerSelection:
    """
    Implementation of the Partner Selection procedures proposed in Section 3.1.1 in the following paper.

    3 partner stocks are selected for a target stock based on four different approaches namely, Traditional approach,
    Extended approach, Geometric approach and Extremal approach.

    In this module, target stock implies the ticker for which a unique combination of stocks is returned.
    The stocks present in this unique combination are called partner stocks.

    `Stübinger, J., Mangold, B. and Krauss, C., 2018. Statistical arbitrage with vine copulas. Quantitative Finance, 18(11), pp.1831-1849.
    <https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf>`__
    """

    def __init__(self, prices: pd.DataFrame, n: int = 50):
        """
        Inputs the price series required for further calculations.

        It also includes preprocessing steps described in the paper, before starting the Partner Selection procedures.
        These steps include, finding the returns and ranked returns of the stocks, and calculating the top n
        correlated stocks for each stock in the universe.

        :param prices: (pd.DataFrame) Contains price series of all stocks in the universe.
        :param n: (int) For each target stock, the total number of stocks taken into consideration for partner stocks
            in the final combination, from 500 stocks in the universe.
        """

        if len(prices) == 0:
            raise Exception("Input does not contain any data.")

        if not isinstance(prices, pd.DataFrame):
            raise Exception("Partner Selection Class requires a pandas DataFrame as input.")

        self.universe = prices  # Contains daily prices for all stocks in universe
        self.num_stocks_per_target = n # Total number of stocks taken into consideration as partner stocks for each target
        self.returns, self.ranked_returns = self._get_returns()  # Daily returns and corresponding ranked returns

        # Correlation matrix containing all stocks in universe
        self.correlation_matrix = self._correlation()
        # For each stock in universe, tickers of top n most correlated stocks are stored
        self.top_n_correlations = self._top_n_tickers()
        # Quadruple combinations for all stocks in universe
        self.all_quadruples = self._generate_all_combinations()

    def _correlation(self) -> pd.DataFrame:
        """
        Calculates correlation between all stocks in the universe.

        :return: (pd.DataFrame) Correlation Matrix.
        """

        return self.ranked_returns.corr(method='pearson')  # Pearson or spearman, we get same results as input is ranked

    def _get_returns(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Calculating daily returns and ranked daily returns of the stocks.

        :return (tuple):
            returns_df : (pd.DataFrame) Dataframe consists of daily returns.
            returns_df_ranked : (pd.DataFrame) Dataframe consists of ranked daily returns between [0,1].
        """

        returns_df = self.universe.pct_change()
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().dropna()

        # Calculating rank of daily returns for each stock. 'first' method is used to assign ranks in order they appear
        returns_df_ranked = returns_df.rank(axis=0, method='first', pct=True)
        return returns_df, returns_df_ranked

    def _tickers_list(self, col: pd.Series) -> list:
        """
        Returns list of tickers ordered according to correlations with target.

        :param col: (pd.Series) Correlation data for a stock.
        :return: (list) List of top n tickers.
        """

        # Sort the column data in descending order and return the index of top n rows
        return col.sort_values(ascending=False)[1: self.num_stocks_per_target + 1].index.to_list()

    def _top_n_tickers(self) -> pd.DataFrame:
        """
        Calculates the top n correlated stocks for each target stock.

        :return: (pd.DataFrame) Dataframe consisting of n columns for each stock in the universe.
        """

        # Returns DataFrame with all stocks as indices and their respective top n correlated stocks as columns
        return self.correlation_matrix.apply(self._tickers_list, axis=0).T

    def _generate_all_combinations(self, d: int = 4) -> pd.DataFrame:
        """
        Method generates unique combinations for all target stocks in universe.

        :param d: (int) Number of partner stocks.
        :return: (pd.DataFrame) Consists of all combinations for every target stock.
        """

        generate_all_combinations_helper_wrapper = functools.partial(self._generate_all_combinations_helper, d=d)

        return self.top_n_correlations.apply(generate_all_combinations_helper_wrapper, axis=1)

    @staticmethod
    def _generate_all_combinations_helper(row: pd.Series, d: int) -> list:
        """
        Helper function which generates unique combinations for each target stock.

        :param row: (pd.Series) List of n partner stocks.
        :param d: (int) Number of partner stocks.
        :return: (list) Combinations.
        """

        target = row.name
        combinations = []
        for comb in itertools.combinations(row, d - 1):
            combinations.append([target] + list(comb))

        return combinations

    @staticmethod
    def _prepare_combinations_of_partners(stock_selection: list) -> np.array:
        """
        Helper function to calculate all combinations for a target stock and it's potential partners.
        Stocks are treated as integers for vectorization purposes.

        :param stock_selection: (list) The target stock has to be the first element of the array.
        :return: (np.array) The possible combinations for the quadruples. Shape (19600,4).
        """

        # We will convert the stock names into integers and then get a list of all combinations with a length of 3
        num_of_stocks = len(stock_selection)
        # We turn our partner stocks into numerical indices so we can use them directly for indexing
        partner_stocks_idx = np.arange(1, num_of_stocks)  # Basically exclude the target stock
        partner_stocks_idx_combs = itertools.combinations(partner_stocks_idx, 3)

        return np.array(list((0,) + comb for comb in partner_stocks_idx_combs))

    # Method 1
    def traditional(self, n_targets: int = 5) -> list:
        """
        This method implements the first procedure described in Section 3.1.1.

        For all possible quadruples of a given stock, we calculate the sum of all pairwise correlations.
        For every target stock the quadruple with the highest sum is returned.

        :param n_targets: (int) Number of target stocks to select.
        :return: (list) List of all selected quadruples.
        """

        output_matrix = []  # Stores the final set of quadruples.
        # Iterating on the top n indices for each target stock.
        for target in self.top_n_correlations.index[:n_targets]:

            stock_selection = [target] + self.top_n_correlations.loc[target].tolist()  # List of n + 1 stocks including target
            data_subset = self.correlation_matrix.loc[stock_selection, stock_selection]
            all_possible_combinations = self._prepare_combinations_of_partners(stock_selection)  # Shape: (19600, 4)

            final_quadruple = get_sum_correlations_vectorized(data_subset, all_possible_combinations)[0]
            # Appending the final quadruple for each target to the output matrix
            output_matrix.append(final_quadruple)

        return output_matrix

    # Method 2
    def extended(self, n_targets: int = 5) -> list:
        """
        This method implements the second procedure described in Section 3.1.1.

        It involves calculating the multivariate version of Spearman's correlation for all possible quadruples of a given stock.
        The final measure taken into consideration is the mean of the three versions of Spearman's rho given in
        `Schmid and Schmidt (2007) <https://wisostat.uni-koeln.de/fileadmin/sites/statistik/pdf_publikationen/SchmidSchmidtSpearmansRho.pdf>`__.
        For every target stock the quadruple with the highest calculated measure is returned.

        :param n_targets: (int) Number of target stocks to select.
        :return: (list) List of all selected quadruples.
        """

        ecdf_df = self.returns.apply(get_quantiles_data, axis=0)  # Calculating ranks of returns using quantiles data

        output_matrix = []  # Stores the final set of quadruples
        # Iterating on the top n indices for each target stock
        for target in self.top_n_correlations.index[:n_targets]:
            stock_selection = [target] + self.top_n_correlations.loc[target].tolist()  # List of n + 1 stocks including target
            data_subset = ecdf_df[stock_selection]
            all_possible_combinations = self._prepare_combinations_of_partners(stock_selection) # Shape: (19600, 4)

            final_quadruple = multivariate_rho_vectorized(data_subset, all_possible_combinations)[0]
            # Appending the final quadruple for each target to the output matrix
            output_matrix.append(final_quadruple)

        return output_matrix

    # Method 3
    def geometric(self, n_targets: int = 5) -> list:
        """
        This method implements the third procedure described in Section 3.1.1.

        It involves calculating the four dimensional diagonal measure for all possible quadruples of a given stock.
        For example, visually, say we are in 2D, we have a Quantile-Quantile plot for the data,
        and this measure is just the sum of Euclidean distance for all data points to the y=x line (diagonal).
        For every target stock the quadruple with the lowest diagonal measure is returned.

        :param n_targets: (int) Number of target stocks to select.
        :return: (list) List of all selected quadruples.
        """

        output_matrix = []  # Stores the final set of quadruples
        # Iterating on the top n indices for each target stock
        for target in self.top_n_correlations.index[:n_targets]:
            stock_selection = [target] + self.top_n_correlations.loc[target].tolist()  # List of n + 1 stocks including target
            data_subset = self.ranked_returns[stock_selection]
            all_possible_combinations = self._prepare_combinations_of_partners(stock_selection)  # Shape: (19600, 4)

            final_quadruple = diagonal_measure_vectorized(data_subset, all_possible_combinations)[0]
            # Appending the final quadruple for each target to the output matrix
            output_matrix.append(final_quadruple)

        return output_matrix

    # Method 4
    def extremal(self, n_targets: int = 5, d: int = 4) -> list:
        """
        This method implements the fourth procedure described in Section 3.1.1.

        It involves calculating a non-parametric test statistic based on
        `Mangold (2015) <https://www.statistik.rw.fau.de/files/2016/03/IWQW-10-2015.pdf>`__ to measure the
        degree of deviation from independence. Main focus of this measure is the occurrence of joint extreme events.

        :param n_targets: (int) Number of target stocks to select.
        :param d: (int) Number of partner stocks(including target stock).
        :return output_matrix: (list) List of all selected combinations.
        """

        if d > self.num_stocks_per_target or d < 2:
            raise Exception("Please make sure number of partner stocks d is 2<=d<=n.")

        co_variance_matrix = get_co_variance_matrix(d)
        all_combinations = self._generate_all_combinations(d)
        output_matrix = []  # Stores the final set of combinations
        # Iterating on the top n indices for each target stock
        for target in self.top_n_correlations.index[:n_targets]:
            max_measure = -np.inf  # Variable used to extract the desired maximum value
            final_combination = None  # Stores the final desired combination

            # Iterating on all unique combinations generated for a target
            for combination in all_combinations[target]:
                measure = extremal_measure(self.ranked_returns[combination], co_variance_matrix)
                if measure > max_measure:
                    max_measure = measure
                    final_combination = combination
            # Appending the final combination for each target to the output matrix
            output_matrix.append(final_combination)

        return output_matrix

    def plot_selected_pairs(self, quadruples: list) -> list:
        """
        For the list of quadruples, this method plots the line plots of the cumulative returns of all stocks in quadruple.

        :param quadruples: (list) List of quadruples.
        :return: (list) List of Axes objects.
        """

        if len(quadruples) == 0:
            raise Exception("Input list is empty.")

        _, axs = plt.subplots(len(quadruples), figsize=(15, 3 * len(quadruples)))

        plt.subplots_adjust(hspace=0.6)

        if len(quadruples) == 1:
            quadruple = quadruples[0]
            data = self.universe.loc[:, quadruple].apply(lambda x: np.log(x).diff()).cumsum()
            sns.lineplot(ax=axs, data=data, legend=quadruple)
            axs.set_title(f'Final Quadruple of stocks with {quadruple[0]} as target')
            axs.set_ylabel('Cumulative Daily Returns')
        else:
            for i, quadruple in enumerate(quadruples):
                data = self.universe.loc[:, quadruple].apply(lambda x: np.log(x).diff()).cumsum()
                sns.lineplot(ax=axs[i], data=data, legend=quadruple)
                axs[i].set_title(f'Final Quadruple of stocks with {quadruple[0]} as target')
                axs[i].set_ylabel('Cumulative Daily Returns')

        return axs
