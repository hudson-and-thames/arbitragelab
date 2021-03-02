import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from statsmodels.distributions.empirical_distribution import ECDF
from arbitragelab.copula_approach.vine_copula_partner_selection_utils import get_sum_correlations, multivariate_rho, diagonal_measure, extremal_measure, get_co_variance_matrix


class PartnerSelection:
    """
    Implementation of the Partner Selection procedures proposed in Section 3.1.1 in the following paper.

    3 partner stocks are selected for a target stock based on four different approaches namely, Traditional approach,
    Extended approach,Geometric approach and Extremal approach.
    https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
    """

    def __init__(self, prices: pd.DataFrame):
        """
        Inputs the price series required for further calculations.
        Also includes preprocessing steps described in the paper, before starting the Partner Selection procedures.
        These steps include, finding the returns and ranked returns of the stocks, and calculating the top 50
        correlated stocks for each stock in the universe.

        :param prices: (pd.DataFrame): Contains price series of all stocks in universe
        """

        if len(prices) == 0:
            raise Exception("Input does not contain any data")

        if not isinstance(prices, pd.DataFrame):
            raise Exception("Partner Selection Class requires a pandas DataFrame as input")

        self.universe = prices  # Contains daily prices for all stocks in universe.
        self.returns, self.ranked_returns = self._get_returns()  # Daily returns and corresponding ranked returns.

        # Correlation matrix containing all stocks in universe
        self.correlation_matrix = self._correlation()
        # For each stock in universe, tickers of top 50 most correlated stocks are stored
        self.top_50_correlations = self._top_50_tickers()
        # Quadruple combinations for all stocks in universe
        self.all_quadruples = self._generate_all_quadruples()

    def _correlation(self) -> pd.DataFrame:
        """
        Calculates correlation between all stocks in universe.

        :return: (pd.DataFrame) : Correlation Matrix
        """

        return self.ranked_returns.corr(method='pearson')  # Pearson or spearman,we get same results as input is ranked

    def _get_returns(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Calculating daily returns and ranked daily returns of the stocks.

        :return (tuple):
            returns_df : (pd.DataFrame) : Dataframe consists of daily returns
            returns_df_ranked : (pd.DataFrame) : Dataframe consists of ranked daily returns between [0,1]
        """

        returns_df = self.universe.pct_change()
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().dropna()

        # Calculating rank of daily returns for each stock. 'first' method is used to assign ranks in order they appear
        returns_df_ranked = returns_df.rank(axis=0, method='first', pct=True)
        return returns_df, returns_df_ranked

    def _top_50_tickers(self) -> pd.DataFrame:
        """
        Calculates the top 50 correlated stocks for each target stock.

        :return: (pd.DataFrame) : Dataframe consisting of 50 columns for each stock in the universe
        """

        def tickers_list(col):
            """
            Returns list of tickers ordered according to correlations with target.
            """
            # Sort the column data in descending order and return the index of top 50 rows.
            return col.sort_values(ascending=False)[1:51].index.to_list()

        # Returns DataFrame with all stocks as indices and their respective top 50 correlated stocks as columns.
        return self.correlation_matrix.apply(tickers_list, axis=0).T

    def _generate_all_quadruples(self) -> pd.DataFrame:
        """
         Method generates unique quadruples for all target stocks in universe.

         :return: (pd.DataFrame) : consists of all quadruples for every target stock
         """

        return self.top_50_correlations.apply(self._generate_all_quadruples_helper, axis=1)

    @staticmethod
    def _generate_all_quadruples_helper(row: pd.Series) -> list:
        """
         Helper function which generates unique quadruples for each target stock.

         :param row: (pd.Series) : list of 50 partner stocks
         :return: (list) : quadruples
         """

        target = row.name
        quadruples = []
        for triple in itertools.combinations(row, 3):
            quadruples.append([target] + list(triple))
        return quadruples

    # Method 1
    def traditional(self, n_targets=5) -> list:
        """
        This method implements the first procedure described in Section 3.1.1.
        For all possible quadruples of a given stock, we calculate the sum of all pairwise correlations.
        For every target stock the quadruple with the highest sum is returned.

        :param n_targets: (int) : number of target stocks to select
        :return output_matrix: list: List of all selected quadruples
        """

        output_matrix = []  # Stores the final set of quadruples.
        # Iterating on the top 50 indices for each target stock.
        for target in self.top_50_correlations.index[:n_targets]:
            max_sum_correlations = 0  # Variable used to extract the desired maximum value
            final_quadruple = None  # Stores the final desired quadruple

            # Iterating on all unique quadruples generated for a target
            for quadruple in self.all_quadruples[target]:
                sum_correlations = get_sum_correlations(self.correlation_matrix, quadruple)
                if sum_correlations > max_sum_correlations:
                    max_sum_correlations = sum_correlations
                    final_quadruple = quadruple

            print(final_quadruple)
            # Appending the final quadruple for each target to the output matrix
            output_matrix.append(final_quadruple)

        return output_matrix

    # Method 2
    def extended(self, n_targets=5) -> list:
        """
        This method implements the second procedure described in Section 3.1.1.
        It involves calculating the multivariate version of Spearman's correlation
        for all possible quadruples of a given stock.
        For every target stock the quadruple with the highest correlation is returned.

        :param n_targets: (int) : number of target stocks to select
        :return output_matrix: list: List of all selected quadruples
        """

        u = self.returns.copy()  # Generating ranked returns from quantiles using statsmodels ECDF
        for column in self.returns.columns:
            ecdf = ECDF(self.returns.loc[:, column])
            u[column] = ecdf(self.returns.loc[:, column])

        output_matrix = []  # Stores the final set of quadruples.
        # Iterating on the top 50 indices for each target stock.
        for target in self.top_50_correlations.index[:n_targets]:
            max_correlation = -np.inf  # Variable used to extract the desired maximum value
            final_quadruple = None  # Stores the final desired quadruple

            # Iterating on all unique quadruples generated for a target
            for quadruple in self.all_quadruples[target]:
                correlation = multivariate_rho(u[quadruple])
                if correlation > max_correlation:
                    max_correlation = correlation
                    final_quadruple = quadruple

            print(final_quadruple)
            # Appending the final quadruple for each target to the output matrix
            output_matrix.append(final_quadruple)

        return output_matrix

    # Method 3
    def geometric(self, n_targets=5) -> list:
        """
        This method implements the third procedure described in Section 3.1.1.
        It involves calculating the four dimensional diagonal measure for all possible quadruples of a given stock.
        For every target stock the quadruple with the lowest diagonal measure is returned.

        :param n_targets: (int) : number of target stocks to select
        :return output_matrix: list: List of all selected quadruples
        """

        output_matrix = []  # Stores the final set of quadruples.
        # Iterating on the top 50 indices for each target stock.
        for target in self.top_50_correlations.index[:n_targets]:
            min_measure = np.inf  # Variable used to extract the desired minimum value
            final_quadruple = None  # Stores the final desired quadruple

            # Iterating on all unique quadruples generated for a target
            for quadruple in self.all_quadruples[target]:
                measure = diagonal_measure(self.ranked_returns[quadruple])
                if measure < min_measure:
                    min_measure = measure
                    final_quadruple = quadruple
            print(final_quadruple)
            # Appending the final quadruple for each target to the output matrix
            output_matrix.append(final_quadruple)

        return output_matrix

    # Method 4
    def extremal(self, n_targets=5) -> list:
        """
        This method implements the fourth procedure described in Section 3.1.1.
        It involves calculating a non-parametric test statistic based on Mangold (2015) to measure the
        degree of deviation from independence. Main focus of this measure is the occurrence of joint extreme events.

        :param n_targets: (int) : number of target stocks to select
        :return output_matrix: list: List of all selected quadruples
        """

        co_variance_matrix = get_co_variance_matrix()
        output_matrix = []  # Stores the final set of quadruples.
        # Iterating on the top 50 indices for each target stock.
        for target in self.top_50_correlations.index[:n_targets]:
            max_measure = -np.inf  # Variable used to extract the desired maximum value
            final_quadruple = None  # Stores the final desired quadruple

            # Iterating on all unique quadruples generated for a target
            for quadruple in self.all_quadruples[target]:
                measure = extremal_measure(self.ranked_returns[quadruple], co_variance_matrix)
                if measure > max_measure:
                    max_measure = measure
                    final_quadruple = quadruple
            print(final_quadruple)
            # Appending the final quadruple for each target to the output matrix
            output_matrix.append(final_quadruple)

        return output_matrix

    def plot_selected_pairs(self, quadruples: list):
        """
        Plots the final selection of quadruples.
        :param quadruples: List of quadruples
        """

        if quadruples is None:
            raise Exception("Input list is empty")

        fig, axs = plt.subplots(len(quadruples),
                                figsize=(15, 3 * len(quadruples)))

        plt.subplots_adjust(hspace=0.6)
        for i, quadruple in enumerate(quadruples):
            data = self.universe.loc[:, quadruple].apply(lambda x: np.log(x).diff()).cumsum()
            sns.lineplot(ax=axs[i], data=data, legend=quadruple)
            axs[i].set_title(f'Final Quadruple of stocks with {quadruple[0]} as target')
            axs[i].set_ylabel('Cumulative Daily Returns')
        plt.show()

    def plot_all_target_measures(self, target: str, procedure: str):
        """
        Plots a scatterplot showing measures calculated for all possible quadruples of a given target stock.
        :param target: (str) : target stock ticker
        :param procedure: (str) : name of procedure for calculating measure
        """

        if procedure not in ('traditional', 'extended', 'geometric', 'extremal'):
            raise Exception("Please enter a valid procedure name, i.e ('traditional', 'extended', 'geometric', "
                            "'extremal') ")

        measures_list = []
        quadruples = self.all_quadruples[target]  # List of all quadruples
        final_quadruple = None
        final_measure = -np.inf

        co_variance_matrix = None
        u = None

        # Preprocessing steps for some approaches
        if procedure == 'extremal':
            co_variance_matrix = get_co_variance_matrix()
        if procedure == 'extended':
            u = self.returns.copy()  # Generating ranked returns from quantiles using statsmodels ECDF
            for column in self.returns.columns:
                ecdf = ECDF(self.returns.loc[:, column])
                u[column] = ecdf(self.returns.loc[:, column])
        if procedure == 'geometric':
            final_measure = np.inf

        for quadruple in quadruples:
            # Separate functionality for geometric approach because here the minimum measure is required.
            if procedure == 'geometric':
                measure = diagonal_measure(self.ranked_returns[quadruple])
                measures_list.append(measure)
                if measure < final_measure:
                    final_measure = measure
                    final_quadruple = quadruple
                continue

            measure = 0
            if procedure == 'traditional':
                measure = get_sum_correlations(self.correlation_matrix, quadruple)
            elif procedure == 'extended':
                measure = multivariate_rho(u[quadruple])
            elif procedure == 'extremal':
                measure = extremal_measure(self.ranked_returns[quadruple], co_variance_matrix)

            measures_list.append(measure)  # Storing all calculated measures
            if measure > final_measure:
                final_measure = measure
                final_quadruple = quadruple

        print(final_quadruple)

        # Code for plotting the final list of calculated measures
        plt.figure(figsize=(20, 6))
        data = pd.DataFrame(measures_list, columns=['measure'])
        data['indices'] = range(len(measures_list))
        data['hue'] = [0] * len(data)
        if procedure == 'geometric':
            data.loc[data['measure'].idxmin(), 'hue'] = 1
        else:
            data.loc[data['measure'].idxmax(), 'hue'] = 1
        sns.scatterplot(x='indices', y='measure', data=data, alpha=0.5, hue='hue', size='hue',
                        sizes={0: 5, 1: 40}, legend=False)
        plt.title(f"Measures calculated from {procedure} approach for all quadruples of target {target}")
        plt.show()
