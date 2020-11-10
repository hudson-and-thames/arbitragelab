"""
This module implements the ML based Pairs Selection Framework described by by Simão Moraes
Sarmento and Nuno Horta in `"A Machine Learning based Pairs Trading Investment Strategy." <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.
"""

import itertools

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

from arbitragelab.ml_based_pairs_selection.stat_arb_utils import _outer_cointegration_loop, _outer_ou_loop
from arbitragelab.util.indexed_highlight import IndexedHighlight


class PairsSelector:
    """
    Implementation of the Proposed Pairs Selection Framework in the following paper. The
    method consists of three parts; dimensionality reduction, clustering of features and
    finally the selection of pairs with the use of a set of heuristics.
    """

    def __init__(self, universe: pd.DataFrame):
        """
        Constructor

        Sets up the price series needed for the next step

        :param universe: (pd.DataFrame): Asset prices universe
        """

        self.prices_df = universe
        self.feature_vector = None
        self.cluster_pairs_combinations = []
        self.spreads_df = None

        self.coint_pass_pairs = pd.Series({}, dtype=object)
        self.hurst_pass_pairs = pd.Series({}, dtype=object)
        self.hl_pass_pairs = pd.Series({}, dtype=object)

        self.final_pairs = []
        self.clust_labels_ = []

    def dimensionality_reduction_by_components(self, num_features: int = 10):
        """
        Processes and scales the prices universe supplied in the constructor, into returns. Then reduces the resulting
        data using pca down to the amount of dimensions needed to be used as a feature vector in the clustering step.
        Optimal ranges for the dimensions required in the feature vector should be <15.

        :param num_features: (int): Used to select pca n_components to be used in the feature vector
        """

        if self.prices_df is None:
            raise Exception(
                "Please input a valid price series before running this method")

        # cleaning
        returns_df = (self.prices_df - self.prices_df.shift(1))
        returns_df = returns_df / self.prices_df.shift(1)
        returns_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        returns_df.ffill(inplace=True)

        # scaling
        scaler = StandardScaler()
        scaled_returns_df = pd.DataFrame(scaler.fit_transform(returns_df))
        scaled_returns_df.columns = returns_df.columns
        scaled_returns_df.set_index(returns_df.index)
        scaled_returns_df.dropna(inplace=True)

        # reducing
        pca = PCA(n_components=num_features)
        pca.fit(scaled_returns_df)
        self.feature_vector = pd.DataFrame(pca.components_)
        self.feature_vector.columns = returns_df.columns
        self.feature_vector = self.feature_vector.T

    def plot_pca_matrix(self): # pragma: no cover
        """
        Plots the feature vector on a scatter matrix.
        """

        feature_vector = self.feature_vector.copy()

        cols = ["Component " + str(i) for i in feature_vector.columns]
        feature_vector.columns = cols

        axes = pd.plotting.scatter_matrix(
            feature_vector, alpha=0.2, figsize=(15, 15))

        new_labels = [round(float(i.get_text()), 2)
                      for i in axes[0, 0].get_yticklabels()]
        axes[0, 0].set_yticklabels(new_labels)

    def cluster_using_optics(self, args: dict):
        """
        Second step of the framework; Doing Unsupervised Learning on the feature vector supplied from the first step.
        The clustering method used is OPTICS, chosen mainly for it being basically parameterless.

        :param args: (dict): Arguments to be passed to the clustering algorithm
        """

        if self.feature_vector is None:
            raise Exception("The needed feature vector has not been computed yet",
                            "Please run dimensionality_reduction() before this method")

        clust = OPTICS(**args)
        clust.fit(self.feature_vector)
        self.clust_labels_ = clust.labels_

    def cluster_using_dbscan(self, args: dict):
        """
        Second step of the framework; Doing Unsupervised Learning on the feature vector supplied from the first step.
        The second clustering method used is DBSCAN, for when the user needs a more hands on approach to doing the
        clustering step, given the parameter sensitivity of this method.

        :param args: (dict): Arguments to be passed to the clustering algorithm
        """

        if self.feature_vector is None:
            raise Exception("The needed feature vector has not been computed yet",
                            "Please run dimensionality_reduction() before this method")

        clust = DBSCAN(**args)
        clust.fit(self.feature_vector)
        self.clust_labels_ = clust.labels_

    def plot_clustering_info(self, n_dimensions: int = 2,
                             method: str = "", figsize=(10, 10)): # pragma: no cover
        """
        Plots the clusters found on a scatter plot.

        :param n_dimensions: (int): Selected dimension to be used in the T-SNE plot
        :param method: (str): String to be used as title in the plot
        :param figsize: (tuple): Tuple describing the size of the plot
        """

        if self.feature_vector is None:
            raise Exception("The needed feature vector has not been computed yet",
                            "Please run dimensionality_reduction() before this method")

        if self.clust_labels_ is None:
            raise Exception("The needed cluster labels have not been computed yet",
                            "Please run cluster() before this method")

        no_of_classes = len(np.unique(self.clust_labels_))

        fig = plt.figure(1, facecolor='white', figsize=figsize)

        tsne = TSNE(n_components=n_dimensions)
        tsne_fv = pd.DataFrame(tsne.fit_transform(
            self.feature_vector), index=self.feature_vector.index)

        if n_dimensions == 2:

            ax1 = fig.add_subplot(111)

            ax1.spines['left'].set_position('center')
            ax1.spines['left'].set_alpha(0.3)
            ax1.spines['bottom'].set_position('center')
            ax1.spines['bottom'].set_alpha(0.3)
            ax1.spines['right'].set_color('none')
            ax1.spines['top'].set_color('none')

            ax1.xaxis.set_ticks_position('bottom')
            ax1.yaxis.set_ticks_position('left')
            ax1.tick_params(which='major', labelsize=18)

            paths_collection = []

            for klass in range(0, no_of_classes):
                x_klass = tsne_fv[self.clust_labels_ == klass]

                paths = ax1.plot(x_klass.loc[:, 0], x_klass.loc[:, 1], label=list(x_klass.index), markersize=30, alpha=0.75,
                                 marker='.', linestyle='None')

                paths_collection.append(paths)

            IndexedHighlight(np.ravel(paths_collection),
                             formatter='{label}'.format)

            ax1.plot(tsne_fv.iloc[self.clust_labels_ == -1, 0],
                     tsne_fv.iloc[self.clust_labels_ == -1, 1], 'k+', alpha=0.1)

            ax1.set_title('Automatic Clustering\n' + method)

        elif n_dimensions == 3:

            ax1 = plt.subplot(111, projection='3d')
            paths_collection = []

            for klass in range(0, no_of_classes):
                x_klass = tsne_fv[self.clust_labels_ == klass]

                paths = ax1.plot(x_klass.loc[:, 0], x_klass.loc[:, 1], x_klass.loc[:, 2], alpha=0.7,
                                 marker='.', linestyle='None', label=list(x_klass.index), markersize=30)

                paths_collection.append(paths)

            IndexedHighlight(np.ravel(paths_collection),
                             formatter='{label}'.format)

            ax1.plot(tsne_fv.iloc[self.clust_labels_ == -1, 0],
                     tsne_fv.iloc[self.clust_labels_ == -1, 1],
                     tsne_fv.iloc[self.clust_labels_ == -1, 2], 'k+', alpha=0.1)
            ax1.set_title('Automatic Clustering\n' + method)

        else:
            raise Exception("Select a valid dimension!")

        plt.tight_layout()
        plt.show()

    def plot_knee_plot(self): # pragma: no cover
        """
        This method will plot the k-distance graph, ordered from the largest to the smallest value.
        The values where this plot shows an "elbow" should be a reference to the user of the optimal
        ε parameter to be used for the DBSCAN clustering method.
        """

        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(self.feature_vector)
        result = nbrs.kneighbors(self.feature_vector)
        result[0] = np.sort(result[0], axis=0)
        result[0] = result[0][:, 1]

        plt.ylabel('k-distance')
        plt.xlabel('# of data rows')
        plt.plot(result[0])

    def _generate_pairwise_combinations(self, labels: list) -> list:
        """
        This method will loop through all generated clusters (except -1) and generate
        pairwise combinations of the assets in each cluster.

        :param labels: (list) : List of unique labels
        :return pair_combinations: (list) : list of asset name pairs
        """

        if len(labels) == 0:
            raise Exception("No Labels have been found")

        pair_combinations = []

        for labl in labels:
            cluster_x = self.feature_vector[self.clust_labels_ == labl].index
            cluster_x = cluster_x.tolist()

            for combination in list(itertools.combinations(cluster_x, 2)):
                pair_combinations.append(combination)

        return pair_combinations

    @staticmethod
    def get_pairs_by_sector(sectoral_info: pd.DataFrame) -> list:
        """
        This method will loop through all the tickers tagged by sector and generate
        pairwise combinations of the assets for each sector.

        :param sectoral_info: (pd.DataFrame) : List of asset name pairs to be analyzed tagged with respective sector
        :return pair_combinations: (list) : list of asset name pairs
        """

        pair_combinations = []
        sectors = np.unique(sectoral_info['sector'])

        for sector in sectors:
            cluster_x = sectoral_info[sectoral_info['sector']
                                      == sector]['ticker'].values
            if len(cluster_x) > 1:
                for combination in list(itertools.combinations(cluster_x, 2)):
                    pair_combinations.append(combination)

        return pair_combinations

    def _hurst_criterion(self, pairs: pd.DataFrame,
                         hurst_exp_threshold: int = 0.5) -> tuple:
        """
        This method will go through all the pairs given, calculate the needed spread and run
        the hurst exponent test against each one.

        :param pairs: (pd.DataFrame) : DataFrame of asset name pairs to be analyzed
        :param hurst_exp_threshold: (int) : max hurst threshold value
        :return (tuple) :
            spreads_df: (pd.DataFrame) : Hedge ratio adjusted spreads DataFrame
            hurst_pass_pairs: (pd.DataFrame) : list of pairs that passed the hurst check and their respective hurst value
        """

        hurst_pass_pairs = []
        spreads_lst = []
        spreads_cols = []

        if len(pairs) != 0:
            for idx, frame in pairs.iterrows():
                asset_one = self.prices_df.loc[:, idx[1]].values
                asset_two = self.prices_df.loc[:, idx[0]].values

                spread_ts = (asset_one - asset_two * frame['hedge_ratio'])
                hurst_exp = self.hurst(spread_ts)

                if hurst_exp < hurst_exp_threshold:
                    hurst_pass_pairs.append((idx, hurst_exp))
                    spreads_lst.append(spread_ts)
                    spreads_cols.append(str(idx))
        else:
            raise Exception("No pairs have been found")

        spreads_df = pd.DataFrame(data=spreads_lst).T
        spreads_df.columns = spreads_cols
        spreads_df.index = pd.to_datetime(self.prices_df.index)

        hurst_pass_pairs_df = pd.DataFrame(data=hurst_pass_pairs, columns=['pairs', 'hurst_exponent']).set_index('pairs')
        hurst_pass_pairs_df.index.name = None

        return spreads_df, hurst_pass_pairs_df

    @staticmethod
    def _final_criterions(spreads_df: pd.DataFrame, pairs: list,
                          min_crossover_threshold_per_year: int = 12) -> tuple:
        """
        This method consists of the final two criterions checks in the third stage of the proposed
        framework which involves; the calculation and check, of the half life of the given pair spread
        and the amount of mean crossovers throughout a set period, in this case in a year.

        :param spreads_df: (pd.DataFrame) : Hedge ratio adjusted spreads DataFrame
        :param pairs: (list) : List of asset name pairs to be analyzed
        :param min_crossover_threshold_per_year: (int) : minimum amount of mean crossovers per year
        :return (tuple) :
            hl_pass_pairs: (pd.DataFrame) : DataFrame of pairs that passed the half life test and their values
            final_pairs: (pd.DataFrame) : DataFrame of final pairs and their mean crossover counts
        """

        hl_pass_pairs = []
        final_pairs = []

        if len(pairs) != 0:
            ou_results = _outer_ou_loop(
                spreads_df, molecule=pairs, test_period='2Y', cross_overs_per_delta=min_crossover_threshold_per_year)

            final_selection = ou_results[ou_results['hl'] > 1]

            final_selection = final_selection[ou_results['hl'] < 365]

            hl_pass_pairs = final_selection

            final_selection = final_selection[ou_results['crossovers']]

            final_pairs = final_selection

        else:
            raise Exception("No pairs have been found")

        return hl_pass_pairs, final_pairs

    def unsupervised_candidate_pair_selector(
            self, pvalue_threshold: int = 0.01, hurst_exp_threshold: int = 0.5, min_crossover_threshold_per_year: int = 12) -> list: # pragma: no cover
        """
        Third step of the framework; The clusters found in step two are used to generate a list of possible pairwise
        combinations. The combinations generated are then checked to see if they comply with the criteria supplied in the
        paper: the pair being cointegrated, the hurst exponent being <0.5, the spread moves within convenient periods and
        finally that the spread reverts to the mean with enough frequency.

        :param pvalue_threshold: (int) : max p-value threshold to be used in the cointegration tests
        :param hurst_exp_threshold: (int) : max hurst threshold value
        :param min_crossover_threshold_per_year: (int) : minimum amount of mean crossovers per year
        :return final_pairs: (list) : tuple list of final pairs
        """

        if self.clust_labels_ is None:
            raise Exception("The needed cluster labels have not been computed yet",
                            "Please run cluster() before this method")

        # Generate needed pairwise combinations and remove unneccessary
        # duplicates.

        c_labels = np.unique(self.clust_labels_[self.clust_labels_ != -1])
        cluster_x_cointegration_combinations = self._generate_pairwise_combinations(c_labels)
        self.cluster_pairs_combinations = cluster_x_cointegration_combinations

        return self._criterion_selection(cluster_x_cointegration_combinations, pvalue_threshold, hurst_exp_threshold, min_crossover_threshold_per_year)

    def _criterion_selection(self, cluster_x_cointegration_combinations: list, pvalue_threshold: int = 0.01,
                             hurst_exp_threshold: int = 0.5, min_crossover_threshold_per_year: int = 12) -> list:
        """
        Third step of the framework; The clusters found in step two are used to generate a list of possible pairwise
        combinations. The combinations generated are then checked to see if they comply with the criteria supplied in the
        paper: the pair being cointegrated, the hurst exponent being <0.5, the spread moves within convenient periods and
        finally that the spread reverts to the mean with enough frequency.

        :param cluster_x_cointegration_combinations: (list) : list of asset pairs
        :param pvalue_threshold: (int) : max p-value threshold to be used in the cointegration tests
        :param hurst_exp_threshold: (int) : max hurst threshold value
        :param min_crossover_threshold_per_year: (int) : minimum amount of mean crossovers per year
        :return final_pairs: (list) : tuple list of final pairs
        """

        # Selection Criterion One: First, it is imposed that pairs are
        # cointegrated, using a p-value of 1%.

        cointegration_results = _outer_cointegration_loop(
            self.prices_df, cluster_x_cointegration_combinations)

        passing_pairs = cointegration_results.loc[cointegration_results['pvalue']
                                                  <= pvalue_threshold]

        self.coint_pass_pairs = passing_pairs

        # Selection Criterion Two: Then, the spread’s Hurst exponent,
        # represented by H should be smaller than 0.5.

        spreads_df, hurst_pass_pairs = self._hurst_criterion(
            passing_pairs, hurst_exp_threshold)

        self.spreads_df = spreads_df

        self.hurst_pass_pairs = hurst_pass_pairs

        # Selection Criterion Three & Four: Additionally, the half-life period, represented by hl, should
        # lay between one day and one year. Finally, it is imposed that the spread crosses a mean at least
        # 12 times per year.

        hl_pass_pairs, final_pairs = self._final_criterions(
            spreads_df, hurst_pass_pairs.index.values, min_crossover_threshold_per_year)

        self.hl_pass_pairs = hl_pass_pairs

        self.final_pairs = final_pairs

        return final_pairs.index.values


    def plot_selected_pairs(self): # pragma: no cover
        """
        Plots the final selection of pairs.
        """

        if self.final_pairs is None:
            raise Exception("The needed pairs have not been computed yet",
                            "Please run criterion_selector() before this method")

        if len(self.final_pairs) == 0:
            raise Exception("No valid pairs have been found!")

        _, axs = plt.subplots(len(self.final_pairs),
                              figsize=(15, 3 * len(self.final_pairs)))

        for i, frame in enumerate(self.final_pairs.index.values):
            rets_asset_one = np.log(self.prices_df.loc[:, frame[0]]).diff()
            rets_asset_two = np.log(self.prices_df.loc[:, frame[1]]).diff()

            axs[i].plot(rets_asset_one.cumsum())
            axs[i].plot(rets_asset_two.cumsum())
            axs[i].legend([frame[0], frame[1]])

    def describe(self) -> pd.DataFrame:
        """
        Returns the Pairs Selector Summary statistics.
        The following statistics are included - the number of clusters, total possible pair combinations,
        the number of pairs that passed the cointegration threshold, the number of pairs that passed the
        hurst exponent threshold, the number of pairs that passed the half life threshold and the number
        of final set of pairs.

        :return: (pd.DataFrame) Dataframe of summary statistics.
        """

        no_clusters = len(list(set(self.clust_labels_))) - 1
        no_paircomb = len(self.cluster_pairs_combinations)
        no_hurstpair = len(self.hurst_pass_pairs)
        no_hlpair = len(self.hl_pass_pairs)

        info = []

        info.append(("No. of Clusters", no_clusters))
        info.append(("Total Pair Combinations", no_paircomb))
        info.append(("Pairs passing Coint Test", len(self.coint_pass_pairs)))
        info.append(("Pairs passing Hurst threshold", no_hurstpair))
        info.append(("Pairs passing Half Life threshold", no_hlpair))
        info.append(("Final Set of Pairs", len(self.final_pairs)))
        return pd.DataFrame(info)

    def describe_extra(self) -> pd.DataFrame:
        """
        Returns information on each pair selected.
        The following statistics are included - both legs of the pair, cointegration (t-value, p-value, hedge_ratio),
        hurst_exponent, half_life, no_mean_crossovers.

        :return: (pd.DataFrame) Dataframe of pair statistics.
        """

        return pd.concat([self.coint_pass_pairs,
                          self.hurst_pass_pairs,
                          self.hl_pass_pairs
                          ], axis=1) \
            .dropna() \
            .reset_index() \
            .rename(columns={
                'level_0': 'leg_1',
                'level_1': 'leg_2',
                'hl': 'half_life'}) \
            .drop(['constant'], axis=1, errors='ignore') \


    @staticmethod
    def _convert_to_tuple(arr: np.array) -> tuple: # pragma: no cover
        """
        Returns a list converted to a tuple.

        :param arr: (np.array) Input array to be converted.
        :return: (tuple) List converted to tuple.
        """
        return tuple(i for i in arr)

    def describe_pairs_sectoral_info(
            self, leg_1: list, leg_2: list, sectoral_info_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns information on each pair selected.
        The following statistics are included - both legs of the pair, cointegration (t-value, p-value, hedge_ratio),
        hurst_exponent, half_life, no_mean_crossovers.

        :param leg_1: (list) vector Symbol of the first asset.
        :param leg_2: (list) vector Symbol of the second asset.
        :param sectoral_info_df: (pd.DataFrame) DataFrame with two columns (ticker, sector) to be used in the output.
        :return: (pd.DataFrame) Dataframe of pair sectoral statistics.
        """

        leg_1_info = []
        leg_2_info = []

        for tck in leg_1:
            if sectoral_info_df[sectoral_info_df['ticker'] == tck].empty:
                leg_1_info.append(('', '', ''))
            else:
                leg_1_info.append(
                    self._convert_to_tuple(sectoral_info_df[sectoral_info_df['ticker'] == tck].values[0]))

        for tck in leg_2:
            if sectoral_info_df[sectoral_info_df['ticker'] == tck].empty:
                leg_2_info.append(('', '', ''))
            else:
                leg_2_info.append(
                    self._convert_to_tuple(sectoral_info_df[sectoral_info_df['ticker'] == tck].values[0]))

        info_df = pd.concat(
            [pd.DataFrame(leg_1_info), pd.DataFrame(leg_2_info)], axis=1)
        info_df.columns = ['Leg 1 Ticker', 'Industry',
                           'Sector', 'Leg 2 Ticker', 'Industry', 'Sector']

        return info_df

    @staticmethod
    def hurst(data: pd.DataFrame, max_lags: int = 100) -> int:
        """
        Hurst Exponent Calculation

        :param data: (pd.DataFrame) Time Series that is going to be analyzed.
        :param max_lags: (int) Maximum amount of lags to be used calculating tau.
        :return: (int) hurst exponent.
        """
        lags = range(2, max_lags)
        tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag])))
               for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
