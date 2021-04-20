# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the ML based Pairs Selection Framework described by Simão Moraes
Sarmento and Nuno Horta in `"A Machine Learning based Pairs Trading Investment Strategy." <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.
"""

import itertools
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

from arbitragelab.ml_approach.stat_arb_utils import _outer_cointegration_loop, _outer_ou_loop
from arbitragelab.util import devadarsh
from arbitragelab.util.indexed_highlight import IndexedHighlight
from arbitragelab.util.hurst import get_hurst_exponent


class PairsSelector:
    """
    Implementation of the Proposed Pairs Selection Framework in the following paper:
    `"A Machine Learning based Pairs Trading Investment Strategy."
    <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.

    The method consists of three parts: dimensionality reduction, clustering of features and
    finally the selection of pairs with the use of a set of heuristics.
    """

    def __init__(self, universe: pd.DataFrame):
        """
        Constructor.
        Sets up the price series needed for the next step.

        :param universe: (pd.DataFrame) Asset prices universe.
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

        devadarsh.track('PairsSelector')

    def dimensionality_reduction_by_components(self, num_features: int = 10):
        """
        Processes and scales the prices universe supplied in the constructor, into returns.

        Then reduces the resulting data using pca down to the amount of dimensions needed
        to be used as a feature vector in the clustering step. Optimal ranges for the dimensions
        required in the feature vector should be <15.

        :param num_features: (int) Used to select pca n_components to be used in the feature vector.
        """

        if self.prices_df is None:
            raise Exception(
                "Please input a valid price series before running this method.")

        # Cleaning
        returns_df = (self.prices_df - self.prices_df.shift(1))
        returns_df = returns_df / self.prices_df.shift(1)
        returns_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        returns_df.ffill(inplace=True)

        # Scaling
        scaler = StandardScaler()
        scaled_returns_df = pd.DataFrame(scaler.fit_transform(returns_df))
        scaled_returns_df.columns = returns_df.columns
        scaled_returns_df.set_index(returns_df.index)
        scaled_returns_df.dropna(inplace=True)

        # Reducing
        pca = PCA(n_components=num_features)
        pca.fit(scaled_returns_df)
        self.feature_vector = pd.DataFrame(pca.components_)
        self.feature_vector.columns = returns_df.columns
        self.feature_vector = self.feature_vector.T

    def plot_pca_matrix(self, alpha: float = 0.2, figsize: tuple = (15, 15)) -> list:
        """
        Plots the feature vector on a scatter matrix.

        :param alpha: (float) Opacity level to be used in the plot.
        :param figsize: (tuple) Tuple describing the size of the plot.
        :return: (list) List of Axes objects.
        """

        feature_vector = self.feature_vector.copy()

        cols = ["Component " + str(i) for i in feature_vector.columns]
        feature_vector.columns = cols

        axes = pd.plotting.scatter_matrix(
            feature_vector, alpha=alpha, figsize=figsize)

        new_labels = [round(float(i.get_text()), 2)
                      for i in axes[0, 0].get_yticklabels()]
        axes[0, 0].set_yticklabels(new_labels)

        return axes

    def cluster_using_optics(self, **kwargs: dict):
        """
        Second step of the framework;

        Doing Unsupervised Learning on the feature vector supplied from the first step.
        The clustering method used is OPTICS, chosen mainly for it being basically parameterless.

        :param kwargs: (dict) Arguments to be passed to the clustering algorithm.
        """

        if self.feature_vector is None:
            raise Exception("The needed feature vector has not been computed yet.",
                            "Please run dimensionality_reduction() before this method.")

        clust = OPTICS(**kwargs)
        clust.fit(self.feature_vector)
        self.clust_labels_ = clust.labels_

    def cluster_using_dbscan(self, **kwargs: dict):
        """
        Second step of the framework;

        Doing Unsupervised Learning on the feature vector supplied from the first step. The
        second clustering method used is DBSCAN, for when the user needs a more hands-on approach
        to doing the clustering step, given the parameter sensitivity of this method.

        :param kwargs: (dict) Arguments to be passed to the clustering algorithm.
        """

        if self.feature_vector is None:
            raise Exception("The needed feature vector has not been computed yet.",
                            "Please run dimensionality_reduction() before this method.")

        clust = DBSCAN(**kwargs)
        clust.fit(self.feature_vector)
        self.clust_labels_ = clust.labels_

    def plot_clustering_info(self, n_dimensions: int = 2, method: str = "",
                             figsize: tuple = (10, 10)) -> Axes:
        """
        Reduces the feature vector found in the dimensionality reduction step, further
        down to the specified 'n_dimensions' argument using TSNE and then plots the
        clusters found, on a scatter plot.

        :param n_dimensions: (int) Selected dimension to be used in the T-SNE plot.
        :param method: (str) String to be used as title in the plot.
        :param figsize: (tuple) Tuple describing the size of the plot.
        :return: (Axes) Axes object.
        """

        # First we check if the feature vector is empty or else we don't have anything
        # to plot.
        if self.feature_vector is None:
            raise Exception("The needed feature vector has not been computed yet.",
                            "Please run dimensionality_reduction() before this method.")

        # Check if there are enough clusters.
        if len(self.clust_labels_) == 0:
            raise Exception("The needed cluster labels have not been computed yet.",
                            "Please run cluster() before this method.")

        # The plotting methods support both 2d and 3d representations, so here we limit
        # the users choice to either 2 or 3 dimensions.
        if (n_dimensions > 3) or (n_dimensions < 1):
            raise Exception("Select a valid dimension! (more than 1 and less than 3).")

        no_of_classes = len(np.unique(self.clust_labels_))

        fig = plt.figure(facecolor='white', figsize=figsize)

        tsne = TSNE(n_components=n_dimensions)

        tsne_fv = pd.DataFrame(tsne.fit_transform(self.feature_vector),
                               index=self.feature_vector.index)

        if n_dimensions == 2:
            return self.plot_2d_scatter_plot(fig, tsne_fv, no_of_classes, method)

        return self.plot_3d_scatter_plot(tsne_fv, no_of_classes, method)

    def plot_3d_scatter_plot(self, tsne_df: pd.DataFrame, no_of_classes: int,
                             method: str = "") -> Axes:
        """
        Plots the clusters found on a 3d scatter plot. In this method it is
        assumed that the data being plotted has been pre-processed using TSNE
        constrained to three components to provide the best visualization of
        dataset possible.

        :param tsne_df: (pd.DataFrame) Data reduced using T-SNE.
        :param no_of_classes: (int) Number of unique clusters/classes.
        :param method: (str) String to be used as title in the plot.
        :return: (Axes) Axes object.
        """

        ax_object = plt.subplot(111, projection='3d')
        paths_collection = []

        # For each cluster.
        for cluster in range(0, no_of_classes):
            # Get specific cluster data from the tsne processed dataframe.
            cluster_data = tsne_df[self.clust_labels_ == cluster]

            # Plot the cluster data by column index [0, 1, 2] -> [x, y, z].
            paths = ax_object.plot(cluster_data.loc[:, 0], cluster_data.loc[:, 1],
                                   cluster_data.loc[:, 2], alpha=0.7, marker='.',
                                   linestyle='None', label=list(cluster_data.index),
                                   markersize=30)

            # Stash the list of Line2D objects for future use.
            paths_collection.append(paths)

        # Flatten the paths array and instantiate the IndexedHighlight class
        # that will manage the selection and highlighting of the plotted
        # clusters.
        IndexedHighlight(np.ravel(paths_collection),
                         formatter='{label}'.format)

        # Plot the noisy samples which are not included in a leaf cluster labelled as -1,
        # by column index [0, 1, 2] -> [x, y, z].
        ax_object.plot(tsne_df.iloc[self.clust_labels_ == -1, 0], tsne_df.iloc[self.clust_labels_ == -1, 1],
                       tsne_df.iloc[self.clust_labels_ == -1, 2], 'k+', alpha=0.1)

        # Set the chart title.
        ax_object.set_title('Automatic Clustering\n' + method)

        plt.show()

        return ax_object

    def plot_2d_scatter_plot(self, fig: Figure, tsne_df: pd.DataFrame, no_of_classes: int,
                             method: str = "") -> Axes:
        """
        Plots the clusters found on a 2d scatter plot.

        :param fig: (Figure) Figure object, needed for the styling of the spline.
        :param tsne_df: (pd.DataFrame) Data reduced using T-SNE.
        :param no_of_classes: (int) Number of unique clusters/classes.
        :param method: (str) String to be used as title in the plot.
        :return: (Axes) Axes object.
        """

        ax_object = fig.add_subplot(111)

        # Set spine styling.
        ax_object.spines['left'].set_position('center')
        ax_object.spines['left'].set_alpha(0.3)
        ax_object.spines['bottom'].set_position('center')
        ax_object.spines['bottom'].set_alpha(0.3)
        ax_object.spines['right'].set_color('none')
        ax_object.spines['top'].set_color('none')

        ax_object.xaxis.set_ticks_position('bottom')
        ax_object.yaxis.set_ticks_position('left')
        ax_object.tick_params(which='major', labelsize=18)

        paths_collection = []

        # For each cluster.
        for cluster in range(0, no_of_classes):
            # Get specific cluster data from the tsne process dataframe.
            cluster_data = tsne_df[self.clust_labels_ == cluster]

            # Plot the cluster data by column index [0, 1] -> [x, y].
            paths = ax_object.plot(cluster_data.loc[:, 0], cluster_data.loc[:, 1],
                                   label=list(cluster_data.index), markersize=30,
                                   alpha=0.75, marker='.', linestyle='None')

            # Stash the list of Line2D objects for future use.
            paths_collection.append(paths)

        # Flatten the paths array and instantiate the IndexedHighlight class
        # that will manage the selection and highlighting of the plotted
        # clusters.
        IndexedHighlight(np.ravel(paths_collection),
                         formatter='{label}'.format)

        # Plot the noisy samples which are not included in a leaf cluster labelled as -1,
        # by column index [0, 1] -> [x, y].
        ax_object.plot(tsne_df.iloc[self.clust_labels_ == -1, 0],
                       tsne_df.iloc[self.clust_labels_ == -1, 1], 'k+', alpha=0.1)

        # Set the chart title.
        ax_object.set_title('Automatic Clustering\n' + method)

        plt.show()

        return ax_object

    def plot_knee_plot(self) -> Axes:
        """
        This method will plot the k-distance graph, ordered from the largest to the smallest value.

        The values where this plot shows an "elbow" should be a reference to the user of the optimal
        ε parameter to be used for the DBSCAN clustering method.

        :return: (Axes) Axes object.
        """

        fig = plt.figure()
        ax_object = fig.add_subplot()

        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(self.feature_vector)
        result, _ = nbrs.kneighbors(self.feature_vector)
        result = np.sort(result, axis=0)
        result = result[:, 1]

        ax_object.set_ylabel('k-distance')
        ax_object.set_xlabel('# of data rows')
        ax_object.plot(result)
        ax_object.set_title('Knee Plot')

        return ax_object

    def _generate_pairwise_combinations(self, labels: list) -> list:
        """
        This method will loop through all generated clusters (except -1) and generate
        pairwise combinations of the assets in each cluster.

        :param labels: (list) List of unique labels.
        :return: (list) List of asset name pairs.
        """

        if len(labels) == 0:
            raise Exception("No Labels have been found!")

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

        :param sectoral_info: (pd.DataFrame) List of asset name pairs to be analyzed tagged with respective sector.
        :return: (list) List of asset name pairs.
        """

        pair_combinations = []
        sectors = np.unique(sectoral_info['sector'])

        for sector in sectors:
            sector_list = sectoral_info[sectoral_info['sector'] == sector]
            cluster_x = sector_list['ticker'].values
            if len(cluster_x) > 1:
                for combination in list(itertools.combinations(cluster_x, 2)):
                    pair_combinations.append(combination)

        return pair_combinations

    def _hurst_criterion(self, pairs: pd.DataFrame,
                         hurst_exp_threshold: int = 0.5) -> tuple:
        """
        This method will go through all the pairs given, calculate the needed spread and run
        the Hurst exponent test against each one.

        :param pairs: (pd.DataFrame) DataFrame of asset name pairs to be analyzed.
        :param hurst_exp_threshold: (int) Max Hurst threshold value.
        :return: (pd.DataFrame, pd.DataFrame) The first DataFrame consists of the Hedge ratio adjusted spreads
            and the second DataFrame consists of pairs that passed the Hurst check / their respective Hurst value.
        """

        hurst_pass_pairs = []
        spreads_lst = []
        spreads_cols = []

        if len(pairs) == 0:
            raise Exception("No pairs have been found!")

        for idx, frame in pairs.iterrows():
            asset_one = self.prices_df.loc[:, idx[0]].values
            asset_two = self.prices_df.loc[:, idx[1]].values

            spread_ts = (asset_one - asset_two * frame['hedge_ratio'])
            hurst_exp = get_hurst_exponent(spread_ts)

            if hurst_exp < hurst_exp_threshold:
                hurst_pass_pairs.append((idx, hurst_exp))
                spreads_lst.append(spread_ts)
                spreads_cols.append(str(idx))

        spreads_df = pd.DataFrame(data=spreads_lst).T
        spreads_df.columns = spreads_cols
        spreads_df.index = pd.to_datetime(self.prices_df.index)

        hurst_pass_pairs_df = pd.DataFrame(data=hurst_pass_pairs)
        hurst_pass_pairs_df.columns = ['pairs', 'hurst_exponent']
        hurst_pass_pairs_df.set_index('pairs', inplace=True)
        hurst_pass_pairs_df.index.name = None

        return spreads_df, hurst_pass_pairs_df

    @staticmethod
    def _final_criterions(spreads_df: pd.DataFrame,
                          pairs: list, test_period: str = '2Y',
                          min_crossover_threshold_per_year: int = 12) -> tuple:
        """
        This method consists of the final two criterions checks in the third stage of the proposed
        framework which involves; the calculation and check, of the half-life of the given pair spread
        and the amount of mean crossovers throughout a set period, in this case in a year.

        :param spreads_df: (pd.DataFrame) Hedge ratio adjusted spreads DataFrame.
        :param pairs: (list) List of asset name pairs to be analyzed.
        :param test_period: (str) Time delta format, to be used as the time
            period where the mean crossovers will be calculated.
        :param min_crossover_threshold_per_year: (int) Minimum amount of mean crossovers per year.
        :return: (pd.DataFrame, pd.DataFrame) The first is a DataFrame of pairs that passed the half-life
            test and the second is a DataFrame of final pairs and their mean crossover counts.
        """

        if len(pairs) == 0:
            raise Exception("No pairs have been found!")

        ou_results = _outer_ou_loop(spreads_df, molecule=pairs, test_period=test_period,
                                    cross_overs_per_delta=min_crossover_threshold_per_year)

        final_selection = ou_results[ou_results['hl'] > 1]

        final_selection = final_selection.loc[ou_results['hl'] < 365]

        hl_pass_pairs = final_selection

        final_selection = final_selection.loc[ou_results['crossovers']]

        final_pairs = final_selection

        return hl_pass_pairs, final_pairs

    def unsupervised_candidate_pair_selector(self, hedge_ratio_calculation: str = 'OLS',
                                             adf_cutoff_threshold: float = 0.95,
                                             hurst_exp_threshold: int = 0.5,
                                             min_crossover_threshold_per_year: int = 12,
                                             test_period: str = '2Y') -> list:
        """
        Third step of the framework;

        The clusters found in step two are used to generate a list of possible pairwise combinations.
        The combinations generated are then checked to see if they comply with the criteria supplied
        in the paper: the pair being cointegrated, the Hurst exponent being <0.5, the spread moves
        within convenient periods and finally that the spread reverts to the mean with enough frequency.

        :param hedge_ratio_calculation: (str) Defines how hedge ratio is calculated. Can be either 'OLS,
                                        'TLS' (Total Least Squares) or 'min_half_life'.
        :param adf_cutoff_threshold: (float) ADF test threshold used to define if the spread is cointegrated. Can be
                                             0.99, 0.95 or 0.9.
        :param hurst_exp_threshold: (int) Max Hurst threshold value.
        :param min_crossover_threshold_per_year: (int) Minimum amount of mean crossovers per year.
        :param test_period: (str) Time delta format, to be used as the time
            period where the mean crossovers will be calculated.
        :return: (list) Tuple list of final pairs.
        """

        if len(self.clust_labels_) == 0:
            raise Exception("The needed cluster labels have not been computed yet.",
                            "Please run cluster() before this method.")

        # Generate needed pairwise combinations and remove unnecessary
        # duplicates.

        c_labels = np.unique(self.clust_labels_[self.clust_labels_ != -1])
        cluster_x_cointegration_combinations = self._generate_pairwise_combinations(c_labels)
        self.cluster_pairs_combinations = cluster_x_cointegration_combinations

        return self._criterion_selection(cluster_x_cointegration_combinations, hedge_ratio_calculation,
                                         adf_cutoff_threshold, hurst_exp_threshold,
                                         min_crossover_threshold_per_year, test_period)

    def _criterion_selection(self, cluster_x_cointegration_combinations: list, hedge_ratio_calculation: str = 'OLS',
                             adf_cutoff_threshold: float = 0.95, hurst_exp_threshold: int = 0.5,
                             min_crossover_threshold_per_year: int = 12,
                             test_period: str = '2Y') -> list:
        """
        Third step of the framework;

        The clusters found in step two are used to generate a list of possible pairwise combinations.
        The combinations generated are then checked to see if they comply with the criteria supplied
        in the paper: the pair being cointegrated, the Hurst exponent being <0.5, the spread moves
        within convenient periods and finally that the spread reverts to the mean with enough frequency.

        :param cluster_x_cointegration_combinations: (list) List of asset pairs.
        :param hedge_ratio_calculation: (str) Defines how hedge ratio is calculated. Can be either 'OLS,
                                        'TLS' (Total Least Squares) or 'min_half_life'.
        :param adf_cutoff_threshold: (float) ADF test threshold used to define if the spread is cointegrated. Can be
                                             0.99, 0.95 or 0.9.
        :param hurst_exp_threshold: (int) Max Hurst threshold value.
        :param min_crossover_threshold_per_year: (int) Minimum amount of mean crossovers per year.
        :param test_period: (str) Time delta format, to be used as the time
            period where the mean crossovers will be calculated.
        :return: (list) Tuple list of final pairs.
        """

        # Selection Criterion One: First, it is imposed that pairs are cointegrated

        cointegration_results = _outer_cointegration_loop(
            self.prices_df, cluster_x_cointegration_combinations, hedge_ratio_calculation=hedge_ratio_calculation)

        passing_pairs = cointegration_results.loc[cointegration_results['coint_t']
                                                  <= cointegration_results[
                                                      'p_value_{}%'.format(int(adf_cutoff_threshold * 100))]]

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

        hl_pass_pairs, final_pairs = self._final_criterions(spreads_df, hurst_pass_pairs.index.values,
                                                            test_period,
                                                            min_crossover_threshold_per_year)

        self.hl_pass_pairs = hl_pass_pairs

        self.final_pairs = final_pairs

        return final_pairs.index.values

    def plot_selected_pairs(self) -> list:
        """
        Plots the final selection of pairs.

        :return: (list) List of Axes objects.
        """

        if (self.final_pairs is None) or (len(self.final_pairs) == 0):
            raise Exception("The needed pairs have not been computed yet.",
                            "Please run criterion_selector() before this method.")

        if len(self.final_pairs) > 40:
            raise Exception("The amount of pairs to be plotted cannot exceed 40",
                            "without causing system instability.")

        _, axs = plt.subplots(len(self.final_pairs), figsize=(15, 3 * len(self.final_pairs)))

        for ax_object, frame in zip(axs, self.final_pairs.index.values):
            rets_asset_one = np.log(self.prices_df.loc[:, frame[0]]).diff()
            rets_asset_two = np.log(self.prices_df.loc[:, frame[1]]).diff()

            ax_object.plot(rets_asset_one.cumsum())
            ax_object.plot(rets_asset_two.cumsum())
            ax_object.legend([frame[0], frame[1]])

        return axs

    def plot_single_pair(self, pair: tuple) -> Axes:
        """
        Plots the given pair.

        :param pair: (tuple) Tuple of asset names.
        :return: (Axes) Axes object.
        """
        _, ax_object = plt.subplots(1, figsize=(15, 3))

        rets_asset_one = np.log(self.prices_df.loc[:, pair[0]]).diff()
        rets_asset_two = np.log(self.prices_df.loc[:, pair[1]]).diff()

        ax_object.plot(rets_asset_one.cumsum())
        ax_object.plot(rets_asset_two.cumsum())
        ax_object.legend([pair[0], pair[1]])

        return ax_object

    def describe(self) -> pd.DataFrame:
        """
        Returns the Pairs Selector Summary statistics.

        The following statistics are included - the number of clusters, total possible pair combinations,
        the number of pairs that passed the cointegration threshold, the number of pairs that passed the
        Hurst exponent threshold, the number of pairs that passed the half-life threshold and the number
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
        info.append(("Pairs passing Half-Life threshold", no_hlpair))
        info.append(("Final Set of Pairs", len(self.final_pairs)))

        return pd.DataFrame(info)

    def describe_extra(self) -> pd.DataFrame:
        """
        Returns information on each pair selected.

        The following statistics are included - both legs of the pair, cointegration (t-value, p-value, hedge_ratio),
        hurst_exponent, half_life, no_mean_crossovers.

        :return: (pd.DataFrame) Dataframe of pair statistics.
        """

        description_df = pd.concat([self.coint_pass_pairs, self.hurst_pass_pairs, self.hl_pass_pairs], axis=1)
        description_df.dropna(inplace=True)
        description_df.reset_index(inplace=True)
        description_df.rename(columns={'level_0': 'leg_1', 'level_1': 'leg_2', 'hl': 'half_life'}, inplace=True)
        description_df.drop(['constant'], axis=1, errors='ignore', inplace=True)

        return description_df

    @staticmethod
    def _convert_to_tuple(arr: np.array) -> tuple:
        """
        Returns a list converted to a tuple.

        :param arr: (np.array) Input array to be converted.
        :return: (tuple) List converted to tuple.
        """

        return tuple(i for i in arr)

    def describe_pairs_sectoral_info(self, leg_1: list, leg_2: list,
                                     sectoral_info_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns information on each pair selected.

        The following statistics are included - both legs of the pair, cointegration (t-value, p-value, hedge_ratio),
        hurst_exponent, half_life, no_mean_crossovers.

        :param leg_1: (list) Vector of asset names.
        :param leg_2: (list) Vector of asset names.
        :param sectoral_info_df: (pd.DataFrame) DataFrame with two columns [ticker, sector] to be used in the output.
        :return: (pd.DataFrame) Dataframe of pair sectoral statistics.
        """

        leg_1_info = self._loop_through_sectors(leg_1, sectoral_info_df)
        leg_2_info = self._loop_through_sectors(leg_2, sectoral_info_df)

        info_df = pd.concat([leg_1_info, leg_2_info], axis=1)
        info_df.columns = ['Leg 1 Ticker', 'Industry',
                           'Sector', 'Leg 2 Ticker', 'Industry', 'Sector']

        return info_df

    def _loop_through_sectors(self, tickers: list, sectoral_info_df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method that loops through sectoral info.

        :param tickers: (list) Vector of asset names.
        :param sectoral_info_df: (pd.DataFrame) DataFrame with two columns [ticker, sector] to be used in the output.
        :return: (pd.DataFrame) Dataframe of ticker sectoral statistics.
        """

        tck_info = []

        for tck in tickers:
            leg_sector_info = sectoral_info_df[sectoral_info_df['ticker'] == tck]
            if leg_sector_info.empty:
                tck_info.append(('', '', ''))
            else:
                info_as_tuple = self._convert_to_tuple(leg_sector_info.values[0])
                tck_info.append(info_as_tuple)

        return pd.DataFrame(tck_info)
