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

from arbitragelab.util import devadarsh
from arbitragelab.util.indexed_highlight import IndexedHighlight


class OPTICSDBSCANPairsClustering:
    """
    Implementation of the Proposed Pairs Selection Framework in the following paper:
    `"A Machine Learning based Pairs Trading Investment Strategy."
    <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.

    The method consists of 2 parts: dimensionality reduction and clustering of features.
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

        self.clust_labels_ = []

        devadarsh.track('OPTICSDBSCANPairsClustering')

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

    def cluster_using_optics(self, **kwargs: dict) -> list:
        """
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
        return self._generate_pairwise_combinations(self.clust_labels_)

    def cluster_using_dbscan(self, **kwargs: dict) -> list:
        """
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
        return self._generate_pairwise_combinations(self.clust_labels_)

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
