# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module for implementing some quick pairs selection algorithms for copula-based strategies.
"""
# pylint: disable = invalid-name
from typing import Union, List, Tuple
from itertools import combinations
import numpy as np
import scipy.stats as ss
import pandas as pd


class PairsSelector:
    r"""
    The class that quickly select pairs for copula-based trading strategies.

    This class selects potential pairs for copula-based trading strategies. Methods include Spearman's rho, Kendall's
    tau and Euclidean distance on normalized prices. Those methods are relatively quick to perform and is generally
    used in literature for copula-based pairs trading framework. For more sophisticated ML based pairs selection
    methods, please refer to :code:`arbitragelab.ml_approach`.
    """

    def __init__(self):
        """
        Class initiation.
        """

    def rank_pairs(self, stocks_universe: pd.DataFrame, method: str = 'kendall tau',
                   nan_option: Union[str, None] = 'forward fill',
                   keep_num_pairs: Union[int, None] = None) -> pd.Series:
        """
        Rank all pairs from the stocks universe by a given method.

        method choices are 'spearman rho', 'kendall tau', 'euc distance'.
        nan_options choices are 'backward fill', 'linear interp', None.
        keep_num_pairs choices are all integers not greater than the number of all available pairs. None Means keeping
        all pairs.

        Spearman's rho is calculated faster, however the performance suffers from outliers and tied ranks when compared
        to Kendall's tau. Euclidean distance is calculated based on a pair's cumulative return (normalized prices).
        User should keep in mind that Spearman's rho and Kendall's tau will generally give similar results but the top
        pairs may still drift away in terms of normalized prices.

        Note: ALL NaN values need to be filled, otherwise you will likely get the scores all in NaN value. We suggest
        'forward fill' to avoid look-ahead bias. User should be aware that 'linear interp' will linearly interpolate
        the NaN value and will thus introduce look-ahead bias. This method will internally fill the NaN values. Also
        the very first row of data cannot have NaN values.

        :param stocks_universe: (pd.DataFrame) The stocks universe to be analyzed. Require no multi-indexing for
            columns.
        :param method: (pd.DataFrame) Optional. The method to pick pairs. One can choose from ['spearman rho',
            'kendall tau', 'euc distance'] for Spearman's rho, Kendall's tau and Euclidean distance. Defaults to
            'kendall tau'.
        :param nan_option: (Union[str, None]) Optional. The method to fill NaN value. one can choose from
            ['forward fill', 'linear interp', None]. Defaults to 'forward fill'.
        :param keep_num_pairs: (Union[int, None]) Optional. The number of top ranking pairs to keep. Defaults to None,
            which means all pairs will be returned.
        :return: (pd.Series) The selected pairs ranked descending in their scores (top 'correlated' pairs on the top).
        """

        # 0. Preprocessing for NaN value
        stocks_universe = stocks_universe.copy()
        stocks_universe = self._pre_processing_nan(stocks_universe, nan_option)

        # 1. Get all the possible pairs from the stocks universe
        all_names = stocks_universe.columns
        pairs_names = self._select_all_pairs(all_names)

        # 2. Calculating scores based on different methods
        scores = self._calculate_scores(stocks_universe, pairs_names, method)

        # 3. Ranks pairs based on the scores
        scores.sort_values(ascending=False, inplace=True)

        # 4. Return the result in a data frame with columns 'Pair Name', 'Score'. Row index is the rank.
        if keep_num_pairs is not None:
            scores = scores[:keep_num_pairs]

        return scores

    @staticmethod
    def _pre_processing_nan(stocks_universe: pd.DataFrame, nan_option: str) -> pd.DataFrame:
        """
        Pre-processing data for NaN values.

        Available options are 'forward fill', 'linear interp', None.
        ALL NaN values need to be filled, otherwise you will likely get the scores all in NaN value. We suggest
        'forward fill' to avoid look-ahead bias. User should be aware that 'linear interp' will linearly interpolate
        the NaN value and will thus introduce look-ahead bias.

        :param stocks_universe: (pd.DataFrame) The stocks universe to be analyzed. Require no multi-indexing for
            columns.
        :param nan_option: (Union[str, None]) Optional. The method to fill NaN value. one can choose from
            ['forward fill', 'linear interp', None]. Defaults to 'forward fill'.
        :return: (pd.DataFrame) The processed data frame.
        """

        if nan_option == 'forward fill':
            stocks_universe = stocks_universe.fillna(method='ffill', inplace=False)

        if nan_option == 'linear interp':
            stocks_universe = stocks_universe.interpolate(method='linear', inplace=False)

        return stocks_universe

    @staticmethod
    def _select_all_pairs(all_names: pd.Series) -> List[Tuple[str, str]]:
        """
        Select all possible pairs by from the given names.

        :param all_names: (pd.Series) All the available names in the stocks universe.
        :return: (List[Tuple[str, str]]) All the possible pairs to be analyzed in list of tuples of their names.
        """

        comb_iter = combinations(all_names, 2)
        result = list(comb_iter)

        return result

    def _calculate_scores(self, stocks_universe: pd.DataFrame, pairs_names: list, method: str) -> pd.Series:
        """
        Calculate scores based on different methods.

        The scores is recorded in a series with row index multi-indexed in the form of ('ABC', 'XYZ') for stocks pair
        'ABC' and 'XYZ'.

        :param stocks_universe: (pd.DataFrame) The stocks universe to be analyzed. Require no multi-indexing for
            columns.
        :param pairs_names: (list) All the possible pairs to be analyzed in list of tuples of their names.
        :param method: (pd.DataFrame) The method to pick pairs. One can choose from ['spearman rho',
            'kendall tau', 'euc distance'] for Spearman's rho, Kendall's Tau and Euclidean distance.
        :return: (pd.Series) The calculated scores stored in a series.
        """

        # Initialize the scores dictionary for each available pairs.
        scores = {pair_names: None for pair_names in pairs_names}

        # Calculate the scores for each pair for each method.
        if method == 'spearman rho':
            for pair_names in pairs_names:
                scores[pair_names] = self.spearman_rho(s1=stocks_universe[pair_names[0]],
                                                       s2=stocks_universe[pair_names[1]])

        if method == 'kendall tau':
            for pair_names in pairs_names:
                scores[pair_names] = self.kendall_tau(s1=stocks_universe[pair_names[0]],
                                                      s2=stocks_universe[pair_names[1]])

        if method == 'euc distance':
            for pair_names in pairs_names:
                scores[pair_names] = self.euc_distance(s1=stocks_universe[pair_names[0]],
                                                       s2=stocks_universe[pair_names[1]])

        # Convert the scores dictionary to a multi-indexed pd.Series.
        scores_series = pd.Series(data=scores, name='Score')
        scores_series.index.set_names(['Stock 1', 'Stock 2'], inplace=True)

        return scores_series

    @staticmethod
    def spearman_rho(s1: pd.Series, s2: pd.Series) -> float:
        """
        Calculating Spearman's rho for a pair of stocks.

        Complexity is O(N logN).

        :param s1: (pd.Series) Prices series for a stock.
        :param s2: (pd.Series) Prices series for a stock.
        :return: (float) Spearman's rho value.
        """

        result = ss.spearmanr(s1, s2)[0]

        return result

    @staticmethod
    def kendall_tau(s1: pd.Series, s2: pd.Series) -> float:
        """
        Calculating Kendall's tau for a pair of stocks.

        Complexity is O(N^2).

        :param s1: (pd.Series) Prices series for a stock.
        :param s2: (pd.Series) Prices series for a stock.
        :return: (float) Kendall's tau value.
        """

        result = ss.kendalltau(s1, s2)[0]

        return result

    @staticmethod
    def euc_distance(s1: pd.Series, s2: pd.Series) -> float:
        """
        Calculating the negative sum of euclidean distance (2-norm) for a pair of stocks on their normalized prices.

        Complexity is O(N). The result is multiplied by -1 because we want to keep the top results having the smallest
        distance in positive value (thus largest in negative value).

        :param s1: (pd.Series) Prices series for a stock.
        :param s2: (pd.Series) Prices series for a stock.
        :return: (float) Negative sum of Euclidean distance value.
        """

        # Convert to normalized prices.
        s1_normalized = s1 / s1[0]
        s2_normalized = s2 / s2[0]

        # Calculate the sum of Euclidean distance on normalized prices.
        result = np.linalg.norm(s1_normalized - s2_normalized)

        return result * (-1)
