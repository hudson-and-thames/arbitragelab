# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the ML based Pairs Selection Framework described by Simão Moraes
Sarmento and Nuno Horta in `"A Machine Learning based Pairs Trading Investment Strategy." <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.
"""

import pandas as pd

from arbitragelab.ml_approach.stat_arb_utils import _outer_cointegration_loop, _outer_ou_loop
from arbitragelab.util import devadarsh
from arbitragelab.util.hurst import get_hurst_exponent
from arbitragelab.pairs_selection.base import AbstractPairsSelector


# pylint: disable=arguments-differ
class CointegrationPairsSelector(AbstractPairsSelector):
    """
    Implementation of the Proposed Pairs Selection Framework in the following paper:
    `"A Machine Learning based Pairs Trading Investment Strategy."
    <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.
    """

    def __init__(self, prices_df: pd.DataFrame, pairs_to_filter: tuple):
        """
        Constructor.
        Sets up the price series needed for the next step.

        :param prices_df: (pd.DataFrame) Asset prices universe.
        :param pairs_to_filter: (tuple) Tuple of pairs which need to be filtered.
        """

        self.prices_df = prices_df
        self.pairs_to_filter = pairs_to_filter
        self.spreads_df = None

        self.coint_pass_pairs = pd.Series({}, dtype=object)
        self.hurst_pass_pairs = pd.Series({}, dtype=object)
        self.hl_pass_pairs = pd.Series({}, dtype=object)

        self.final_pairs = []

        devadarsh.track('CointegrationPairsSelector')

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
                          min_crossover_threshold_per_year: int = 12,
                          min_half_life: float = 365) -> tuple:
        """
        This method consists of the final two criterions checks in the third stage of the proposed
        framework which involves; the calculation and check, of the half-life of the given pair spread
        and the amount of mean crossovers throughout a set period, in this case in a year.

        :param spreads_df: (pd.DataFrame) Hedge ratio adjusted spreads DataFrame.
        :param pairs: (list) List of asset name pairs to be analyzed.
        :param test_period: (str) Time delta format, to be used as the time
            period where the mean crossovers will be calculated.
        :param min_crossover_threshold_per_year: (int) Minimum amount of mean crossovers per year.
        :param min_half_life: (float) Minimum Half-Life of mean reversion value.
        :return: (pd.DataFrame, pd.DataFrame) The first is a DataFrame of pairs that passed the half-life
            test and the second is a DataFrame of final pairs and their mean crossover counts.
        """

        if len(pairs) == 0:
            raise Exception("No pairs have been found!")

        ou_results = _outer_ou_loop(spreads_df, molecule=pairs, test_period=test_period,
                                    cross_overs_per_delta=min_crossover_threshold_per_year)

        final_selection = ou_results[ou_results['hl'] > 1]

        final_selection = final_selection.loc[ou_results['hl'] < min_half_life]

        hl_pass_pairs = final_selection

        final_selection = final_selection.loc[ou_results['crossovers']]

        final_pairs = final_selection

        return hl_pass_pairs, final_pairs

    def select_pairs(self, hedge_ratio_calculation: str = 'OLS',
                     adf_cutoff_threshold: float = 0.95,
                     hurst_exp_threshold: int = 0.5,
                     min_crossover_threshold_per_year: int = 12,
                     min_half_life: float = 365,
                     test_period: str = '2Y') -> list:
        """
        Apply cointegration selection rules (ADF, Hurst, Min SMA crossover, Min Half-Life) to filter-out pairs.

        Check to see if pairs comply with the criteria supplied in the paper: the pair being cointegrated,
        the Hurst exponent being <0.5, the spread moves within convenient periods and finally that the spread reverts
        to the mean with enough frequency.

        :param hedge_ratio_calculation: (str) Defines how hedge ratio is calculated. Can be either 'OLS,
                                        'TLS' (Total Least Squares) or 'min_half_life'.
        :param adf_cutoff_threshold: (float) ADF test threshold used to define if the spread is cointegrated. Can be
                                             0.99, 0.95 or 0.9.
        :param hurst_exp_threshold: (int) Max Hurst threshold value.
        :param min_crossover_threshold_per_year: (int) Minimum amount of mean crossovers per year.
        :param min_half_life: (float) Minimum Half-Life of mean reversion value.
        :param test_period: (str) Time delta format, to be used as the time
            period where the mean crossovers will be calculated.
        :return: (list) Tuple list of final pairs.
        """

        return self._criterion_selection(self.pairs_to_filter, hedge_ratio_calculation,
                                         adf_cutoff_threshold, hurst_exp_threshold, min_crossover_threshold_per_year,
                                         min_half_life, test_period)

    def _criterion_selection(self, combinations: list, hedge_ratio_calculation: str = 'OLS',
                             adf_cutoff_threshold: float = 0.95, hurst_exp_threshold: int = 0.5,
                             min_crossover_threshold_per_year: int = 12, min_half_life: float = 365,
                             test_period: str = '2Y') -> list:
        """
        Apply selection criterions.

        :param combinations: (list) List of asset pairs.
        :param hedge_ratio_calculation: (str) Defines how hedge ratio is calculated. Can be either 'OLS,
                                        'TLS' (Total Least Squares) or 'min_half_life'.
        :param adf_cutoff_threshold: (float) ADF test threshold used to define if the spread is cointegrated. Can be
                                             0.99, 0.95 or 0.9.
        :param hurst_exp_threshold: (int) Max Hurst threshold value.
        :param min_crossover_threshold_per_year: (int) Minimum amount of mean crossovers per year.
        :param min_half_life: (float) Minimum Half-Life of mean reversion value.
        :param test_period: (str) Time delta format, to be used as the time
            period where the mean crossovers will be calculated.
        :return: (list) Tuple list of final pairs.
        """

        # Selection Criterion One: First, it is imposed that pairs are cointegrated

        cointegration_results = _outer_cointegration_loop(
            self.prices_df, combinations, hedge_ratio_calculation=hedge_ratio_calculation)

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
                                                            min_crossover_threshold_per_year,
                                                            min_half_life)

        self.hl_pass_pairs = hl_pass_pairs

        self.final_pairs = final_pairs

        return final_pairs.index.values

    def describe(self) -> pd.DataFrame:
        """
        Returns the Pairs Selector Summary statistics.

        The following statistics are included - total possible pair combinations,
        the number of pairs that passed the cointegration threshold, the number of pairs that passed the
        Hurst exponent threshold, the number of pairs that passed the half-life threshold and the number
        of final set of pairs.

        :return: (pd.DataFrame) Dataframe of summary statistics.
        """

        no_hurstpair = len(self.hurst_pass_pairs)
        no_hlpair = len(self.hl_pass_pairs)

        info = []

        info.append(("Total Pair Combinations", len(self.pairs_to_filter)))
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
