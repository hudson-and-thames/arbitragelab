# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the ML based Pairs Selection Framework described by Simão Moraes
Sarmento and Nuno Horta in `"A Machine Learning based Pairs Trading Investment Strategy." <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.
"""

import numpy as np
import pandas as pd

from arbitragelab.cointegration_approach import EngleGrangerPortfolio, get_half_life_of_mean_reversion
from arbitragelab.hedge_ratios import construct_spread
from arbitragelab.hedge_ratios import get_ols_hedge_ratio, get_tls_hedge_ratio, get_minimum_hl_hedge_ratio, \
    get_johansen_hedge_ratio
from arbitragelab.hedge_ratios.adf_optimal import get_adf_optimal_hedge_ratio
from arbitragelab.spread_selection.base import AbstractPairsSelector
from arbitragelab.util import segment
from arbitragelab.util.hurst import get_hurst_exponent


# pylint: disable=arguments-differ
class CointegrationSpreadSelector(AbstractPairsSelector):
    """
    Implementation of the Proposed Pairs Selection Framework in the following paper:
    `"A Machine Learning based Pairs Trading Investment Strategy."
    <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.
    H&T team improved it to work not only with pairs, but also with spreads.
    """

    def __init__(self, prices_df: pd.DataFrame, baskets_to_filter: list):
        """
        Constructor.
        Sets up the price series needed for the next step.

        :param prices_df: (pd.DataFrame) Asset prices universe.
        :param baskets_to_filter: (list) List of tuples of tickers baskets to filter (can be pairs (AAA, BBB) or higher dimensions (AAA, BBB, CCC))
        """

        self.prices_df = prices_df
        self.baskets_to_filter = baskets_to_filter
        self.spreads_df = None

        self.hedge_ratio_information = pd.Series(index=['_'.join(x) for x in baskets_to_filter], dtype=object,
                                                 name='hedge_ratio')

        self.coint_pass_pairs = pd.Series({}, dtype=object)
        self.hurst_pass_pairs = pd.Series({}, dtype=object)
        self.hl_pass_pairs = pd.Series({}, dtype=object)

        self.final_pairs = []

        segment.track('CointegrationSpreadSelector')

    def construct_spreads(self, hedge_ratio_calculation: str) -> dict:
        """
        For `self.baskets_to_filter` construct spreads and log hedge ratio calculated based on `hedge_ratio_calculation`.

        :param hedge_ratio_calculation: (str) Defines how hedge ratio is calculated. Can be either 'OLS',
                                        'TLS' (Total Least Squares), 'min_half_life', 'min_adf' or 'johansen'.
        :return: (dict) Dictionary of generated spreads (tuple: pd.Series).
        """
        spreads_dict = {}  # Spread ticker: pd.Series of constructed spread.
        for bundle in self.baskets_to_filter:
            if hedge_ratio_calculation == 'OLS':
                hedge_ratios, _, _, _ = get_ols_hedge_ratio(price_data=self.prices_df[list(bundle)],
                                                            dependent_variable=bundle[0],
                                                            add_constant=False)
            elif hedge_ratio_calculation == 'TLS':
                hedge_ratios, _, _, _ = get_tls_hedge_ratio(price_data=self.prices_df[list(bundle)],
                                                            dependent_variable=bundle[0],
                                                            add_constant=False)
            elif hedge_ratio_calculation == 'min_half_life':
                hedge_ratios, _, _, _ = get_minimum_hl_hedge_ratio(price_data=self.prices_df[list(bundle)],
                                                                   dependent_variable=bundle[0])
            elif hedge_ratio_calculation == 'min_adf':
                hedge_ratios, _, _, _ = get_adf_optimal_hedge_ratio(price_data=self.prices_df[list(bundle)],
                                                                    dependent_variable=bundle[0])
            elif hedge_ratio_calculation == 'johansen':
                hedge_ratios, _, _, _ = get_johansen_hedge_ratio(price_data=self.prices_df[list(bundle)],
                                                                 dependent_variable=bundle[0])
            else:
                raise ValueError('Unknown hedge ratio calculation parameter value.')

            spread = construct_spread(price_data=self.prices_df[list(bundle)], hedge_ratios=pd.Series(hedge_ratios))
            self.hedge_ratio_information['_'.join(bundle)] = hedge_ratios
            spreads_dict['_'.join(bundle)] = spread
        return spreads_dict

    def select_spreads(self, hedge_ratio_calculation: str = 'OLS',
                       adf_cutoff_threshold: float = 0.95,
                       hurst_exp_threshold: float = 0.5,
                       min_crossover_threshold_per_year: int = 12,
                       min_half_life: float = 365,
                       test_period: str = '2Y') -> list:
        """
        Apply cointegration selection rules (ADF, Hurst, Min SMA crossover, Min Half-Life) to filter-out pairs/baskets.

        Check to see if pairs comply with the criteria supplied in the paper: the pair being cointegrated,
        the Hurst exponent being <0.5, the spread moves within convenient periods and finally that the spread reverts
        to the mean with enough frequency.

        :param hedge_ratio_calculation: (str) Defines how hedge ratio is calculated. Can be either 'OLS,
                                        'TLS' (Total Least Squares) or 'min_half_life'.
        :param adf_cutoff_threshold: (float) ADF test threshold used to define if the spread is cointegrated. Can be
                                             0.99, 0.95 or 0.9.
        :param hurst_exp_threshold: (float) Max Hurst threshold value.
        :param min_crossover_threshold_per_year: (int) Minimum amount of mean crossovers per year.
        :param min_half_life: (float) Minimum Half-Life of mean reversion value.
        :param test_period: (str) Time delta format, to be used as the time
            period where the mean crossovers will be calculated.
        :return: (list) Tuple list of final pairs.
        """

        generated_spreads = self.construct_spreads(hedge_ratio_calculation)

        # Selection Criterion One: First, it is imposed that pairs are cointegrated
        cointegration_results = self._outer_cointegration_loop(generated_spreads)

        passing_pairs = cointegration_results.loc[cointegration_results['coint_t']
                                                  <= cointegration_results[
                                                      'p_value_{}%'.format(int(adf_cutoff_threshold * 100))]]

        self.coint_pass_pairs = passing_pairs

        # Selection Criterion Two: Then, the spread’s Hurst exponent,
        # represented by H should be smaller than 0.5.

        hurst_pass_pairs = self._hurst_criterion(generated_spreads, passing_pairs.index.tolist(), hurst_exp_threshold)
        self.hurst_pass_pairs = hurst_pass_pairs

        # Selection Criterion Three & Four: Additionally, the half-life period, represented by hl, should
        # lay between one day and one year. Finally, it is imposed that the spread crosses a mean at least
        # 12 times per year.

        hl_pass_pairs, final_pairs = self._final_criterions(generated_spreads, hurst_pass_pairs.index.values,
                                                            test_period,
                                                            min_crossover_threshold_per_year,
                                                            min_half_life)

        self.hl_pass_pairs = hl_pass_pairs

        self.final_pairs = final_pairs

        return final_pairs.index.values

    def _outer_cointegration_loop(self, spreads_dict: dict) -> pd.DataFrame:
        # pylint: disable=protected-access
        """
        This function calculates the Engle-Granger test for each spread.

        :param spreads_dict: (dict) Dictionary of spread ticker: pd.Series(spread).
        :return: (pd.DataFrame) Cointegration statistics.
        """

        cointegration_results = []

        for iteration, spread_series in enumerate(spreads_dict.values()):
            eg_port = EngleGrangerPortfolio()

            constant = spread_series.mean()
            eg_port._perform_eg_test(spread_series)
            statistic_value = eg_port.adf_statistics.loc['statistic_value'].iloc[0]
            p_value_99 = eg_port.adf_statistics.loc['99%'].iloc[0]
            p_value_95 = eg_port.adf_statistics.loc['95%'].iloc[0]
            p_value_90 = eg_port.adf_statistics.loc['90%'].iloc[0]

            cointegration_results.append(
                [statistic_value, p_value_99, p_value_95, p_value_90,
                 constant])
            self._print_progress(iteration + 1, len(spreads_dict), prefix='Outer Cointegration Loop Progress:',
                                 suffix='Complete')

        return pd.DataFrame(cointegration_results,
                            index=list(spreads_dict.keys()),
                            columns=['coint_t', 'p_value_99%', 'p_value_95%', 'p_value_90%', 'constant'])

    @staticmethod
    def _hurst_criterion(spreads_dict: dict, spread_tickers: list, hurst_exp_threshold: int = 0.5) -> pd.Series:
        """
        This method will go through all the pairs given, calculate the needed spread and run
        the Hurst exponent test against each one.

        :param spreads_dict: (dict) Dictionary of spread ticker: pd.Series(spread).
        :param spread_tickers: (list) List of spread tickers to analyze.
        :param hurst_exp_threshold: (int) Max Hurst threshold value.
        :return: (pd.Series) Series of filtered spreads with Hurst exponent values.
        """

        hurst_pass_pairs_dict = {}

        for ticker in spread_tickers:
            spread_series = spreads_dict[ticker]
            hurst_exp = get_hurst_exponent(spread_series.values)

            if hurst_exp < hurst_exp_threshold:
                hurst_pass_pairs_dict[ticker] = hurst_exp
        hurst_pass_pairs_series = pd.Series(hurst_pass_pairs_dict)
        hurst_pass_pairs_series.name = 'hurst_exponent'
        return hurst_pass_pairs_series

    def _final_criterions(self, spreads_dict: dict,
                          pairs: list, test_period: str = '2Y',
                          min_crossover_threshold_per_year: int = 12,
                          min_half_life: float = 365) -> tuple:
        """
        This method consists of the final two criterions checks in the third stage of the proposed
        framework which involves; the calculation and check, of the half-life of the given pair spread
        and the amount of mean crossovers throughout a set period, in this case in a year.

        :param spreads_dict: (dict) Dictionary of spread ticker: pd.Series(spread).
        :param pairs: (list) List of asset name pairs to be analyzed.
        :param test_period: (str) Time delta format, to be used as the time
            period where the mean crossovers will be calculated.
        :param min_crossover_threshold_per_year: (int) Minimum amount of mean crossovers per year.
        :param min_half_life: (float) Minimum Half-Life of mean reversion value.
        :return: (pd.DataFrame, pd.DataFrame) The first is a DataFrame of pairs that passed the half-life
            test and the second is a DataFrame of final pairs and their mean crossover counts.
        """

        ou_results = self._outer_ou_loop(spreads_dict, molecule=pairs, test_period=test_period,
                                         cross_overs_per_delta=min_crossover_threshold_per_year)

        final_selection = ou_results[ou_results['hl'] > 1]

        final_selection = final_selection.loc[ou_results['hl'] < min_half_life]

        hl_pass_pairs = final_selection

        final_selection = final_selection.loc[ou_results['crossovers']]

        final_pairs = final_selection

        return hl_pass_pairs, final_pairs

    def _outer_ou_loop(self, spreads_dict: dict, test_period: str,
                       cross_overs_per_delta: int, molecule: list) -> pd.DataFrame:
        # pylint: disable=too-many-locals
        """
        This function gets mean reversion calculations (half-life and number of
        mean cross overs) for each pair in the molecule. Uses the linear regression
        method to get the half-life, which is much lighter computationally wise
        compared to the version using the OrnsteinUhlenbeck class.

        Note that when mean reversion is expected, lambda / StdErr has a negative value.
        This result implies that the expected duration of mean reversion lambda is
        inversely proportional to the absolute value of lambda.

        :param spreads_dict: (dict) Dictionary of spread ticker: pd.Series(spread).
        :param test_period: (str) Time delta format, to be used as the time
            period where the mean crossovers will be calculated.
        :param cross_overs_per_delta: (int) Crossovers per time delta selected.
        :param molecule: (list) Indices of pairs.
        :return: (pd.DataFrame) Mean Reversion statistics.
        """

        ou_results = []

        for iteration, ticker in enumerate(molecule):
            # Split the spread in two periods. The training data is used to
            # extract the long term mean of the spread. Then the mean is used
            # to find the the number of crossovers in the test period.
            spread = spreads_dict[ticker]
            test_df = spread.last(test_period)
            train_df = spread.iloc[: -len(test_df)]

            long_term_mean = np.mean(train_df)

            centered_series = test_df - long_term_mean

            # Set the spread to a mean of zero and classifies each value
            # based on their sign.
            cross_over_indices = np.where(np.diff(np.sign(centered_series)))[0]
            cross_overs_dates = spread.index[cross_over_indices]

            # Resample the mean crossovers series to yearly index and count
            # each occurence in each year.
            cross_overs_counts = cross_overs_dates.to_frame().resample('Y').count()
            cross_overs_counts.columns = ['counts']

            # Check that the number of crossovers are in accordance with the given selection
            # criteria.
            if cross_overs_per_delta is not None:
                cross_overs = len(cross_overs_counts[cross_overs_counts['counts'] >= cross_overs_per_delta]) > 0
            else:
                cross_overs = True

            # Append half-life and number of cross overs.
            half_life = get_half_life_of_mean_reversion(data=spread)
            ou_results.append([half_life, cross_overs])

            self._print_progress(iteration + 1, len(molecule), prefix='Outer OU Loop Progress:',
                                 suffix='Complete')

        return pd.DataFrame(ou_results, index=molecule, columns=['hl', 'crossovers'])

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

        info.append(("Total Pair Combinations", len(self.baskets_to_filter)))
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

        description_df = pd.concat(
            [self.coint_pass_pairs, self.hurst_pass_pairs, self.hl_pass_pairs, self.hedge_ratio_information], axis=1)
        description_df.dropna(inplace=True)
        description_df.reset_index(inplace=True)
        description_df.rename(columns={'level_0': 'leg_1', 'level_1': 'leg_2', 'hl': 'half_life'}, inplace=True)
        description_df.drop(['constant'], axis=1, errors='ignore', inplace=True)

        return description_df
