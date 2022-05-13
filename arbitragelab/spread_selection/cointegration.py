# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
This module implements the ML based Pairs Selection Framework described by Simão Moraes
Sarmento and Nuno Horta in `"A Machine Learning based Pairs Trading Investment Strategy."
<http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.
"""
# pylint: disable=arguments-differ

from functools import reduce
import numpy as np
import pandas as pd

from arbitragelab.cointegration_approach import EngleGrangerPortfolio, get_half_life_of_mean_reversion
from arbitragelab.hedge_ratios import construct_spread
from arbitragelab.hedge_ratios import get_ols_hedge_ratio, get_tls_hedge_ratio, get_minimum_hl_hedge_ratio, \
    get_johansen_hedge_ratio, get_box_tiao_hedge_ratio
from arbitragelab.hedge_ratios.adf_optimal import get_adf_optimal_hedge_ratio
from arbitragelab.spread_selection.base import AbstractPairsSelector
from arbitragelab.util import segment
from arbitragelab.util.hurst import get_hurst_exponent


class CointegrationSpreadSelector(AbstractPairsSelector):
    """
    Implementation of the Proposed Pairs Selection Framework in the following paper:
    `"A Machine Learning based Pairs Trading Investment Strategy."
    <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.
    H&T team improved it to work not only with pairs, but also with spreads.
    """

    def __init__(self, prices_df: pd.DataFrame = None, baskets_to_filter: list = None):
        """
        Constructor.
        Sets up the price series needed for the next step.

        :param prices_df: (pd.DataFrame) Asset prices universe.
        :param baskets_to_filter: (list) List of tuples of tickers baskets to filter
            (can be pairs (AAA, BBB) or higher dimensions (AAA, BBB, CCC)).
        """

        self.prices_df = None
        self.baskets_to_filter = None
        self.spreads_dict = None  # spread ticker: spread series.
        self.hedge_ratio_information = None

        if prices_df is not None and baskets_to_filter is not None:
            self.set_prices(prices_df, baskets_to_filter)

        self.final_pairs = []
        self.selection_logs = pd.DataFrame(columns=['coint_t', 'p_value_99%', 'p_value_95%', 'p_value_90%',
                                                    'hurst_exponent', 'half_life', 'crossovers', 'hedge_ratio'])

        segment.track('CointegrationSpreadSelector')

    def set_prices(self, prices_df: pd.DataFrame, baskets_to_filter: list):
        """
        Sets up the price series needed for the next step.

        :param prices_df: (pd.DataFrame) Asset prices universe.
        :param baskets_to_filter: (list) List of tuples of tickers baskets to filter
            (can be pairs (AAA, BBB) or higher dimensions (AAA, BBB, CCC)).
        """

        self.prices_df = prices_df
        self.baskets_to_filter = baskets_to_filter
        self.hedge_ratio_information = pd.Series(index=['_'.join(x) for x in baskets_to_filter], dtype=object,
                                                 name='hedge_ratio')

    def construct_spreads(self, hedge_ratio_calculation: str) -> dict:
        """
        For `self.baskets_to_filter` construct spreads and log hedge ratio
        calculated based on `hedge_ratio_calculation`.

        :param hedge_ratio_calculation: (str) Defines how hedge ratio is calculated. Can be either 'OLS',
            'TLS' (Total Least Squares), 'min_half_life', 'min_adf', 'johansen', 'box_tiao'.
        :return: (dict) Dictionary of generated spreads (tuple: pd.Series).
        """

        spreads_dict = {}  # Spread ticker: pd.Series of constructed spread
        iteration = 0

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
                hedge_ratios, _, _, _, _ = get_minimum_hl_hedge_ratio(price_data=self.prices_df[list(bundle)],
                                                                      dependent_variable=bundle[0])
            elif hedge_ratio_calculation == 'min_adf':
                hedge_ratios, _, _, _, _ = get_adf_optimal_hedge_ratio(price_data=self.prices_df[list(bundle)],
                                                                       dependent_variable=bundle[0])
            elif hedge_ratio_calculation == 'johansen':
                hedge_ratios, _, _, _ = get_johansen_hedge_ratio(price_data=self.prices_df[list(bundle)],
                                                                 dependent_variable=bundle[0])
            elif hedge_ratio_calculation == 'box_tiao':
                hedge_ratios, _, _, _ = get_box_tiao_hedge_ratio(price_data=self.prices_df[list(bundle)],
                                                                 dependent_variable=bundle[0])
            else:
                raise ValueError('Unknown hedge ratio calculation parameter value.')

            spread = construct_spread(price_data=self.prices_df[list(bundle)], hedge_ratios=pd.Series(hedge_ratios))
            self.hedge_ratio_information['_'.join(bundle)] = hedge_ratios
            spreads_dict['_'.join(bundle)] = spread

            self._print_progress(iteration + 1, len(self.baskets_to_filter), prefix='Spread construction:',
                                 suffix='Complete')
            iteration += 1

        return spreads_dict

    def select_spreads(self, hedge_ratio_calculation: str = 'OLS',
                       adf_cutoff_threshold: float = 0.95,
                       hurst_exp_threshold: float = 0.5,
                       min_crossover_threshold: int = 12,
                       min_half_life: float = 365) -> list:
        """
        Apply cointegration selection rules (ADF, Hurst, Min SMA crossover, Min Half-Life) to filter-out pairs/baskets.

        Check to see if pairs comply with the criteria supplied in the paper: the pair being cointegrated,
        the Hurst exponent being <0.5, the spread moves within convenient periods and finally that the spread reverts
        to the mean with enough frequency.

        :param hedge_ratio_calculation: (str) Defines how hedge ratio is calculated. Can be either 'OLS',
            'TLS' (Total Least Squares), 'min_half_life', 'min_adf', 'johansen', 'box_tiao'.
        :param adf_cutoff_threshold: (float) ADF test threshold used to define if the spread is cointegrated.
            Can be 0.99, 0.95 or 0.9.
        :param hurst_exp_threshold: (float) Max Hurst threshold value.
        :param min_crossover_threshold: (int) Minimum amount of mean crossovers per year.
        :param min_half_life: (float) Minimum Half-Life of mean reversion value.
        :return: (list) Tuple list of final pairs.
        """

        self.spreads_dict = self.construct_spreads(hedge_ratio_calculation)

        # Generation statistics
        iteration = 0
        for spread_ticker, spread in self.spreads_dict.items():
            spread.name = spread_ticker
            self.generate_spread_statistics(spread_series=spread, log_info=True)
            self._print_progress(iteration + 1, len(self.spreads_dict), prefix='Statistics generation:',
                                 suffix='Complete')
            iteration += 1

        self.selection_logs['hedge_ratio'] = self.hedge_ratio_information.copy()

        passing_spreads = self.apply_filtering_rules(adf_cutoff_threshold, hurst_exp_threshold,
                                                     min_crossover_threshold, min_half_life)

        return passing_spreads


    def apply_filtering_rules(self,
                              adf_cutoff_threshold: float = 0.95,
                              hurst_exp_threshold: float = 0.5,
                              min_crossover_threshold: int = 12,
                              min_half_life: float = 365) -> list:
        """
        Apply cointegration selection rules (ADF, Hurst, Min SMA crossover, Min Half-Life) to filter-out pairs/baskets.

        Check to see if pairs comply with the criteria supplied in the paper: the pair being cointegrated,
        the Hurst exponent being <0.5, the spread moves within convenient periods and finally that the spread reverts
        to the mean with enough frequency.

        :param adf_cutoff_threshold: (float) ADF test threshold used to define if the spread is cointegrated.
            Can be 0.99, 0.95 or 0.9.
        :param hurst_exp_threshold: (float) Max Hurst threshold value.
        :param min_crossover_threshold: (int) Minimum amount of mean crossovers per year.
        :param min_half_life: (float) Minimum Half-Life of mean reversion value.
        :return: (list) Tuple list of final pairs.
        """

        cointegration_passing = self.selection_logs[self.selection_logs['coint_t']
                                                    <= self.selection_logs[
                                                        'p_value_{}%'.format(int(adf_cutoff_threshold * 100))]].index
        hurst_passing = self.selection_logs[self.selection_logs['hurst_exponent'] <= hurst_exp_threshold].index
        crossover_passing = self.selection_logs[self.selection_logs['crossovers'] >= min_crossover_threshold].index
        hl_passing = self.selection_logs[(self.selection_logs['half_life'] > 0) &
                                         (self.selection_logs['half_life'] <= min_half_life)].index

        return reduce(np.intersect1d, (cointegration_passing, hurst_passing, crossover_passing, hl_passing))

    def generate_spread_statistics(self, spread_series: pd.Series, log_info: bool = True) -> dict:
        """
        Check if the spread passes all filters.

        :param spread_series: (pd.Series) Spread values series.
        :param log_info: (bool) Flag indicating that information should be logged into `self.selection_logs`.
        :return: (dict) Dictionary with statistics.
        """

        # Selection Criterion One: First, it is imposed that pairs are cointegrated
        statistics = self._cointegration_statistics(spread_series)

        # Selection Criterion Two: Then, the spread’s Hurst exponent,
        # represented by H should be smaller than 0.5
        statistics['hurst_exponent'] = get_hurst_exponent(spread_series.values)

        # Selection Criterion Three: half life of mean reversion
        statistics['half_life'] = get_half_life_of_mean_reversion(spread_series)

        # Final Criterion: it is imposed that the spread crosses a mean at least n times
        statistics['crossovers'] = self._get_n_crossovers(spread_series)
        if log_info is True:
            self.selection_logs.loc[spread_series.name] = statistics

        return statistics

    @staticmethod
    def _get_n_crossovers(spread_series: pd.Series) -> int:
        """
        Get number of crossovers over mean.

        :param spread_series: (pd.Series) Spread values series.
        :return: (int) Number of crossovers.
        """

        centered_series = spread_series - spread_series.mean()
        cross_over_indices = np.where(np.diff(np.sign(centered_series)))[0]

        return len(cross_over_indices)

    @staticmethod
    def _cointegration_statistics(spread_series: pd.Series) -> dict:
        # pylint: disable=protected-access
        """
        This function calculates the Engle-Granger test statistics.

        :param spread_series: (pd.Series) Spread series.
        :return: (dict) Cointegration statistics.
        """

        eg_port = EngleGrangerPortfolio()
        eg_port._perform_eg_test(spread_series)
        statistic_value = eg_port.adf_statistics.loc['statistic_value'].iloc[0]
        p_value_99 = eg_port.adf_statistics.loc['99%'].iloc[0]
        p_value_95 = eg_port.adf_statistics.loc['95%'].iloc[0]
        p_value_90 = eg_port.adf_statistics.loc['90%'].iloc[0]

        return {'coint_t': statistic_value, 'p_value_99%': p_value_99,
                'p_value_95%': p_value_95, 'p_value_90%': p_value_90}
