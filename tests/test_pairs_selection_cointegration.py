# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests function of Pairs Selection module:
spread_selection/cointegration.py
"""
# pylint: disable=protected-access

import os
import unittest
import pandas as pd
import numpy as np
import matplotlib

from arbitragelab.spread_selection.cointegration import CointegrationSpreadSelector


class TestCointegrationSelector(unittest.TestCase):
    """
    Tests CointegrationSpreadSelector class.
    """

    def setUp(self):
        """
        Loads price universe and instantiates the pairs selection class.
        """

        np.random.seed(0)

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/sp100_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.data.dropna(inplace=True)

    def test_hurst_criterion(self):
        """
        Verifies private hurst processing method.
        """

        idx = [('A', 'AVB'), ('ABMD', 'AZO'), ('ABMD', 'AZO', 'AMZN')]
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=idx)
        spreads_dict = pairs_selector.construct_spreads('johansen')

        result = pairs_selector._hurst_criterion(spreads_dict, ['_'.join(x) for x in idx])
        self.assertCountEqual(result.index, ['_'.join(x) for x in idx])

    def test_final_criterions(self):
        """
        Verifies private final criterions processing method.
        """

        idx = [('A', 'AVB'), ('ABMD', 'AZO'), ('BA', 'CF')]
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=idx)

        # Generate the inputs needed for the final criterions method test.
        spreads_dict = pairs_selector.construct_spreads('OLS')
        result = pairs_selector._hurst_criterion(spreads_dict, ['_'.join(x) for x in idx])

        hl_pairs, final_pairs = pairs_selector._final_criterions(
            spreads_dict, result.index.values, min_crossover_threshold_per_year=0
        )

        # Check that the third pair passes the Half Life Test.
        self.assertCountEqual(hl_pairs.index, ['_'.join(x) for x in [idx[2]]])
        # Check that 3rd pair pass through to the final list.
        self.assertCountEqual(final_pairs.index, ['_'.join(x) for x in [idx[2]]])

    def test_criterion_selector_min_adf(self):
        """
        Verifies final user exposed criterion selection method with optimal ADF hedge ratio calculation.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        coint_pairs = [('BA', 'CF')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)

        result = pairs_selector.select_spreads(hedge_ratio_calculation='min_adf', adf_cutoff_threshold=0.9,
                                               min_crossover_threshold_per_year=None)

        # Assert that only 2 pairs passes cointegration tests and only 1 pair passes all tests.
        self.assertCountEqual(result, ['BA_CF'])
        self.assertCountEqual(pairs_selector.coint_pass_pairs.index, ['_'.join(x) for x in coint_pairs])

    def test_criterion_selector_tls(self):
        """
        Verifies final user exposed criterion selection method with TLS hedge ration calculation.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        coint_pairs = [('BA', 'CF'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)

        result = pairs_selector.select_spreads(hedge_ratio_calculation='TLS', adf_cutoff_threshold=0.9,
                                               hurst_exp_threshold=0.55, min_crossover_threshold_per_year=None)
        # Assert that only 2 pairs passes cointegration tests and only 1 pair passes all tests.
        self.assertCountEqual(result, ['BA_CF', 'BKR_CE'])
        self.assertCountEqual(pairs_selector.coint_pass_pairs.index, ['_'.join(x) for x in coint_pairs])

    def test_criterion_selector_min_hl(self):
        """
        Verifies final user exposed criterion selection method with minimum HL hedge ration calculation.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)

        result = pairs_selector.select_spreads(hedge_ratio_calculation='min_half_life', adf_cutoff_threshold=0.95,
                                               hurst_exp_threshold=0.55, min_crossover_threshold_per_year=8)

        # Assert that only 2 pairs passes cointegration tests and only 1 pair passes all tests.
        self.assertCountEqual(result, ['BA_CF'])
        self.assertCountEqual(pairs_selector.coint_pass_pairs.index, ['BA_CF'])

        # Check value error raise for unknown hedge ratio input.
        with self.assertRaises(ValueError):
            pairs_selector.select_spreads(hedge_ratio_calculation='johansen', adf_cutoff_threshold=0.95,
                                          hurst_exp_threshold=0.55, min_crossover_threshold_per_year=8)

    def test_unsupervised_candidate_pair_selector(self):
        """
        Tests the parent candidate pair selection method.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)

        _ = pairs_selector.select_spreads(hedge_ratio_calculation='OLS', adf_cutoff_threshold=0.9,
                                          min_crossover_threshold_per_year=None)

        self.assertTrue(
            type(pairs_selector.select_spreads(adf_cutoff_threshold=0.9, min_crossover_threshold_per_year=4)), list)

        final_pairs = pd.DataFrame(index=[('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')])
        pairs_selector.final_pairs = final_pairs
        selected_pairs_return = pairs_selector.plot_selected_pairs()

        # Check if returned plot object is a list of Axes objects.
        self.assertTrue(type(selected_pairs_return), list)

        with self.assertRaises(Exception):
            pairs_list = list((('F', 'V'),) * 45)
            final_pairs = pd.DataFrame(index=pairs_list)
            pairs_selector.final_pairs = final_pairs
            pairs_selector.plot_selected_pairs()

    def test_description_methods(self):
        """
        Tests the various pair description methods.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)
        _ = pairs_selector.select_spreads(hedge_ratio_calculation='OLS', adf_cutoff_threshold=0.9,
                                          min_crossover_threshold_per_year=None)

        # Test return of the describe method.
        intro_descr = pairs_selector.describe()
        self.assertEqual(type(intro_descr), pd.DataFrame)

        # Test return of the extended describe method.
        extended_descr = pairs_selector.describe_extra()
        self.assertEqual(type(extended_descr), pd.DataFrame)

        # Test return of the sectoral based method with empty sector info dataframe input.
        empty_sectoral_df = pd.DataFrame(columns=['ticker', 'sector', 'industry'])
        empty_sectoral_descr = pairs_selector.describe_pairs_sectoral_info(['AJG'], ['ICE'], empty_sectoral_df)
        self.assertEqual(type(empty_sectoral_descr), pd.DataFrame)

        sector_info = pd.DataFrame(data=[
            ('AJG', 'sector', 'industry'),
            ('ICE', 'sector', 'industry')
        ])
        sector_info.columns = ['ticker', 'sector', 'industry']
        full_sectoral_descr = pairs_selector.describe_pairs_sectoral_info(['AJG'], ['ICE'], sector_info)

        # Test return of the sectoral based method with full sector info dataframe input.
        self.assertEqual(type(full_sectoral_descr), pd.DataFrame)

    def test_plotting_methods(self):
        """
        Tests all plotting methods.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)

        # Test the final pairs plotting method with no information.
        with self.assertRaises(Exception):
            pairs_selector.plot_selected_pairs()

        # Test single pair plot return object.
        singlepair_pyplot_obj = pairs_selector.plot_single_pair(('AJG', 'ABMD'))
        self.assertTrue(issubclass(type(singlepair_pyplot_obj), matplotlib.axes.SubplotBase))
