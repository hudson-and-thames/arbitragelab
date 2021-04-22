# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests function of Pairs Selection module:
pairs_selection/cointegration.py
"""
import os
import unittest
import pandas as pd
import numpy as np
import matplotlib

from arbitragelab.pairs_selection import CointegrationPairsSelector


# pylint: disable=protected-access


class TestCointegrationSelector(unittest.TestCase):
    """
    Tests CointegrationPairsSelector class.
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
        idx = [('A', 'AVB'), ('ABMD', 'AZO')]
        pairs_selector = CointegrationPairsSelector(prices_df=self.data, pairs_to_filter=idx)
        # Setup needed information to validate the hurst criterion return.
        hedge_ratios = [0.832406370860649, 70]

        input_pairs = pd.DataFrame(data=hedge_ratios, index=idx)
        input_pairs.columns = ['hedge_ratio']

        result = pairs_selector._hurst_criterion(input_pairs)
        hurst_pp = pd.Series(result[1].index)
        pd.testing.assert_series_equal(pd.Series(idx), hurst_pp)

        # Test the hurst criterion with invalid input data.
        with self.assertRaises(Exception):
            pairs_selector._hurst_criterion([])

    def test_final_criterions(self):
        """
        Verifies private final criterions processing method.
        """

        hedge_ratios = [0.832406370860649, 70]
        idx = [('A', 'AVB'), ('ABMD', 'AZO')]
        pairs_selector = CointegrationPairsSelector(prices_df=self.data, pairs_to_filter=idx)
        input_pairs = pd.DataFrame(data=hedge_ratios, index=idx)
        input_pairs.columns = ['hedge_ratio']

        # Generate the inputs needed for the final criterions method test.
        spreads_df, hurst_pass_pairs = pairs_selector._hurst_criterion(input_pairs)

        hl_pairs, final_pairs = pairs_selector._final_criterions(
            spreads_df, hurst_pass_pairs.index.values
        )

        hl_pairs_sr = pd.Series(hl_pairs.index)
        final_pairs_sr = pd.Series(final_pairs.index)

        # Check that the first pair passes the Half Life Test.
        pd.testing.assert_series_equal(pd.Series([idx[0]]), hl_pairs_sr)

        # Check that 1 pair pass through to the final list.
        pd.testing.assert_series_equal(pd.Series([('A', 'AVB')], dtype=object), final_pairs_sr)

        # Test final criterions method using invalid data.
        with self.assertRaises(Exception):
            pairs_selector._final_criterions([], [])

    def test_criterion_selector_ols(self):
        """
        Verifies final user exposed criterion selection method with OLS hedge ratio calculation.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        coint_pairs = [('BA', 'CF'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationPairsSelector(prices_df=self.data, pairs_to_filter=input_pairs)

        result = pairs_selector._criterion_selection(input_pairs, adf_cutoff_threshold=0.9,
                                                     min_crossover_threshold_per_year=0,
                                                     hedge_ratio_calculation='OLS')
        result = pd.Series(result)

        coint_pp = pairs_selector.coint_pass_pairs.index
        coint_pp = pd.Series(coint_pp)

        # Assert that only 2 pairs passes cointegration tests and only 1 pair passes all tests.
        pd.testing.assert_series_equal(pd.Series(coint_pairs), coint_pp)
        pd.testing.assert_series_equal(pd.Series(final_pairs), result)

    def test_criterion_selector_tls(self):
        """
        Verifies final user exposed criterion selection method with TLS hedge ration calculation.
        """

        final_pairs = [('BA', 'CF'), ('BKR', 'CE')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY')]
        coint_pairs = [('BA', 'CF'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationPairsSelector(prices_df=self.data, pairs_to_filter=input_pairs)

        result = pairs_selector._criterion_selection(input_pairs, adf_cutoff_threshold=0.9,
                                                     min_crossover_threshold_per_year=None,
                                                     hurst_exp_threshold=0.55,
                                                     hedge_ratio_calculation='TLS')
        result = pd.Series(result)

        coint_pp = pairs_selector.coint_pass_pairs.index
        coint_pp = pd.Series(coint_pp)

        # Assert that only 2 pairs passes cointegration tests and only 2 pairs passes all tests (no crossover).
        pd.testing.assert_series_equal(pd.Series(coint_pairs), coint_pp)
        pd.testing.assert_series_equal(pd.Series(final_pairs), result)

    def test_criterion_selector_min_hl(self):
        """
        Verifies final user exposed criterion selection method with minimum HL hedge ration calculation.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        coint_pairs = [('BA', 'CF')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationPairsSelector(prices_df=self.data, pairs_to_filter=input_pairs)

        result = pairs_selector._criterion_selection(input_pairs, adf_cutoff_threshold=0.95,
                                                     min_crossover_threshold_per_year=8,
                                                     hurst_exp_threshold=0.55,
                                                     hedge_ratio_calculation='min_half_life')
        # Check value error raise for unknown hedge ratio input.
        with self.assertRaises(ValueError):
            pairs_selector._criterion_selection(input_pairs, adf_cutoff_threshold=0.95,
                                                min_crossover_threshold_per_year=8,
                                                hurst_exp_threshold=0.55,
                                                hedge_ratio_calculation='johansen')
        result = pd.Series(result)

        coint_pp = pairs_selector.coint_pass_pairs.index
        coint_pp = pd.Series(coint_pp)

        # Assert that only 2 pairs passes cointegration tests and only 2 pairs passes all tests (no crossover).
        pd.testing.assert_series_equal(pd.Series(coint_pairs), coint_pp)
        pd.testing.assert_series_equal(pd.Series(final_pairs), result)

    def test_unsupervised_candidate_pair_selector(self):
        """
        Tests the parent candidate pair selection method.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationPairsSelector(prices_df=self.data, pairs_to_filter=input_pairs)

        # Tests pair selector with invalid data seed clustering data.
        with self.assertRaises(Exception):
            pairs_selector.select_pairs()

        self.assertTrue(
            type(pairs_selector.select_pairs(adf_cutoff_threshold=0.9,
                                             min_crossover_threshold_per_year=4)), list)

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
        pairs_selector = CointegrationPairsSelector(prices_df=self.data, pairs_to_filter=input_pairs)

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
        pairs_selector = CointegrationPairsSelector(prices_df=self.data, pairs_to_filter=input_pairs)

        # Test the final pairs plotting method with no information.
        with self.assertRaises(Exception):
            pairs_selector.plot_selected_pairs()

        # Test single pair plot return object.
        singlepair_pyplot_obj = pairs_selector.plot_single_pair(('AJG', 'ABMD'))
        self.assertTrue(issubclass(type(singlepair_pyplot_obj), matplotlib.axes.SubplotBase))
