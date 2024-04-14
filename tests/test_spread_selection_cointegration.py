"""
Tests function of Pairs Selection module:
spread_selection/cointegration.py
"""
# pylint: disable=protected-access
# pylint: disable=invalid-name

import os
import unittest
import pandas as pd
import numpy as np

from arbitragelab.spread_selection.cointegration import CointegrationSpreadSelector
from arbitragelab.hedge_ratios import get_ols_hedge_ratio


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

    def test_criterion_selector_ols(self):
        """
        Verifies final user exposed criterion selection method with OLS hedge ratio calculation.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        coint_pairs = [('BA', 'CF'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)

        result = pairs_selector.select_spreads(hedge_ratio_calculation='OLS', adf_cutoff_threshold=0.9,
                                               hurst_exp_threshold=0.55, min_crossover_threshold=0)
        logs = pairs_selector.selection_logs.copy()
        # Assert that only 2 pairs passes cointegration tests and only 1 pair passes all tests.
        self.assertCountEqual(result, ['BA_CF', 'BKR_CE'])
        self.assertCountEqual(logs[logs['coint_t'] <= logs['p_value_90%']].index.to_list(),
                              ['_'.join(x) for x in coint_pairs])

    def test_criterion_selector_tls(self):
        """
        Verifies final user exposed criterion selection method with TLS hedge ration calculation.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE'), ('AES', 'CE', 'AMZN')]
        coint_pairs = [('BA', 'CF'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)

        result = pairs_selector.select_spreads(hedge_ratio_calculation='TLS', adf_cutoff_threshold=0.9,
                                               hurst_exp_threshold=0.55, min_crossover_threshold=0)
        logs = pairs_selector.selection_logs.copy()
        # Assert that only 2 pairs passes cointegration tests and only 1 pair passes all tests.
        self.assertCountEqual(result, ['BA_CF', 'BKR_CE'])
        self.assertCountEqual(logs[logs['coint_t'] <= logs['p_value_90%']].index.to_list(),
                              ['_'.join(x) for x in coint_pairs])

    def test_criterion_selector_min_adf(self):
        """
        Verifies final user exposed criterion selection method with optimal ADF hedge ratio calculation.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)

        result = pairs_selector.select_spreads(hedge_ratio_calculation='min_adf', adf_cutoff_threshold=0.9,
                                               hurst_exp_threshold=0.55, min_crossover_threshold=0)
        logs = pairs_selector.selection_logs.copy()
        # Assert that only 1 pair passes cointegration tests and only 1 pair passes all tests.
        self.assertCountEqual(result, ['BA_CF'])
        self.assertCountEqual(logs[logs['coint_t'] <= logs['p_value_90%']].index.to_list(), ['BA_CF'])

    def test_criterion_selector_min_hl(self):
        """
        Verifies final user exposed criterion selection method with minimum HL hedge ration calculation.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)

        result = pairs_selector.select_spreads(hedge_ratio_calculation='min_half_life', adf_cutoff_threshold=0.95,
                                               hurst_exp_threshold=0.5, min_crossover_threshold=0)
        logs = pairs_selector.selection_logs.copy()
        # Assert that only 1 pair passes cointegration tests and only 1 pair passes all tests.
        self.assertCountEqual(result, ['BA_CF'])
        self.assertCountEqual(logs[logs['coint_t'] <= logs['p_value_90%']].index.to_list(), ['BA_CF'])

        # Check value error raise for unknown hedge ratio input.
        with self.assertRaises(ValueError):
            pairs_selector.select_spreads(hedge_ratio_calculation='my_own_hedge', adf_cutoff_threshold=0.95,
                                          hurst_exp_threshold=0.55, min_crossover_threshold=8)

    def test_criterion_selector_box_tiao(self):
        """
        Verifies final user exposed criterion selection method with Box-Tiao ration calculation.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)

        result = pairs_selector.select_spreads(hedge_ratio_calculation='box_tiao', adf_cutoff_threshold=0.95,
                                               hurst_exp_threshold=0.52, min_crossover_threshold=50)
        logs = pairs_selector.selection_logs.copy()
        # Assert that only 1 pair passes cointegration tests and only 1 pair passes all tests.
        self.assertCountEqual(result, ['BA_CF'])
        self.assertCountEqual(logs[logs['coint_t'] <= logs['p_value_90%']].index.to_list(), ['BA_CF', 'BKR_CE'])

    def test_criterion_selector_johansen(self):
        """
        Verifies final user exposed criterion selection method with Box-Tiao ration calculation.
        """

        final_pairs = [('BA', 'CF')]
        other_pairs = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE')]
        input_pairs = final_pairs + other_pairs
        pairs_selector = CointegrationSpreadSelector(prices_df=self.data, baskets_to_filter=input_pairs)

        result = pairs_selector.select_spreads(hedge_ratio_calculation='johansen', adf_cutoff_threshold=0.95,
                                               hurst_exp_threshold=0.52, min_crossover_threshold=50)
        logs = pairs_selector.selection_logs.copy()
        # Assert that only 1 pair passes cointegration tests and only 1 pair passes all tests.
        self.assertCountEqual(result, ['BA_CF'])
        self.assertCountEqual(logs[logs['coint_t'] <= logs['p_value_90%']].index.to_list(), ['BA_CF'])

    def test_internal_functions(self):
        """
        Tests `generate_spread_statistics`, 'apply_filtering_rules` function.
        """

        rs = np.random.RandomState(42)
        X_returns = rs.normal(0, 1, 101)
        X = pd.Series(np.cumsum(X_returns), name='X') + 50

        noise = rs.normal(0, 1, 101)
        Y = 5 * X + noise
        Y.name = 'Y'

        cointegrated_series = pd.concat([X, Y], axis=1)
        _, _, _, residuals = get_ols_hedge_ratio(price_data=cointegrated_series, dependent_variable='Y')
        pairs_selector = CointegrationSpreadSelector(prices_df=None, baskets_to_filter=None)
        stats = pairs_selector.generate_spread_statistics(residuals, log_info=True)

        # Test spread statistics.
        self.assertAlmostEqual(stats['coint_t'], -11, delta=1e-2)
        self.assertAlmostEqual(stats['half_life'], 0.62, delta=1e-2)
        self.assertAlmostEqual(stats['hurst_exponent'], -0.111, delta=1e-2)
        self.assertAlmostEqual(stats['crossovers'], 55)

        # Test filtering function.
        result = pairs_selector.apply_filtering_rules(adf_cutoff_threshold=0.99, hurst_exp_threshold=0.5)
        self.assertEqual(len(result), 1)
        result_2 = pairs_selector.apply_filtering_rules(min_half_life=0.2)  # Too strict.
        self.assertEqual(len(result_2), 0)
        result_3 = pairs_selector.apply_filtering_rules(min_crossover_threshold=100)  # Too strict.
        self.assertEqual(len(result_3), 0)
