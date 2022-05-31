# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Unit tests for copula_strategy_basic, and additional features of copula_generate.
"""
# pylint: disable = invalid-name, protected-access, too-many-locals, unsubscriptable-object, too-many-statements

import os
import unittest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arbitragelab.trading.copula_approach import BasicCopulaTradingRule
from arbitragelab.copula_approach import find_marginal_cdf


class TestBasicCopulaStrategy(unittest.TestCase):
    """
    Test the BasicCopulaStrategy class.
    """

    def setUp(self):
        """
        Get the correct directory and data.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + "/test_data/BKD_ESC_2009_2011.csv"
        self.stocks = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_marginal_cdf(self):
        """
        Testing the find_marginal_cdf() method.
        """

        # Create sample data frame and compute the percentile
        data = {'col1': [0, 1, 2, 3, 4, 5], 'col2': [0, 2, 4, 6, np.nan, 10], 'col3': [np.nan, 2, 4, 6, 8, 10]}
        quantile_dict = {k: find_marginal_cdf(v) for k, v in data.items()}
        # Expected result
        expected = {'col1': [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1],
                    'col2': [1 / 5, 2 / 5, 3 / 5, 4 / 5, np.nan, 5/5],
                    'col3': [np.nan, 1 / 5, 2 / 5, 3 / 5, 4 / 5, 5/5]}
        for col, values in data.items():
            np.testing.assert_array_almost_equal(quantile_dict[col](values), expected[col], decimal=4)

        # Checking the cdfs
        test_input = [-100, -1, 1.5, 2, 3, 10, np.nan]
        expec_qt1 = [0.1667, 0.3333, 0.5, 0.66667, 0.83333, 1, np.nan]
        np.testing.assert_array_almost_equal(expec_qt1, find_marginal_cdf(test_input)(test_input), decimal=4)

    def test_exit_trigger_or(self):
        """
        Testing the exit trigger under 'or' logic.
        """

        cop_trading = BasicCopulaTradingRule(exit_rule='or', exit_probabilities=(0.5, 0.5))
        cop_trading.current_probabilities = (0.4, 0.6)
        cop_trading.prev_probabilities = (0.6, 0.6)
        res = cop_trading._check_who_exits()

        # s1
        # s1 x-ing down, no one exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.4, 0.6])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([1, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing up, no one exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([1, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing down, s1 exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.4, 0.6])
        who_exits = pd.Series([1, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([1, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing up, s1 exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([1, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([1, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing down, s2 exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.4, 0.6])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing up, s2 exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2
        # s2 x-ing down, no one exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 1])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2 x-ing up, no one exited before
        pre_condi_probs = pd.Series([0.6, 0.4])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 1])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2 x-ing down, s1 exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([1, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2 x-ing up, s1 exited before
        pre_condi_probs = pd.Series([0.6, 0.4])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([1, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2 x-ing down, s2 exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 1])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2 x-ing up, s2 exited before
        pre_condi_probs = pd.Series([0.6, 0.4])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 1])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 and s2
        # s1 x-ing up, s2 xing down, no one exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing up, s2 xing down, s1 exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([1, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing up, s2 xing down, s2 exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing down, s2 xing up, no one exited before
        pre_condi_probs = pd.Series([0.6, 0.4])
        condi_probs = pd.Series([0.4, 0.6])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # No trigger
        pre_condi_probs = pd.Series([0.6, 0.55])
        condi_probs = pd.Series([0.65, 0.6])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 0
        expected_who_exits = pd.Series([0, 1])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

    def test_exit_trigger_and(self):
        """
        Testing the exit trigger under 'and' logic.
        """

        BCS = BasicCopulaStrategy()
        exit_rule = 'and'
        exit_thresholds = (0.5, 0.5)

        # s1
        # s1 x-ing down, no one exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.4, 0.6])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 0
        expected_who_exits = pd.Series([1, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing up, no one exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 0
        expected_who_exits = pd.Series([1, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing down, s1 exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.4, 0.6])
        who_exits = pd.Series([1, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 0
        expected_who_exits = pd.Series([1, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing up, s1 exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([1, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 0
        expected_who_exits = pd.Series([1, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing down, s2 exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.4, 0.6])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing up, s2 exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2
        # s2 x-ing down, no one exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 0
        expected_who_exits = pd.Series([0, 1])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2 x-ing up, no one exited before
        pre_condi_probs = pd.Series([0.6, 0.4])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 0
        expected_who_exits = pd.Series([0, 1])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2 x-ing down, s1 exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([1, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2 x-ing up, s1 exited before
        pre_condi_probs = pd.Series([0.6, 0.4])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([1, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2 x-ing down, s2 exited before
        pre_condi_probs = pd.Series([0.6, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 0
        expected_who_exits = pd.Series([0, 1])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s2 x-ing up, s2 exited before
        pre_condi_probs = pd.Series([0.6, 0.4])
        condi_probs = pd.Series([0.6, 0.6])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 0
        expected_who_exits = pd.Series([0, 1])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 and s2
        # s1 x-ing up, s2 xing down, no one exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing up, s2 xing down, s1 exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([1, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing up, s2 xing down, s2 exited before
        pre_condi_probs = pd.Series([0.4, 0.6])
        condi_probs = pd.Series([0.6, 0.4])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # s1 x-ing down, s2 xing up, no one exited before
        pre_condi_probs = pd.Series([0.6, 0.4])
        condi_probs = pd.Series([0.4, 0.6])
        who_exits = pd.Series([0, 0])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 1
        expected_who_exits = pd.Series([0, 0])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # No trigger
        pre_condi_probs = pd.Series([0.6, 0.55])
        condi_probs = pd.Series([0.65, 0.6])
        who_exits = pd.Series([0, 1])
        signal, who_exits = BCS._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits, exit_thresholds)

        expected_signal = 0
        expected_who_exits = pd.Series([0, 1])
        self.assertEqual(signal, expected_signal)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

    def test_get_cur_position(self):
        """
        Testing position generation from get_cur_position method.

        Testing open position generation here mostly since the exit has been thoroughly tested. We use 'and' logic
        for exit signal.
        """

        BCS = BasicCopulaStrategy()
        exit_rule = 'and'

        # Testing long position
        # Long when no exiting signal
        pre_condi_probs = pd.Series([0.45, 0.55])
        condi_probs = pd.Series([0.01, 0.99])
        who_exits = pd.Series([0, 0])
        # Previously no position
        pre_pos = 0
        cur_pos, _ = BCS.get_cur_position(condi_probs, pre_condi_probs, pre_pos, exit_rule, who_exits)
        expected_pos = 1
        self.assertEqual(cur_pos, expected_pos)
        # Previously long
        pre_pos = 1
        cur_pos, _ = BCS.get_cur_position(condi_probs, pre_condi_probs, pre_pos, exit_rule, who_exits)
        expected_pos = 1
        self.assertEqual(cur_pos, expected_pos)
        # Previously short
        pre_pos = -1
        cur_pos, _ = BCS.get_cur_position(condi_probs, pre_condi_probs, pre_pos, exit_rule, who_exits)
        expected_pos = 1
        self.assertEqual(cur_pos, expected_pos)

        # Long when there is exiting signal
        pre_condi_probs = pd.Series([0.55, 0.45])
        condi_probs = pd.Series([0.01, 0.99])
        who_exits = pd.Series([0, 1])
        pre_pos = -1
        expected_pos = 0  # Expecting no position
        expected_who_exits = pd.Series([0, 0])
        cur_pos, who_exits = BCS.get_cur_position(condi_probs, pre_condi_probs, pre_pos, exit_rule, who_exits)
        self.assertEqual(cur_pos, expected_pos)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # Testing short position
        # Short when no exiting signal
        pre_condi_probs = pd.Series([0.55, 0.45])
        condi_probs = pd.Series([0.99, 0.01])
        who_exits = pd.Series([0, 0])
        # Previously no position
        pre_pos = 0
        cur_pos, _ = BCS.get_cur_position(condi_probs, pre_condi_probs, pre_pos, exit_rule, who_exits)
        expected_pos = -1
        self.assertEqual(cur_pos, expected_pos)
        # Previously long
        pre_pos = 1
        cur_pos, _ = BCS.get_cur_position(condi_probs, pre_condi_probs, pre_pos, exit_rule, who_exits)
        expected_pos = -1
        self.assertEqual(cur_pos, expected_pos)
        # Previously short
        pre_pos = -1
        cur_pos, _ = BCS.get_cur_position(condi_probs, pre_condi_probs, pre_pos, exit_rule, who_exits)
        expected_pos = -1
        self.assertEqual(cur_pos, expected_pos)

        # Short when there is exiting signal
        pre_condi_probs = pd.Series([0.45, 0.55])
        condi_probs = pd.Series([0.99, 0.01])
        who_exits = pd.Series([0, 1])
        pre_pos = 1
        expected_pos = 0  # Expecting no position
        expected_who_exits = pd.Series([0, 0])
        cur_pos, who_exits = BCS.get_cur_position(condi_probs, pre_condi_probs, pre_pos, exit_rule, who_exits)
        self.assertEqual(cur_pos, expected_pos)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

        # Check holding current position when there is no signal
        pre_condi_probs = pd.Series([0.45, 0.55])
        condi_probs = pd.Series([0.44, 0.76])
        who_exits = pd.Series([0, 1])
        # Previously long
        pre_pos = 1
        expected_pos = 1  # Expecting long
        expected_who_exits = pd.Series([0, 1])
        cur_pos, who_exits = BCS.get_cur_position(condi_probs, pre_condi_probs, pre_pos, exit_rule, who_exits)
        self.assertEqual(cur_pos, expected_pos)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)
        # Previously short
        pre_pos = -1
        expected_pos = -1  # Expecting short
        expected_who_exits = pd.Series([0, 1])
        cur_pos, who_exits = BCS.get_cur_position(condi_probs, pre_condi_probs, pre_pos, exit_rule, who_exits)
        self.assertEqual(cur_pos, expected_pos)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)
        # Previously no position
        pre_pos = 0
        expected_pos = 0  # Expecting no position
        expected_who_exits = pd.Series([0, 1])
        cur_pos, who_exits = BCS.get_cur_position(condi_probs, pre_condi_probs, pre_pos, exit_rule, who_exits)
        self.assertEqual(cur_pos, expected_pos)
        pd.testing.assert_series_equal(who_exits, expected_who_exits, check_dtype=False, check_less_precise=3)

    def test_condi_probs(self):
        """
        Testing generating conditional probabilities for condi_probs method.
        """

        # Initiate a BasicCopulaStrategy Class with a mixed CTG copula
        ctg = cgmix.CTGMixCop(cop_params=(4, 0.9, 4, 4), weights=(0.2, 0.4, 0.4))
        BCS = BasicCopulaStrategy(copula=ctg)

        # Check exception handling for NaN
        nan_data = {'s1': [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1, 0, 1],
                    's2': [1, np.nan, 2 / 5, 3 / 5, 1 / 5, 4 / 5, 5 / 9, 0, 1]}
        quantiles = pd.DataFrame.from_dict(nan_data)
        with self.assertRaises(ValueError):
            BCS.get_condi_probs(quantiles)

        # Check conditional probs
        data = {'s1': [0, 1 / 6, 2 / 6, 2 / 5, 4 / 6, 5 / 6, 1 / 9, 0, 1],
                's2': [1, 1 / 2, 2 / 5, 2 / 6, 1 / 5, 4 / 5, 5 / 9, 0, 1]}
        quantiles = pd.DataFrame.from_dict(data)
        expec_data = {'s1': [0, 0.015752, 0.297881, 0.643371, 0.990806, 0.707080, 0.003577, 0.237227, 0.711624],
                      's2': [1, 0.971310, 0.643371, 0.297881, 0.006264, 0.442387, 0.991056, 0.237227, 0.711624]}
        expec_condi_probs = pd.DataFrame.from_dict(expec_data)
        condi_probs = BCS.get_condi_probs(quantiles)
        pd.testing.assert_frame_equal(expec_condi_probs, condi_probs, check_dtype=False, check_less_precise=3)

        # Initiate a BasicCopulaStrategy Class with a N14 copula
        n14 = cg.N14(theta=4)
        BCS = BasicCopulaStrategy(copula=n14)

        # Check exception handling for NaN
        nan_data = {'s1': [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1, 0, 1],
                    's2': [1, np.nan, 2 / 5, 3 / 5, 1 / 5, 4 / 5, 5 / 9, 0, 1]}
        quantiles = pd.DataFrame.from_dict(nan_data)
        with self.assertRaises(ValueError):
            BCS.get_condi_probs(quantiles)

        # Check conditional probs
        data = {'s1': [0, 1 / 6, 2 / 6, 2 / 5, 4 / 6, 5 / 6, 1 / 9, 0, 1],
                's2': [1, 1 / 2, 2 / 5, 2 / 6, 1 / 5, 4 / 5, 5 / 9, 0, 1]}
        quantiles = pd.DataFrame.from_dict(data)
        expec_data = {'s1': [0, 0.009365, 0.292179, 0.679199, 0.997501, 0.742779, 0.001348, 0.261490, 0.594602],
                      's2': [1, 0.985101, 0.679200, 0.292179, 0.002212, 0.379101, 0.997212, 0.261490, 0.594602]}
        expec_condi_probs = pd.DataFrame.from_dict(expec_data)
        condi_probs = BCS.get_condi_probs(quantiles)
        pd.testing.assert_frame_equal(expec_condi_probs, condi_probs, check_dtype=False, check_less_precise=3)

    @staticmethod
    def test_get_positions():
        """
        Testing the get_positions method.
        """

        # Initiate a BasicCopulaStrategy Class with a mixed CTG copula
        ctg = cgmix.CTGMixCop(cop_params=(4, 0.9, 4, 4), weights=(0.2, 0.4, 0.4))
        BCS = BasicCopulaStrategy(copula=ctg)
        # Directly use arbitraty quantile data, so cdfs are just identity functions
        cdf1 = lambda x: x
        cdf2 = lambda x: x
        # Note: those are not conditional probabilities
        data = {'col1': [0.55, 0.55, 0.99, 0.86, 0.45, 0.44, 0.01, 0.43, 0.54, 0.56, 0.56],
                'col2': [0.45, 0.45, 0.01, 0.52, 0.52, 0.67, 0.99, 0.76, 0.75, 0.32, 0.32]}
        data_df = pd.DataFrame.from_dict(data)
        # Conditional probabilities calculated from above data using this copula are
        # 0   0.729714  0.262389
        # 1   0.729714  0.262389
        # 2   0.999930  0.000070
        # 3   0.979290  0.031447
        # 4   0.322782  0.663628
        # 5   0.089938  0.912821
        # 6   0.000070  0.999930
        # 7   0.035802  0.967859
        # 8   0.106652  0.913587
        # 9   0.915000  0.070514
        # 10  0.915000  0.070514

        # Under 'and' exiting logic
        # Get positions, basic
        positions = BCS.get_positions(data_df, cdf1, cdf2)
        expec_pos = pd.Series([0, 0, -1, -1, 0, 0, 1, 1, 1, 0, 0])
        pd.testing.assert_series_equal(positions, expec_pos, check_dtype=False, check_less_precise=3)
        # Get positions, initial long
        positions = BCS.get_positions(data_df, cdf1, cdf2, init_pos=1)
        expec_pos = pd.Series([1, 1, -1, -1, 0, 0, 1, 1, 1, 0, 0])
        pd.testing.assert_series_equal(positions, expec_pos, check_dtype=False, check_less_precise=3)
        # Get positions, initial short
        positions = BCS.get_positions(data_df, cdf1, cdf2, init_pos=-1)
        expec_pos = pd.Series([-1, -1, -1, -1, 0, 0, 1, 1, 1, 0, 0])
        pd.testing.assert_series_equal(positions, expec_pos, check_dtype=False, check_less_precise=3)
        # Get positions, custom open thresholds
        positions = BCS.get_positions(data_df, cdf1, cdf2, open_thresholds=(0.2, 0.8))
        expec_pos = pd.Series([0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0])
        pd.testing.assert_series_equal(positions, expec_pos, check_dtype=False, check_less_precise=3)
        # Get positions, custom exit thresholds
        positions = BCS.get_positions(data_df, cdf1, cdf2, exit_thresholds=(0.3, 0.8))
        expec_pos = pd.Series([0.0, 0.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0])
        pd.testing.assert_series_equal(positions, expec_pos, check_dtype=False, check_less_precise=3)

        # Under 'or' exiting logic
        # Get positions, basic
        positions = BCS.get_positions(data_df, cdf1, cdf2)
        expec_pos = pd.Series([0.0, 0.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0])
        pd.testing.assert_series_equal(positions, expec_pos, check_dtype=False, check_less_precise=3)
        # Get positions, initial long
        positions = BCS.get_positions(data_df, cdf1, cdf2, init_pos=1)
        expec_pos = pd.Series([1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0])
        pd.testing.assert_series_equal(positions, expec_pos, check_dtype=False, check_less_precise=3)
        # Get positions, initial short
        positions = BCS.get_positions(data_df, cdf1, cdf2, init_pos=-1)
        expec_pos = pd.Series([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0])
        pd.testing.assert_series_equal(positions, expec_pos, check_dtype=False, check_less_precise=3)
        # Get positions, custom open thresholds
        positions = BCS.get_positions(data_df, cdf1, cdf2, open_thresholds=(0.2, 0.8))
        expec_pos = pd.Series([0.0, 0.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0])
        pd.testing.assert_series_equal(positions, expec_pos, check_dtype=False, check_less_precise=3)
        # Get positions, custom exit thresholds
        positions = BCS.get_positions(data_df, cdf1, cdf2, exit_thresholds=(0.3, 0.8))
        expec_pos = pd.Series([0.0, 0.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0])
        pd.testing.assert_series_equal(positions, expec_pos, check_dtype=False, check_less_precise=3)

    def test_fit_copula_pure(self):
        """
        Testing the fit_copula method for pure copulas.
        """

        # Check if_renew
        BCS = BasicCopulaStrategy()
        _ = BCS.fit_copula(data=self.stocks, copula_name='Gumbel')
        _ = BCS.fit_copula(data=self.stocks, copula_name='N14', if_renew=False)
        self.assertIsInstance(BCS.copula, cg.Gumbel)

        # Create all names
        copula_names = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14', 'Gaussian', 'Student']
        result_dicts = [None] * 8
        fitted_copulas = [None] * 8

        # Fit to all pure copulas and store the results
        for i in range(8):
            result_dicts[i], fitted_copulas[i], _, _ = BCS.fit_copula(data=self.stocks, copula_name=copula_names[i])

        # Check Gumbel
        self.assertEqual(result_dicts[0]['Copula Name'], 'Gumbel')
        self.assertAlmostEqual(result_dicts[0]['SIC'], -1991.9496657125103, delta=1e-3)
        self.assertAlmostEqual(result_dicts[0]['AIC'], -1996.8584204971112, delta=1e-3)
        self.assertAlmostEqual(result_dicts[0]['HQIC'], -1994.9956755450632, delta=1e-3)
        self.assertAlmostEqual(result_dicts[0]['Log-likelihood'], 999.4312042665017, delta=1e-3)
        self.assertAlmostEqual(fitted_copulas[0].theta, 4.82392, delta=1e-3)

        # Check Clayton
        self.assertEqual(result_dicts[1]['Copula Name'], 'Clayton')
        self.assertAlmostEqual(result_dicts[1]['SIC'], -1977.2018488567405, delta=1e-3)
        self.assertAlmostEqual(result_dicts[1]['AIC'], -1982.1106036413414, delta=1e-3)
        self.assertAlmostEqual(result_dicts[1]['HQIC'], -1980.2478586892935, delta=1e-3)
        self.assertAlmostEqual(result_dicts[1]['Log-likelihood'], 992.0572958386168, delta=1e-3)
        self.assertAlmostEqual(fitted_copulas[1].theta, 7.64783, delta=1e-3)

        # Check Frank
        self.assertEqual(result_dicts[2]['Copula Name'], 'Frank')
        self.assertAlmostEqual(result_dicts[2]['SIC'], -2018.1903966292455, delta=1e-3)
        self.assertAlmostEqual(result_dicts[2]['AIC'], -2023.0991514138464, delta=1e-3)
        self.assertAlmostEqual(result_dicts[2]['HQIC'], -2021.2364064617984, delta=1e-3)
        self.assertAlmostEqual(result_dicts[2]['Log-likelihood'], 1012.5515697248693, delta=1e-3)
        self.assertAlmostEqual(fitted_copulas[2].theta, 17.4799, delta=1e-3)

        # Check Joe
        self.assertEqual(result_dicts[3]['Copula Name'], 'Joe')
        self.assertAlmostEqual(result_dicts[3]['SIC'], -1134.9875103888069, delta=1e-3)
        self.assertAlmostEqual(result_dicts[3]['AIC'], -1139.8962651734078, delta=1e-3)
        self.assertAlmostEqual(result_dicts[3]['HQIC'], -1138.0335202213598, delta=1e-3)
        self.assertAlmostEqual(result_dicts[3]['Log-likelihood'], 570.95012660465, delta=1e-3)
        self.assertAlmostEqual(fitted_copulas[3].theta, 8.41627, delta=1e-3)

        # Check N13
        self.assertEqual(result_dicts[4]['Copula Name'], 'N13')
        self.assertAlmostEqual(result_dicts[4]['SIC'], -2206.720787545359, delta=1e-3)
        self.assertAlmostEqual(result_dicts[4]['AIC'], -2211.6295423299603, delta=1e-3)
        self.assertAlmostEqual(result_dicts[4]['HQIC'], -2209.766797377912, delta=1e-3)
        self.assertAlmostEqual(result_dicts[4]['Log-likelihood'], 1106.8167651829262, delta=1e-3)
        self.assertAlmostEqual(fitted_copulas[4].theta, 13.0064, delta=1e-3)

        # Check N14
        self.assertEqual(result_dicts[5]['Copula Name'], 'N14')
        self.assertAlmostEqual(result_dicts[5]['SIC'], -2107.0744287234816, delta=1e-3)
        self.assertAlmostEqual(result_dicts[5]['AIC'], -2111.9831835080827, delta=1e-3)
        self.assertAlmostEqual(result_dicts[5]['HQIC'], -2110.1204385560345, delta=1e-3)
        self.assertAlmostEqual(result_dicts[5]['Log-likelihood'], 1056.9935857719875, delta=1e-3)
        self.assertAlmostEqual(fitted_copulas[5].theta, 4.32392, delta=1e-3)

        # Check Gaussian
        self.assertEqual(result_dicts[6]['Copula Name'], 'Gaussian')
        self.assertAlmostEqual(result_dicts[6]['SIC'], -2202.8581502050783, delta=1e-3)
        self.assertAlmostEqual(result_dicts[6]['AIC'], -2207.7669049896795, delta=1e-3)
        self.assertAlmostEqual(result_dicts[6]['HQIC'], -2205.9041600376313, delta=1e-3)
        self.assertAlmostEqual(result_dicts[6]['Log-likelihood'], 1104.8854465127858, delta=1e-3)
        self.assertAlmostEqual(fitted_copulas[6].rho, 0.94745, delta=1e-3)

        # Check Student
        self.assertEqual(result_dicts[7]['Copula Name'], 'Student')
        self.assertAlmostEqual(result_dicts[7]['SIC'], -2262.3979134229617, delta=1e-3)
        self.assertAlmostEqual(result_dicts[7]['AIC'], -2272.2114230160437, delta=1e-3)
        self.assertAlmostEqual(result_dicts[7]['HQIC'], -2268.4899330880676, delta=1e-3)
        self.assertAlmostEqual(result_dicts[7]['Log-likelihood'], 1138.111699531974, delta=1e-3)
        self.assertAlmostEqual(fitted_copulas[7].rho, 0.94745, delta=1e-3)
        self.assertAlmostEqual(fitted_copulas[7].nu, 5.72431, delta=1e-3)

    def test_fit_copula_mixcop(self):
        """
        Testing the fit_copula method for mixed copulas. This test is slow.
        """

        np.random.seed(724)
        data_df = self.stocks

        # Fit CFGMixCop to data using BCS
        BCS = BasicCopulaStrategy()
        result_dict, fitted_copula, _, _ = BCS.fit_copula(data_df, copula_name='CFGMixCop', gamma_scad=0.6)
        self.assertEqual(result_dict['Copula Name'], 'CFGMixCop')
        self.assertAlmostEqual(result_dict['SIC'], -2216.2580056272263, delta=1e-3)
        self.assertAlmostEqual(result_dict['AIC'], -2240.761659669632, delta=1e-3)
        self.assertAlmostEqual(result_dict['HQIC'], -2231.488054789991, delta=1e-3)
        self.assertAlmostEqual(result_dict['Log-likelihood'], 1125.410859864846, delta=1e-3)
        np.testing.assert_array_almost_equal(np.array([6.626, 4, 4.787]), fitted_copula.cop_params, decimal=2)
        np.testing.assert_array_almost_equal(np.array([0.504756, 0, 0.495244]), fitted_copula.weights, decimal=2)

        # Fit CTGMixCop to data using BCS
        result_dict, fitted_copula, _, _ = BCS.fit_copula(data_df, copula_name='CTGMixCop', gamma_scad=0.6)
        self.assertEqual(result_dict['Copula Name'], 'CTGMixCop')
        self.assertAlmostEqual(result_dict['SIC'], -2209.358326366888, delta=1e-3)
        self.assertAlmostEqual(result_dict['AIC'], -2238.7506149531737, delta=1e-3)
        self.assertAlmostEqual(result_dict['HQIC'], -2227.6343853622056, delta=1e-3)
        self.assertAlmostEqual(result_dict['Log-likelihood'], 1125.4173916449236, delta=1e-3)
        np.testing.assert_array_almost_equal(np.array([6.61214214, 0.51157163, 3.99975286, 4.79351409]),
                                             fitted_copula.cop_params, decimal=3)
        np.testing.assert_array_almost_equal(np.array([0.51051289, 0, 0.48948711]), fitted_copula.weights, decimal=3)
        # Reset the random seed
        np.random.seed(None)

    @staticmethod
    def test_get_cop_density_abs_class_method():
        """
        Testing the get_cop_density method in the Copula abstract class.
        """

        # u and v's for checking copula density
        us = [0, 1, 1, 0, 0.3, 0.7, 0.5]
        vs = [0, 1, 0, 1, 0.7, 0.3, 0.5]

        # Check for Frank
        cop = cg.Frank(theta=15)
        expected_densities = [1.499551e+01, 1.499551e+01, 4.589913e-06, 4.589913e-06, 3.700169e-02, 3.700169e-02,
                              3.754150e+00]
        densities = [cop.get_cop_density(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_densities, densities, decimal=4)

        # Check for N14
        cop = cg.N14(theta=5)
        expected_densities = [28447.02084968514, 114870.71560409002, 5.094257802793674e-28, 5.094257802793674e-28,
                              0.03230000161691527, 0.03230000161691527, 3.8585955174356292]
        densities = [cop.get_cop_density(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_densities, densities, decimal=4)

    @staticmethod
    def test_get_cop_eval_abs_class_method():
        """
        Testing the get_cop_eval method in the Copula abstract class.
        """

        # u and v's for checking copula density
        us = [0, 1, 1, 0, 0.3, 0.7, 0.5]
        vs = [0, 1, 0, 1, 0.7, 0.3, 0.5]

        # Check for Frank
        cop = cg.Frank(theta=15)
        expected_cop_evals = [1.4997754984537027e-09, 0.9999800014802172, 9.999999999541836e-06, 9.999999999541836e-06,
                              0.2998385964795436, 0.2998385964795436, 0.4538270500610275]
        cop_evals = [cop.get_cop_eval(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_cop_evals, cop_evals, decimal=4)

        # Check for N14
        cop = cg.N14(theta=5)
        expected_cop_evals = [5.3365811321695815e-06, 0.9999885130266996, 9.999999999999982e-06, 9.999999999999982e-06,
                              0.2999052240847225, 0.2999052240847225, 0.4545364421835644]
        cop_evals = [cop.get_cop_eval(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_cop_evals, cop_evals, decimal=4)

    @staticmethod
    def test_get_condi_prob_abs_class_method():
        """
        Testing the get_cop_eval method in the Copula abstract class.
        """

        # u and v's for checking copula density
        us = [0, 1, 1, 0, 0.3, 0.7, 0.5]
        vs = [0, 1, 0, 1, 0.7, 0.3, 0.5]

        # Check for Frank
        cop = cg.Frank(theta=15)
        expected_condi_probs = [0.0001499663031859714, 0.999850033362625, 0.9999999999541043, 4.589568752277862e-11,
                                0.0024452891307218463, 0.9975547108692793, 0.5000000000000212]
        condi_probs = [cop.get_condi_prob(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_condi_probs, condi_probs, decimal=4)

        # Check for N14
        cop = cg.N14(theta=5)
        expected_condi_probs = [0.2703284430766092, 0.5743481526129369, 0.9999999999999973, 2.4387404375289055e-33,
                                0.0019649735178081406, 0.9984409781303989, 0.5122647216509384]
        condi_probs = [cop.get_condi_prob(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_condi_probs, condi_probs, decimal=4)

    def test_plot_abs_class_method(self):
        """
        Testing the plot method in the Copula abstract class.
        """

        cov = [[1, 0.5], [0.5, 1]]
        nu = 4
        theta = 5
        gumbel = cg.Gumbel(theta=theta)
        frank = cg.Frank(theta=theta)
        clayton = cg.Clayton(theta=theta)
        joe = cg.Joe(theta=theta)
        n13 = cg.N13(theta=theta)
        n14 = cg.N14(theta=theta)
        gaussian = cg.GaussianCopula(cov=cov)
        student = cg.Student(cov=cov, nu=nu)

        # Initiate without an axes
        axs = dict()
        axs['Gumbel'] = gumbel.plot(200)
        axs['Frank'] = frank.plot(200)
        axs['Clayton'] = clayton.plot(200)
        axs['Joe'] = joe.plot(200)
        axs['N13'] = n13.plot(200)
        axs['N14'] = n14.plot(200)
        axs['Gaussian'] = gaussian.plot(200)
        axs['Student'] = student.plot(200)
        plt.close()

        for key in axs:
            self.assertEqual(str(type(axs[key])), "<class 'matplotlib.axes._subplots.AxesSubplot'>")

        # Initiate with axes
        _, ax = plt.subplots()
        axs = dict()
        axs['Gumbel'] = gumbel.plot(200, ax)
        axs['Frank'] = frank.plot(200, ax)
        axs['Clayton'] = clayton.plot(200, ax)
        axs['Joe'] = joe.plot(200, ax)
        axs['N13'] = n13.plot(200, ax)
        axs['N14'] = n14.plot(200, ax)
        axs['Gaussian'] = gaussian.plot(200, ax)
        axs['Student'] = student.plot(200, ax)
        plt.close()

        for key in axs:
            self.assertEqual(str(type(axs[key])), "<class 'matplotlib.axes._subplots.AxesSubplot'>")
