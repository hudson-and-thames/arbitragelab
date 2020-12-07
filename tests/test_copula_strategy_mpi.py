# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Unit tests for basic copula strategy.
"""
# pylint: disable = invalid-name,  protected-access
import os
import unittest
import datetime as dt
import numpy as np
import pandas as pd
from arbitragelab.copula_approach import copula_strategy_mpi


class TestCopulaStrategyMPI(unittest.TestCase):
    """
    Testing methods in CopulaStrategyMPI
    """

    def setUp(self):
        project_path = os.path.dirname(__file__)
        self.data_path = project_path + r'/test_data'

        self.pair_prices = pd.read_csv(self.data_path + r'/BKD_ESC_2009_2011.csv', index_col=0)
        formatted_dates = [dt.datetime.strptime(d, '%m/%d/%Y').date() for d in self.pair_prices.index]
        self.pair_prices.index = formatted_dates

    def test_exit_trigger(self):
        """
        Testing the exiting trigger logic from _exit_trigger method.
        """

        CS = copula_strategy_mpi.CopulaStrategyMPI()
        flag_0_0 = pd.Series([0, 0])
        flag_05_05 = pd.Series([0.5, 0.5])
        flag_n05_n05 = pd.Series([-0.5, -0.5])
        flag_02_05 = pd.Series([0.2, 0.5])
        flag_01_3 = pd.Series([0.1, 3])
        flag_3_0 = pd.Series([3, 0])
        flag_n2_n2 = pd.Series([-2, -2])

        self.assertTrue(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_3_0, open_based_on=[-1, 1]))
        self.assertTrue(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_n2_n2, open_based_on=[-1, 1]))
        self.assertTrue(CS._exit_trigger(pre_flag=flag_05_05, raw_cur_flag=flag_0_0, open_based_on=[-1, 1]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_02_05, raw_cur_flag=flag_05_05, open_based_on=[-1, 1]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_02_05, raw_cur_flag=flag_01_3, open_based_on=[-1, 1]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_0_0, open_based_on=[-1, 1]))

        self.assertTrue(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_3_0, open_based_on=[1, 1]))
        self.assertTrue(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_n2_n2, open_based_on=[1, 1]))
        self.assertTrue(CS._exit_trigger(pre_flag=flag_n05_n05, raw_cur_flag=flag_05_05, open_based_on=[1, 1]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_02_05, raw_cur_flag=flag_05_05, open_based_on=[1, 1]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_02_05, raw_cur_flag=flag_01_3, open_based_on=[1, 1]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_0_0, open_based_on=[1, 1]))

        self.assertFalse(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_3_0, open_based_on=[-1, 2]))
        self.assertTrue(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_n2_n2, open_based_on=[-1, 2]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_05_05, raw_cur_flag=flag_0_0, open_based_on=[-1, 2]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_02_05, raw_cur_flag=flag_05_05, open_based_on=[-1, 2]))
        self.assertTrue(CS._exit_trigger(pre_flag=flag_02_05, raw_cur_flag=flag_01_3, open_based_on=[-1, 2]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_0_0, open_based_on=[-1, 2]))

        self.assertFalse(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_3_0, open_based_on=[1, 2]))
        self.assertTrue(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_n2_n2, open_based_on=[1, 2]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_n05_n05, raw_cur_flag=flag_05_05, open_based_on=[1, 2]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_02_05, raw_cur_flag=flag_05_05, open_based_on=[1, 2]))
        self.assertTrue(CS._exit_trigger(pre_flag=flag_02_05, raw_cur_flag=flag_01_3, open_based_on=[1, 2]))
        self.assertFalse(CS._exit_trigger(pre_flag=flag_0_0, raw_cur_flag=flag_0_0, open_based_on=[1, 2]))

    def test_get_position_and_reset_flag(self):
        """
        Testing position opening logic from _get_positions_and_reset_flag method.
        """

        CS = copula_strategy_mpi.CopulaStrategyMPI()
        flag_0_0 = pd.Series([0, 0])
        flag_1_0 = pd.Series([1, 0])
        flag_0_1 = pd.Series([0, 1])
        flag_n1_0 = pd.Series([-1, 0])
        flag_0_n1 = pd.Series([0, -1])
        flag_3_0 = pd.Series([3, 0])

        # Long trigger test.
        result = CS._get_position_and_reset_flag(pre_flag=flag_0_0, raw_cur_flag=flag_0_1,
                                                 pre_position=0, open_based_on=[0, 0])
        self.assertEqual(result, (1, False, [1, 2]))
        result = CS._get_position_and_reset_flag(pre_flag=flag_0_0, raw_cur_flag=flag_n1_0,
                                                 pre_position=0, open_based_on=[0, 0])
        self.assertEqual(result, (1, False, [1, 1]))

        # Short trigger test.
        result = CS._get_position_and_reset_flag(pre_flag=flag_0_0, raw_cur_flag=flag_1_0,
                                                 pre_position=0, open_based_on=[0, 0])
        self.assertEqual(result, (-1, False, [-1, 1]))
        result = CS._get_position_and_reset_flag(pre_flag=flag_0_0, raw_cur_flag=flag_0_n1,
                                                 pre_position=0, open_based_on=[0, 0])
        self.assertEqual(result, (-1, False, [-1, 2]))

        # Long then short.
        result = CS._get_position_and_reset_flag(pre_flag=flag_0_0, raw_cur_flag=flag_0_n1,
                                                 pre_position=1, open_based_on=[1, 1])
        self.assertEqual(result, (-1, False, [-1, 2]))
        result = CS._get_position_and_reset_flag(pre_flag=flag_0_0, raw_cur_flag=flag_1_0,
                                                 pre_position=1, open_based_on=[1, 2])
        self.assertEqual(result, (-1, False, [-1, 1]))

        # Short then long
        result = CS._get_position_and_reset_flag(pre_flag=flag_0_0, raw_cur_flag=flag_0_1,
                                                 pre_position=-1, open_based_on=[-1, 2])
        self.assertEqual(result, (1, False, [1, 2]))
        result = CS._get_position_and_reset_flag(pre_flag=flag_0_0, raw_cur_flag=flag_n1_0,
                                                 pre_position=-1, open_based_on=[-1, 1])
        self.assertEqual(result, (1, False, [1, 1]))

        # Exit
        result = CS._get_position_and_reset_flag(pre_flag=flag_0_0, raw_cur_flag=flag_3_0,
                                                 pre_position=-1, open_based_on=[-1, 1])
        self.assertEqual(result, (0, True, [0, 0]))
        # The following is ambiguous. The author did not specify the following situation.
        result = CS._get_position_and_reset_flag(pre_flag=flag_0_0, raw_cur_flag=flag_3_0,
                                                 pre_position=-1, open_based_on=[-1, 2])
        self.assertEqual(result, (-1, False, [-1, 1]))

    def test_cur_flag_and_position(self):
        """
        Testing current flag and positions generating from _cur_flag_and_position method.
        """

        CS = copula_strategy_mpi.CopulaStrategyMPI()

        # result = (cur_flag: pd.Series, cur_position: int, open_based_on: list)
        # Check long.
        result = CS._cur_flag_and_position(mpi=pd.Series([0.4, 0.9]), pre_flag=pd.Series([0.3, 0.3]),
                                           pre_position=0, open_based_on=[0, 0], enable_reset_flag=True)
        np.testing.assert_array_almost_equal(result[0], pd.Series([0.2, 0.7]), decimal=6)
        self.assertEqual(result[1:], (1, [1, 2]))
        # Check short.
        result = CS._cur_flag_and_position(mpi=pd.Series([0.9, 0.4]), pre_flag=pd.Series([0.3, 0.3]),
                                           pre_position=0, open_based_on=[0, 0], enable_reset_flag=True)
        np.testing.assert_array_almost_equal(result[0], pd.Series([0.7, 0.2]), decimal=6)
        self.assertEqual(result[1:], (-1, [-1, 1]))
        # Check exit by x-ing 0, reset flag.
        result = CS._cur_flag_and_position(mpi=pd.Series([0.9, 0.4]), pre_flag=pd.Series([-0.3, -0.3]),
                                           pre_position=1, open_based_on=[1, 1], enable_reset_flag=True)
        np.testing.assert_array_almost_equal(result[0], pd.Series([0, 0]), decimal=6)
        self.assertEqual(result[1:], (0, [0, 0]))
        # Check exit by reaching stop loss position, reset flag.
        result = CS._cur_flag_and_position(mpi=pd.Series([0.9, 0.4]), pre_flag=pd.Series([1.7, 0]),
                                           pre_position=1, open_based_on=[1, 1], enable_reset_flag=True)
        np.testing.assert_array_almost_equal(result[0], pd.Series([0, 0]), decimal=6)
        self.assertEqual(result[1:], (0, [0, 0]))
        # Check exit by x-ing 0, no reset flag.
        result = CS._cur_flag_and_position(mpi=pd.Series([0.9, 0.4]), pre_flag=pd.Series([-0.3, -0.3]),
                                           pre_position=1, open_based_on=[1, 1], enable_reset_flag=False)
        np.testing.assert_array_almost_equal(result[0], pd.Series([0.1, -0.4]), decimal=6)
        self.assertEqual(result[1:], (0, [0, 0]))
        # Check exit by reaching stop loss position, no reset flag.
        result = CS._cur_flag_and_position(mpi=pd.Series([0.9, 0.4]), pre_flag=pd.Series([1.7, 0]),
                                           pre_position=1, open_based_on=[1, 1], enable_reset_flag=False)
        np.testing.assert_array_almost_equal(result[0], pd.Series([2.1, -0.1]), decimal=6)
        self.assertEqual(result[1:], (0, [0, 0]))

    def test_get_position_and_flags(self):
        """
        Testing positions and flags generation from get_position_and_flags method.
        """

        CSMPI = copula_strategy_mpi.CopulaStrategyMPI(opening_triggers=(-0.6, 0.6), stop_loss_positions=(-2, 2))
        returns = CSMPI.to_returns(pair_prices=self.pair_prices)
        # Fit an N14 copula.
        s1_series = returns.iloc[:, 0]
        s2_series = returns.iloc[:, 1]
        _, copula, cdf1, cdf2 = CSMPI.fit_copula(s1_series, s2_series, copula_name='N14')
        # Forming positions and flags.
        _, _ = CSMPI.get_positions_and_flags(returns, cdf1, cdf2, enable_reset_flag=True)
        # Check goodness of copula fit by its coefficient.
        self.assertAlmostEqual(copula.theta, 1.3989032623979707)
        # Check numbers of triggers.
        self.assertEqual(CSMPI._long_count, 338)
        self.assertEqual(CSMPI._exit_count, 31)
        self.assertEqual(CSMPI._short_count, 478)
