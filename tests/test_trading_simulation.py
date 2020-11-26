# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Tests function of Minimum Profit Condition Optimization module:
cointegration_approach/trading_simulation.py
"""

import os
import pickle
import unittest

import pandas as pd
from matplotlib.dates import num2date
from matplotlib.figure import Figure

from arbitragelab.cointegration_approach.trading_simulation import TradingSim


class TestTradingSimulation(unittest.TestCase):
    """
    Test Minimum Profit Condition Optimization module: trading strategy simulation and plotting.
    """

    def setUp(self):
        """
        Set up the parameters for simulations.
        """

        # Load a pre-calculated trade signal
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/trading_strategy_example.pkl'
        with open(data_path, 'rb') as signal_f:
            self.signal, self.num_of_shares, self.cond_lines = pickle.load(signal_f)

    def test_initialize(self):
        """
        Test the initialize report function.
        """

        # Set up a trading account
        trader = TradingSim(1000000.)

        # Initialize the report
        trader.initialize_report()

        # Test if the report has been properly initialized
        self.assertIsNotNone(trader._mtm)
        self.assertIsNotNone(trader._report)

    def test_summary(self):
        """
        Test the summary function.
        """

        # Set up a trading account
        trader = TradingSim(1000000.)

        # Trade the strategy and generate the report
        reports = trader.summary(self.signal, self.num_of_shares)

        # See if the columns of the report are correctly generated
        report_cols = ["Trade Date", "Trade Type", "Leg 1", "Leg 1 Shares", "Leg 1 Price",
                       "Leg 2", "Leg 2 Shares", "Leg 2 Price"]
        self.assertListEqual(list(reports.columns), report_cols)

        # See if the trade counts matches
        trade_num = 39
        self.assertEqual(len(reports), trade_num)

        # See if the final total profit matches
        final_profit = trader._mtm['Total Equity'][-1] - 1000000.
        self.assertAlmostEqual(final_profit, 3584.14)

    def test_plot_strategy(self):
        """
        Test the plotting function.
        """

        trader = TradingSim(1000000.)

        # Trade the strategy and generate the report
        _ = trader.summary(self.signal, self.num_of_shares)

        # Plot the figures
        fig1, fig2 = trader.plot_strategy(self.signal, self.num_of_shares, self.cond_lines)

        # First check if the plot object has been generated
        self.assertTrue(issubclass(type(fig1), Figure))
        self.assertTrue(issubclass(type(fig2), Figure))

        ax1, _ = fig1.get_axes()

        # Check the xlim of the plot when nothing was specified
        plot_no_spec_xlim_left = num2date(ax1.get_xlim()[0])
        plot1_day = plot_no_spec_xlim_left.day
        plot1_month = plot_no_spec_xlim_left.month
        plot1_year = plot_no_spec_xlim_left.year

        self.assertEqual(plot1_year, 2018)
        self.assertEqual(plot1_month, 11)
        self.assertEqual(plot1_day, 27)

        # Now check when the xlim has been set to specific dates
        fig1, fig2 = trader.plot_strategy(self.signal, self.num_of_shares, self.cond_lines,
                                          start_date=pd.Timestamp(2019, 2, 1), end_date=pd.Timestamp(2020, 5, 1))
        _, ax2 = fig1.get_axes()
        ax3 = fig2.get_axes()

        plot_spec_xlim_left = num2date(ax2.get_xlim()[0])
        plot_spec_xlim_right = num2date(ax3[0].get_xlim()[1])

        # Now check the left value of the xlim
        plot2_day = plot_spec_xlim_left.day
        plot2_month = plot_spec_xlim_left.month
        plot2_year = plot_spec_xlim_left.year

        self.assertEqual(plot2_year, 2019)
        self.assertEqual(plot2_month, 2)
        self.assertEqual(plot2_day, 1)

        # Now check the right value of the xlim
        plot3_day = plot_spec_xlim_right.day
        plot3_month = plot_spec_xlim_right.month
        plot3_year = plot_spec_xlim_right.year

        self.assertEqual(plot3_year, 2020)
        self.assertEqual(plot3_month, 5)
        self.assertEqual(plot3_day, 1)

        # Now check the figure size
        _, fig2 = trader.plot_strategy(self.signal, self.num_of_shares, self.cond_lines, figw=25, figh=15,
                                       start_date=pd.Timestamp(2019, 2, 1), end_date=pd.Timestamp(2020, 5, 1))

        width, height = fig2.get_size_inches()
        self.assertEqual(width, 25.)
        self.assertEqual(height, 15.)