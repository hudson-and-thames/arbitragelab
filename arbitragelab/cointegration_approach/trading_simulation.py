# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=invalid-name
"""
This module simulates trading based on the minimum profit trading signal, reports the trades,
and plots the equity curve.
"""

from typing import Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TradingSim:
    """
    This class simulates the trades based on an optimized minimum profit trading signal.
    It plots the trading signal on the cointegration error and the equity curves as well.
    """

    def __init__(self, starting_equity: float = np.Inf):
        """
        Setting up a new trading account for simulating the trading strategy.

        :param starting_equity: (float) The amount available to trade in this simulation.
        """

        # Set position and P&L to 0, fund the account with the dollar amount specified
        self._position = np.zeros((2, ))
        self._base_equity_value = starting_equity
        self._total_trades = np.zeros((2,))
        self._report = dict()

        # Record mark-to-market P&L everyday to give a proper view of drawdowns during the trades
        self._pnl = 0.
        self._mtm = None

    def initialize_report(self):
        """
        Initialize the dictionary for trade reports.
        """

        # Dictionary for generating the equity curve and trade report DataFrame
        self._report = {
            "Trade Date": [],
            "Trade Type": [],
            "Leg 1": [],
            "Leg 1 Shares": [],
            "Leg 1 Price": [],
            "Leg 2": [],
            "Leg 2 Shares": [],
            "Leg 2 Price": []}

        self._mtm = {
            "P&L": [],
            "Total Equity": []}

    def _trade(self, signals: pd.DataFrame, num_of_shares: np.array):
        """
        Trade the cointegrated pairs based on the optimized signal.

        :param signals: (pd.DataFrame) Dataframe that contains asset prices and trade signals.
        :param num_of_shares: (np.array) Optimized number of shares to trade.
        """

        # Generate report
        self.initialize_report()

        # Trading periods in the trade_df
        period = len(signals)

        # Add a flag to let the simulator know if a U-trade is currently open or a L-trade
        current_trade = 0

        # Start trading
        entry_price = np.zeros((2, ))

        for i in range(period):
            current_price = signals.iloc[i, [0, 1]].values

            # Check mark-to-market P&L
            trade_pnl = np.dot(current_price - entry_price, self._position)

            # Record mark-to-market P&L
            self._mtm['P&L'].append(trade_pnl)
            self._mtm['Total Equity'].append(self._base_equity_value + trade_pnl)

            if current_trade == 0:
                # No position, and the opening trade condition is satisfied
                # Before opening the trade, check if the dollar constraint allows us to open
                capital_req = np.dot(current_price, num_of_shares)

                # Capital requirement satisfied, open the position
                if capital_req <= self._base_equity_value:
                    # Record the entry price.
                    entry_price = current_price

                    # Do we open a U-trade or L-trade?
                    if signals['otc_U'].iloc[i]:
                        # U-trade, short share S1, long share S2
                        self._report['Trade Type'].append("U-trade Open")
                        self._position = num_of_shares * np.array([-1, 1])
                        current_trade = 1

                    elif signals['otc_L'].iloc[i]:
                        # L-trade, long share S1, short share S2
                        self._report['Trade Type'].append("L-trade Open")
                        self._position = num_of_shares * np.array([1, -1])
                        current_trade = -1
                    else:
                        # No opening condition met, forward to next day
                        continue

                    # Bookkeeping
                    self._report['Trade Date'].append(signals.index[i].date())
                    self._report['Leg 1'].append(signals.columns[0])
                    self._report['Leg 2'].append(signals.columns[1])
                    self._report['Leg 1 Price'].append(entry_price[0])
                    self._report['Leg 2 Price'].append(entry_price[1])
                    self._report['Leg 1 Shares'].append(self._position[0])
                    self._report['Leg 2 Shares'].append(self._position[1])

                # Make sure the trade will not be closed on the same day (using elif)

            else:
                # We have a trade on
                if current_trade == 1 and signals['ctc_U'].iloc[i]:
                    # The open trade is a U-trade
                    self._report['Trade Type'].append("U-trade Close")
                    self._total_trades[0] += 1

                elif current_trade == -1 and signals['ctc_L'].iloc[i]:
                    # The open trade is a L-trade
                    self._report['Trade Type'].append("L-trade Close")
                    self._total_trades[1] += 1

                else:
                    # No condition triggered, just forward to next day
                    continue

                # Bookkeeping
                self._report['Trade Date'].append(signals.index[i].date())
                self._report['Leg 1'].append(signals.columns[0])
                self._report['Leg 2'].append(signals.columns[1])
                self._report['Leg 1 Price'].append(current_price[0])
                self._report['Leg 2 Price'].append(current_price[1])
                self._report['Leg 1 Shares'].append(-1 * self._position[0])
                self._report['Leg 2 Shares'].append(-1 * self._position[1])

                # Add the final profit to base equity value.

                self._base_equity_value += trade_pnl

                # Clear the trade book.
                current_trade = 0
                entry_price = np.zeros((2,))
                self._position = np.zeros((2,))

                # Will not close the trade on the same day (using elif)

    def summary(self, signals: pd.DataFrame, num_of_shares: np.array) -> pd.DataFrame:
        """
        Trade the strategy and generate the trade reports.

        :param signals: (pd.DataFrame) Dataframe that contains asset prices and trade signals.
        :param num_of_shares: (np.array) Optimized number of shares to trade.
        :return (pd.DataFrame, np.array): A dataframe that contains each opening/closing trade details, P&L,
            and equity curves; a NumPy array that represents the number of U-trade and L-trade over the period
        """

        self._trade(signals, num_of_shares)
        report_df = pd.DataFrame(self._report)
        return report_df

    def _plot_signals(self, signals: pd.DataFrame, num_of_shares: np.array, cond_lines: np.array,
                      figw: float = 15, figh: float = 10, start_date: Optional[pd.Timestamp] = None,
                      end_date: Optional[pd.Timestamp] = None) -> plt.Figure:
        """
        Plot the spread and the signals

        :param signals: (pd.DataFrame) Dataframe that contains the trading signal.
        :param num_of_shares: (np.array) Numpy array that contains the number of shares.
        :param cond_lines: (np.array) Numpy array that contains the trade initiation/close signal line.
        :param figw: (float) Figure width.
        :param figh: (float) Figure height.
        :param start_date: (pd.Timestamp) The starting point of the plot.
        :param end_date: (pd.Timestamp) The end point of the plot.
        :return: (plt.Figure) The object of the plot.
        """

        # Retrieve the trade report for trading dates
        report = self.summary(signals, num_of_shares)

        # Define the ticks on the x-axis
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('\n%Y')
        months_fmt = mdates.DateFormatter('%b')

        # Plot the price action of each leg as well as the cointegration error
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(figw, figh), gridspec_kw={'height_ratios': [2.5, 1]})

        # Plot prices
        axes[0].plot(signals.iloc[:, 0], label=signals.columns[0])
        axes[0].plot(signals.iloc[:, 1], label=signals.columns[1])
        axes[0].legend(loc='upper left', fontsize=12)
        axes[0].tick_params(axis='y', labelsize=14)

        # Plot cointegration error
        axes[1].plot(signals.iloc[:, 2], label='spread')
        axes[1].legend(loc='best', fontsize=12)

        # Plot signal lines
        axes[1].axhline(y=cond_lines[1], color='black')  # Closing condition
        axes[1].axhline(y=cond_lines[0], color='red')  # L-trade opens
        axes[1].axhline(y=cond_lines[2], color='green')  # U-trade opens

        # Formatting the tick labels
        axes[1].xaxis.set_major_locator(years)
        axes[1].xaxis.set_major_formatter(years_fmt)
        axes[1].xaxis.set_minor_locator(months)
        axes[1].xaxis.set_minor_formatter(months_fmt)
        axes[1].tick_params(axis='x', labelsize=14)
        axes[1].tick_params(axis='y', labelsize=14)

        # Define the date range of the plot
        if start_date is not None and end_date is not None:
            axes[1].set_xlim((start_date, end_date))

        # Plot arrows for buy and sell signal
        for idx in range(len(report)):
            trade_type = report.iloc[idx]['Trade Type']
            trade_date = report.iloc[idx]['Trade Date']
            arrow_xpos = mdates.date2num(trade_date)
            arrow_ypos = signals.loc[pd.Timestamp(trade_date)]['coint_error']

            # Green arrow for opening U-trade, red arrow for opening L-trade, black arrow for closing the trade.
            if trade_type == "U-trade Open":
                axes[1].annotate("", (arrow_xpos, arrow_ypos), xytext=(0, 15),
                                 textcoords='offset points', arrowprops=dict(arrowstyle='-|>', color='green'))
            elif trade_type == "L-trade Open":
                axes[1].annotate("", (arrow_xpos, arrow_ypos), xytext=(0, -15),
                                 textcoords='offset points', arrowprops=dict(arrowstyle='-|>', color='red'))
            elif trade_type == "L-trade Close":
                axes[1].annotate("", (arrow_xpos, arrow_ypos), xytext=(0, 15),
                                 textcoords='offset points', arrowprops=dict(arrowstyle='-|>', color='black'))
            else:
                axes[1].annotate("", (arrow_xpos, arrow_ypos), xytext=(0, -15),
                                 textcoords='offset points', arrowprops=dict(arrowstyle='-|>', color='black'))

        fig.suptitle("Optimal Pre-set Boundaries and Trading Signals", fontsize=20)
        return fig

    def _plot_pnl_curve(self, signal: pd.DataFrame, figw: float = 15., figh: float = 10.,
                        start_date: Optional[pd.Timestamp] = None,
                        end_date: Optional[pd.Timestamp] = None) -> plt.Figure:
        """
        Plot the equity curve (marked to market daily) trading the strategy.

        :param signal: (pd.DataFrame) Dataframe containing the trading signal.
        :param figw: (float) Figure width.
        :param figh: (float) Figure heght.
        :param start_date: (pd.Timestamp) The starting point of the plot.
        :param end_date: (pd.Timestamp) The end point of the plot.
        :return: (plt.Figure) The object of the plot.
        """

        # Build the equity curve dataframe
        equity_curve_df = pd.DataFrame(self._mtm)

        # Set up date index
        equity_curve_df.index = signal.index

        # Define the ticks on the x-axis
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('\n%Y')
        months_fmt = mdates.DateFormatter('%b')

        # Plot the equity curve
        fig, ax = plt.subplots(figsize=(figw, figh))
        ax.plot(equity_curve_df['Total Equity'])

        # Formatting the tick labels
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
        ax.xaxis.set_minor_formatter(months_fmt)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        # Define the date range of the plot
        if start_date is not None and end_date is not None:
            ax.set_xlim((start_date, end_date))

        fig.suptitle("P&L Curve of the Trading Strategy", fontsize=20)
        return fig

    def plot_strategy(self, signal: pd.DataFrame, num_of_shares: np.array, cond_lines: np.array,
                      figw: float = 15, figh: float = 10, start_date: Optional[pd.Timestamp] = None,
                      end_date: Optional[pd.Timestamp] = None) -> Tuple[plt.Figure, plt.Figure]:
        """
        Plot the trading signal and the PnL curve of the trading strategy

        :param signal: (pd.DataFrame) Dataframe containing the trading signal.
        :param num_of_shares: (np.array) Numpy array that contains the number of shares.
        :param cond_lines: (np.array) Numpy array that contains the trade initiation/close signal line.
        :param figw: (float) Figure width.
        :param figh: (float) Figure heght.
        :param start_date: (pd.Timestamp) The starting point of the plot.
        :param end_date: (pd.Timestamp) The end point of the plot.
        :return: (plt.Figure, plt.Figure) The object of the trading signal plot; the object of the P&L curve plot.
        """

        # A wrapper function to plot two figures
        fig1 = self._plot_signals(signal, num_of_shares, cond_lines, figw=figw, figh=figh,
                                  start_date=start_date, end_date=end_date)
        fig2 = self._plot_pnl_curve(signal, figw=figw, figh=figh, start_date=start_date, end_date=end_date)

        return fig1, fig2
