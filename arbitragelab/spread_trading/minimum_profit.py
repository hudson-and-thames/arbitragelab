# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Module for signal generation for Minimum Profit Optimization Trading strategy.
"""

from collections import deque
from uuid import UUID

import numpy as np
import pandas as pd

from arbitragelab.util import segment


class MinimumProfitTradingRule:
    """
    This class implements trading strategy from the Minimum Profit Optimization method from the paper
    by Lin, Y.-X., McCrae, M., and Gulati, C.
    `"Loss protection in pairs trading through minimum profit bounds: a cointegration approach"
    <http://downloads.hindawi.com/archive/2006/073803.pdf>`_

    The strategy generates a signal by when
    ``spread <= buy_level`` or ``spread => sell_level`` and exit from a position when
    ``|spread| <= |close_level|``.

    This strategy allows only one open trade at a time.
    """

    def __init__(self, shares: np.array, optimal_levels: np.array, spread_window: int = 10):
        """
        Class constructor.

        :param shares: (np.array) Number of shared to trade per leg in the cointegration pair.
        :param optimal_levels: (np.array) Optimal levels to enter aenter and close a trade.
        :param spread_window: (int) Number of previous spread values to print when reporting trades.
        """

        segment.track('MinimumProfitTradingRule')

        self.open_trades = {}
        self.closed_trades = {}

        # Testing validity of inputs
        assert optimal_levels[0] <= optimal_levels[1], "Closing level for a buy signal must be higher than entry"
        assert optimal_levels[1] <= optimal_levels[2], "Closing level for a sell signal must be lower than entry"
        self.entry_buy_signal = optimal_levels[0]
        self.entry_sell_signal = optimal_levels[2]
        self.exit_signal = optimal_levels[1]
        self.shares = shares
        self.spread_series = deque(maxlen=spread_window)
        self.trade = 0  # Current trade status

    def update_spread_value(self, latest_spread_value: float):
        """
        Adds latest spread value to `self.spread_series`. Once it is done - one can check entry/exit signals.

        :param latest_spread_value: (float) Latest spread value.
        """

        self.spread_series.append(latest_spread_value)

    def check_entry_signal(self) -> tuple:
        """
        Function which checks entry condition for a spread series based on `self.entry_buy_signal`,
        `self.entry_sell_signal`, `self.spread_series`.

        :return: (tuple) Tuple of boolean entry flag and side (if entry flag is True).
        """

        # Long entry
        if self.trade == 0 and self.spread_series[-1] <= self.entry_buy_signal:
            side = 1
            self.trade = 1
            return True, side

        # Short entry
        if self.trade == 0 and self.spread_series[-1] >= self.entry_sell_signal:
            side = -1
            self.trade = -1
            return True, side

        return False, None

    def add_trade(self, start_timestamp: pd.Timestamp, side_prediction: int,
                  uuid: UUID = None, shares: np.array = None):
        """
        Adds a new trade to track. Calculates trigger prices and trigger timestamp.

        :param start_timestamp: (pd.Timestamp) Timestamp of the future label.
        :param side_prediction: (int) External prediction for the future label.
        :param uuid: (str) Unique identifier used to link label to tradelog action.
        :param shares: (np.array) Number of shares bought and sold per asset.
        """

        self.open_trades[start_timestamp] = {
            'exit_level': self.exit_signal,
            'start_value': self.spread_series[-1],
            'spread_series': list(self.spread_series),
            'uuid': uuid,
            'side': side_prediction,
            'shares': shares,
            'latest_update_timestamp': start_timestamp
        }

    def update_trades(self, update_timestamp: pd.Timestamp) -> list:
        """
        Checks whether any of the thresholds are triggered and currently open trades should be closed.

        :param update_timestamp: (pd.Timestamp) New timestamp to check vertical threshold.
        :return: (list) List of closed trades.
        """

        formed_trades_uuid = []  # Array of trades formed (uuid)
        to_close = {}  # Trades to close

        for timestamp, data in self.open_trades.items():
            data['latest_update_timestamp'] = update_timestamp
            if data['side'] > 0:
                if self.spread_series[-1] >= data['exit_level']:
                    self.trade = 0
                    to_close[timestamp] = data
            else:
                if self.spread_series[-1] <= data['exit_level']:
                    self.trade = 0
                    to_close[timestamp] = data

        if len(to_close) != 0:
            for timestamp, data in to_close.items():
                label_data = {'t1': update_timestamp, 'pt': self.spread_series[-1],
                              'uuid': data['uuid'], 'start_value': data['start_value'],
                              'end_value': self.spread_series[-1], 'side': data['side']}
                formed_trades_uuid.append(data['uuid'])
                self.closed_trades[timestamp] = label_data

            self.open_trades = {
                timestamp: data
                for timestamp, data in self.open_trades.items()
                if timestamp not in to_close
            }

        return formed_trades_uuid
