"""
Module for signal generation for Bollinger Bands Trading strategy.
"""

from collections import deque
from statistics import mean, stdev
from uuid import UUID

import numpy as np
import pandas as pd


class BollingerBandsTradingRule:
    """
    This class implements Bollinger Bands strategy from the book
    by E.P Chan: `"Algorithmic Trading: Winning Strategies and Their Rationale"
    <https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146>`_,
    page 70. The strategy generates a signal by when
    ``|z_score| >= entry_z_score`` and exit from a position when
    ``|z_score| <= |entry_z_score + z_score_delta|``.
    """

    def __init__(self, sma_window: int, std_window: int, entry_z_score: float = 3,
                 exit_z_score_delta: float = 6):
        """
        Class constructor.

        :param sma_window: (int) Window for SMA (Simple Moving Average).
        :param std_window: (int) Window for SMD (Simple Moving st. Deviation).
        :param entry_z_score: (float) Z-score value to enter (long or short) the position.
        :param exit_z_score_delta: (float) Number of z-score movements from `entry_z_score` to exit a position.
                                           If `entry_z_score` = 3, `exit_z_score_delta` = 6, it means that we either
                                           enter a position when z-score is >= 3 or <= -3, and exit when z-score
                                           <= -3 or >= 3 respectively.
        """

        self.open_trades = {}
        self.closed_trades = {}
        assert entry_z_score > 0, "Entry z-score must be positive"
        assert entry_z_score > 0, "Exit z-score delta must be positive"
        self.entry_z_score = entry_z_score
        self.exit_z_score_delta = exit_z_score_delta
        self.sma_window = sma_window
        self.std_window = std_window
        self.spread_series = deque(maxlen=max(sma_window, std_window))

    @staticmethod
    def get_z_score(spread_slice: deque, sma_window: int, std_window: int) -> float:
        """
        Calculate z-score of recent spread values.

        :param spread_slice: (deque) Spread deque to take values from.
        :param sma_window: (int) Window for SMA (Simple Moving Average).
        :param std_window: (int) Window for SMD (Simple Moving st. Deviation).
        :return: (float) Z score.
        """

        spread_slice_list = list(spread_slice)
        mean_spread = mean(spread_slice_list[-sma_window:])
        std_spread = stdev(spread_slice_list[-std_window:])

        return (spread_slice[-1] - mean_spread) / std_spread

    def update_spread_value(self, latest_spread_value: float):
        """
        Adds latest spread value to `self.spread_series`. Once it is done - one can check entry/exit signals.

        :param latest_spread_value: (float) Latest spread value.
        """

        self.spread_series.append(latest_spread_value)

    def check_entry_signal(self) -> tuple:
        """
        Function which checks entry condition for a spread series based on `self.z_score_entry`, `self.spread_series`.

        :return: (tuple) Tuple of boolean entry flag and side (if entry flag is True).
        """

        # Check if std is non-zero, otherwise it will return np.inf which is not tradable
        if np.std(self.spread_series, ddof=1) == 0:
            return False, None

        # Calculate z score
        z_score = self.get_z_score(self.spread_series, sma_window=self.sma_window, std_window=self.std_window)

        # Long entry
        if z_score <= - self.entry_z_score:
            side = 1
            return True, side
        # Short entry
        if z_score >= self.entry_z_score:
            side = -1
            return True, side
        return False, None

    def add_trade(self, start_timestamp: pd.Timestamp, side_prediction: int,
                  uuid: UUID = None):
        """
        Adds a new trade to track. Calculates trigger prices and trigger timestamp.

        :param start_timestamp: (pd.Timestamp) Timestamp of the future label.
        :param side_prediction: (int) External prediction for the future label.
        :param uuid: (str) Unique identifier used to link label to tradelog action.
        """

        initial_z_score = self.get_z_score(self.spread_series, sma_window=self.sma_window, std_window=self.std_window)
        self.open_trades[start_timestamp] = {
            'exit_z_score': initial_z_score + side_prediction * self.exit_z_score_delta,
            'start_value': self.spread_series[-1],
            'spread_series': list(self.spread_series),
            'uuid': uuid,
            'side': side_prediction,
            'latest_update_timestamp': start_timestamp,
            'initial_z_score': initial_z_score
        }

    def update_trades(self, update_timestamp: pd.Timestamp) -> list:
        """
        Checks whether any of the thresholds are triggered and currently open trades should be closed.

        :param update_timestamp: (pd.Timestamp) New timestamp to check vertical threshold.
        :return: (list) List of closed trades.
        """

        formed_trades_uuid = []  # Array of trades formed (uuid)
        to_close = {}  # Trades to close
        z_score = self.get_z_score(self.spread_series, self.sma_window, self.std_window)
        for timestamp, data in self.open_trades.items():
            data['latest_update_timestamp'] = update_timestamp
            if data['side'] > 0:
                if z_score >= data['exit_z_score']:
                    to_close[timestamp] = data
            else:
                if z_score <= data['exit_z_score']:
                    to_close[timestamp] = data

        if len(to_close) != 0:
            for timestamp, data in to_close.items():
                label_data = {'t1': update_timestamp, 'pt': z_score,
                              'uuid': data['uuid'], 'start_value': data['start_value'],
                              'end_value': self.spread_series[-1], 'side': data['side'],
                              'initial_z_score': data['initial_z_score']}
                formed_trades_uuid.append(data['uuid'])
                self.closed_trades[timestamp] = label_data

            self.open_trades = {
                timestamp: data
                for timestamp, data in self.open_trades.items()
                if timestamp not in to_close
            }

        return formed_trades_uuid
