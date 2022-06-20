# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Master module that implements the basic copula trading strategy.

This module is a realization of the methodology in the following paper:
`Liew, R.Q. and Wu, Y., 2013. Pairs trading: A copula approach. Journal of Derivatives & Hedge Funds, 19(1), pp.12-30.
<https://dr.ntu.edu.sg/bitstream/10220/17826/1/jdhf20131a.pdf>`__

This module is almost identical in terms of functionality as copula_strategy. But is designed with better efficiency,
better structure, native pandas support, and supports mixed copulas. The trading logic is more clearly defined and all
wrapped in one method for easier adjustment when needed, due to the ambiguities from the paper.
"""

# pylint: disable = invalid-name
from typing import Tuple, Callable, Dict
from uuid import UUID

import pandas as pd


class BasicCopulaTradingRule:
    """
    This is the threshold basic copula trading strategy implemented by [Liew et al. 2013]. First, one uses
    formation period prices to train a copula, then trade based on conditional probabilities calculated from the
    quantiles of the current price u1 and u2. If we define the spread as stock 1 in relation to stock 2, then the
    logic is as follows (All the thresholds can be customized via open_thresholds, exit_thresholds parameters):

            - If P(U1 <= u1 | U2 = u2) <= 0.05 AND P(U2 <= u2 | U1 = u1) >= 0.95, then stock 1 is under-valued and
              stock 2 is over-valued. Thus we long the spread.
            - If P(U1 <= u1 | U2 = u2) >= 0.95 AND P(U2 <= u2 | U1 = u1) <= 0.05, then stock 2 is under-valued and
              stock 1 is over-valued. Thus we short the spread.
            - We close the position if the conditional probabilities cross with 0.5 (`exit_probabilities`).

    For the exiting condition, the author proposed a closure when stock 1 AND 2's conditional probabilities cross
    0.5. However, we found it sometimes too strict and fails to exit a position when it should occasionally. Hence
    we also provide the OR logic implementation. You can use it by setting exit_rule='or'. Also note that the
    signal generation is independent from the current position.
    """

    def __init__(self, open_probabilities: Tuple[float, float] = (0.05, 0.95),
                 exit_probabilities: Tuple[float, float] = (0.5, 0.5), exit_rule: str = 'and'):
        """
        Class constructor.

        :param open_probabilities: (tuple) Optional. The default lower and upper threshold for opening a position for
            trading signal generation. Defaults to (0.05, 0.95).
        :param exit_probabilities: (tuple) Optional. The default lower and upper threshold for exiting a position for
            trading signal generation. Defaults to (0.5, 0.5).
        :param exit_rule: (str) Optional. The logic for triggering an exit signal. Available choices are 'and', 'or'.
            They indicate whether both conditional probabilities need to cross 0.5. Defaults to 'and'.
        """
        self.open_probabilities = open_probabilities
        self.exit_probabilities = exit_probabilities
        self.exit_rule = exit_rule

        # Trading info.
        self.open_trades = {}
        self.closed_trades = {}
        self.current_probabilities = tuple()
        self.prev_probabilities = tuple()

        self.copula = None  # Fit copula.
        self.cdf_x = None
        self.cdf_y = None

    def set_copula(self, copula: object):
        """
        Set fit copula to `self.copula`.

        :param copula: (object) Fit copula object.
        """
        self.copula = copula

    def set_cdf(self, cdf_x: Callable[[float], float], cdf_y: Callable[[float], float]):
        """
        Set marginal C.D.Fs functions which transform X, Y values into probabilities, usually ECDFs are used. One can
        use `construct_ecdf_lin` function from copula_calculations module.


        :param cdf_x: (func) Marginal C.D.F. for series X.
        :param cdf_y: (func) Marginal C.D.F. for series Y.
        """
        self.cdf_x = cdf_x
        self.cdf_y = cdf_y

    def update_probabilities(self, x_value: float, y_value: float):
        """
        Update latest probabilities (p1,p2) values from empirical `x_value` and `y_value`,
        where
            p1=self.copula.get_condi_prob(self.cdf_x(x_value), self.cdf_y(y_value)),
            p2=self.copula.get_condi_prob(self.cdf_y(y_value), self.cdf_x(x_value)),

        As a result, updated probabilities are stored in `self.current_probabilities` and previous probabilities are
        stored in `self.prev_probabilities`. These containers are used to check entry/exit signals.
        """
        if self.copula is None:
            raise ValueError('Copula object was not set! Use `self.set_copula()` first.')

        if self.cdf_x is None or self.cdf_y is None:
            raise ValueError('CDF x or y is not set. Use `self.set_cdf()` first.')

        p1 = self.copula.get_condi_prob(u=self.cdf_x(x_value), v=self.cdf_y(y_value))
        p2 = self.copula.get_condi_prob(u=self.cdf_y(y_value), v=self.cdf_x(x_value))

        self.prev_probabilities = (self.current_probabilities[0], self.current_probabilities[1])
        self.current_probabilities = (p1, p2)

    def check_entry_signal(self) -> tuple:
        """
        Function which checks entry condition based on `self.current_probabilities`.

        - If P(U1 <= u1 | U2 = u2) <= 0.05 AND P(U2 <= u2 | U1 = u1) >= 0.95, then stock 1 is under-valued and
              stock 2 is over-valued. Thus we long the spread.
        - If P(U1 <= u1 | U2 = u2) >= 0.95 AND P(U2 <= u2 | U1 = u1) <= 0.05, then stock 2 is under-valued and
          stock 1 is over-valued. Thus we short the spread.

        :return: (tuple) Tuple of boolean entry flag and side (if entry flag is True).
        """

        # Long entry
        if self.current_probabilities[0] <= self.open_probabilities[0] and self.current_probabilities[1] >= \
                self.open_probabilities[1]:
            side = 1
            return True, side
        # Short entry
        if self.current_probabilities[0] >= self.open_probabilities[1] and self.current_probabilities[1] <= \
                self.open_probabilities[0]:
            side = -1
            return True, side
        return False, None

    def add_trade(
            self,
            start_timestamp: pd.Timestamp,
            side_prediction: int,
            uuid: UUID = None,
    ):
        """
        Adds a new trade to track. Calculates trigger timestamp.

        :param start_timestamp: (pd.Timestamp) Timestamp of the future label.
        :param side_prediction: (int) External prediction for the future label.
        :param uuid: (str) Unique identifier used to link label to tradelog action.
        """

        self.open_trades[start_timestamp] = {
            'uuid': uuid,
            'side': side_prediction,
            'latest_update_timestamp': start_timestamp,
            'initial_proba': self.current_probabilities
        }

    def _check_who_exits(self) -> Dict[int, bool]:
        """
        Check exit flags for longs and shorts.

        :return: (dict) Dict of {-1: True/False, 1: True/False}.
        """
        lower_exit_threshold = self.exit_probabilities[0]
        upper_exit_threshold = self.exit_probabilities[1]

        # Check if there are any crossings.
        prob_u1_up = (self.prev_probabilities[0] <= lower_exit_threshold
                      and self.current_probabilities[0] >= upper_exit_threshold)  # Prob u1 crosses upward
        prob_u1_down = (self.prev_probabilities[0] >= upper_exit_threshold
                        and self.current_probabilities[0] <= lower_exit_threshold)  # Prob u1 crosses downward
        prob_u2_up = (self.prev_probabilities[1] <= lower_exit_threshold
                      and self.current_probabilities[1] >= upper_exit_threshold)  # Prob u2 crosses upward
        prob_u2_down = (self.prev_probabilities[1] >= upper_exit_threshold
                        and self.current_probabilities[1] <= lower_exit_threshold)  # Prob u2 crosses downward

        # Check at this step which variable crossed the band.
        exit_dict = {-1: False, 1: False}
        exit_dict[-1] = prob_u1_up and prob_u2_down if self.exit_rule == 'and' else prob_u1_up or prob_u2_down
        exit_dict[1] = prob_u2_up and prob_u1_down if self.exit_rule == 'and' else prob_u2_up or prob_u1_down
        return exit_dict

    def update_trades(self, update_timestamp: pd.Timestamp) -> list:
        """
        Checks whether any of the thresholds are triggered and currently open trades should be closed. Before using the
        method, one should have called `self.update_probabilities()` to update recent probalities.

        :param update_timestamp: (pd.Timestamp) New timestamp to check vertical threshold.
        :return: (list) of closed trades.
        """

        formed_trades_uuid = []  # Array of trades formed (uuid)
        to_close = {}  # Trades to close.
        exit_flag = self._check_who_exits()
        for timestamp, data in self.open_trades.items():
            data['latest_update_timestamp'] = update_timestamp
            if exit_flag[data['side']] is True:
                to_close[timestamp] = data

        if len(to_close) != 0:
            for timestamp, data in to_close.items():
                label_data = {'t1': update_timestamp, 'exit_proba': self.current_probabilities,
                              'uuid': data['uuid'], 'side': data['side'],
                              'initial_proba': data['initial_proba']}
                formed_trades_uuid.append(data['uuid'])
                self.closed_trades[timestamp] = label_data

            self.open_trades = {
                timestamp: data
                for timestamp, data in self.open_trades.items()
                if timestamp not in to_close
            }

        return formed_trades_uuid
