# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Module for signal generation for Multivariate Cointegration Trading strategy.
"""

import warnings
from collections import deque
from uuid import UUID

import numpy as np
import pandas as pd

from arbitragelab.util import segment


class MultivariateCointegrationTradingRule:
    """
    This class implements trading strategy from the Multivariate Contegration method from the paper
    by Galenko, A., Popova, E. and Popova, I. in
    `"Trading in the presence of cointegration" <http://www.ntuzov.com/Nik_Site/Niks_files/Research/papers/stat_arb/Galenko_2007.pdf>`_

    The strategy generates a signal based on the notional values from the MultivariateCointegration
    class, in particular the number of shares to go long and short per each asset in a portfilio.

    It's advised to re-estimate the cointegration vector (i.e. re-run the MultivariateCointegration)
    each month.

    The strategy rebalances the portfolio of assets with each new entry, meaning that the opened at time t
    should be closed at time t+1, and the new trade should be opened.

    This strategy allows only one open trade at a time.
    """

    def __init__(self, coint_vec: np.array, nlags: int = 30,
                 dollar_invest: float = 1.e7):
        """
        Class constructor.

        :param coint_vec: (np.array) Cointegration vector, b.
        :param nlags: (int) Amount of lags for cointegrated returns sum, corresponding to the parameter P in the paper.
        :param dollar_invest: (float) The value of long/short positions, corresponding to the parameter C in the paper.
        """

        segment.track('MultivariateCointegrationTradingRule')

        self.open_trades = {}
        self.closed_trades = {}

        self.coint_vec = coint_vec
        self.pos_coef_asset = self.coint_vec[self.coint_vec >= 0]
        self.neg_coef_asset = self.coint_vec[self.coint_vec < 0]
        self.nlags = nlags
        self.dollar_invest = dollar_invest

        self.price_series = pd.DataFrame()
        self.trade = 0  # Current trade status

    def update_price_values(self, latest_price_values: pd.Series):
        """
        Adds latest price values of assets to `self.price_series`.

        :param latest_price_values: (pd.Series) Latest price values.
        """

        # Update the training dataframe and keep the last nlags values
        #print('AAA')
        #print(type(self.price_series))
        #print(type(latest_price_values))
        #print(self.price_series)
        #print(latest_price_values)


        self.price_series = self.price_series.append(latest_price_values)
        self.price_series = self.price_series.iloc[-self.nlags:]

    @staticmethod
    def calc_log_price(price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the log price of each asset for position size calculation.

        :param price_df: (pd.DataFrame) Dataframe that contains the raw asset price.
        :return: (pd.DataFrame) Log prices of the assets.
        """

        # Return log price
        return price_df.apply(np.log)

    def get_signal(self) -> tuple:
        """
        Function which calculates the number of shares to trade in the current timestamp based on
        the price changes and cointegration vector from the MultivariateCointegration class.

        :return: (np.array, np.array, np.array, np.array) The number of shares to trade;
            the notional values of positions.
        """

        # Get last price of assets
        if self.price_series.empty:
            warnings.warn("No price series found. Please first use update_price_values.")
            return

        last_price = self.price_series.iloc[-1]

        # Generate parameters for a strategy
        # Calculate log prices
        log_price = self.calc_log_price(self.price_series)

        # Calculate the cointegration error Y_t, recover the date index
        coint_error = np.dot(log_price, self.coint_vec)
        coint_error_df = pd.DataFrame(coint_error)
        coint_error_df.index = log_price.index

        # Calculate the return Z_t by taking the difference. Drop the NaN of the first data point
        realization = coint_error_df.diff().dropna()

        # Calculate the direction of the trade
        sign = np.sign(realization.sum()).values[0]

        # Calculate notional values
        pos_notional = self.pos_coef_asset * sign * self.dollar_invest / self.pos_coef_asset.sum()
        neg_notional = self.neg_coef_asset * sign * self.dollar_invest / self.neg_coef_asset.sum()

        # Calculate number of shares
        pos_shares = pos_notional / last_price[self.pos_coef_asset.index]
        neg_shares = neg_notional / last_price[self.neg_coef_asset.index]

        # Calculate actual notional values due to rounding
        pos_notional = np.floor(pos_shares) * last_price[self.pos_coef_asset.index]
        neg_notional = np.floor(neg_shares) * last_price[self.neg_coef_asset.index]

        # Assign the correct sign to the number of shares according to the sign of CC
        return -1. * np.floor(pos_shares), np.floor(neg_shares), -1. * pos_notional, neg_notional

    def add_trade(
            self,
            start_timestamp: pd.Timestamp,
            pos_shares: np.array,
            neg_shares: np.array,
            uuid: UUID = None,
    ):
        """
        Adds a new trade to track.

        :param start_timestamp: (pd.Timestamp) Timestamp of the future label.
        :param pos_shares: (np.array) Number of shares bought per asset.
        :param neg_shares: (np.array) Number of shares sold per asset.
        :param uuid: (str) Unique identifier used to link label to tradelog action..
        """

        self.open_trades[start_timestamp] = {
            'start_prices': self.price_series.iloc[-1],
            'price_series': list(self.price_series),
            'uuid': uuid,
            'pos_shares': pos_shares,
            'neg_shares': neg_shares,
            'latest_update_timestamp': start_timestamp
        }

    def update_trades(self, update_timestamp: pd.Timestamp) -> list:
        """
        Closes previously opened trade and updates list of closed trades.

        :param update_timestamp: (pd.Timestamp) New timestamp to check vertical threshold.
        :return: (list) of closed trades.
        """

        formed_trades_uuid = []  # Array of trades formed (uuid)
        to_close = {}  # Trades to close.

        for timestamp, data in self.open_trades.items():
            data['latest_update_timestamp'] = update_timestamp
            to_close[timestamp] = data


        if len(to_close) != 0:
            for timestamp, data in to_close.items():
                label_data = {'t1': update_timestamp, 'pt': self.price_series.iloc[-1],
                              'uuid': data['uuid'], 'start_prices': data['start_prices'],
                              'end_prices': self.price_series.iloc[-1], 'pos_shares': data['pos_shares'],
                              'neg_shares': data['neg_shares']}
                formed_trades_uuid.append(data['uuid'])
                self.closed_trades[timestamp] = label_data

            self.open_trades = {
                timestamp: data
                for timestamp, data in self.open_trades.items()
                if timestamp not in to_close
            }

        return formed_trades_uuid
