# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This module Threshold Filter described in Dunis et al. (2005).
"""

import pandas as pd


class ThresholdFilter:
    """
    Threshold Filter implementation.
    """

    def __init__(self, buy_threshold: float, sell_threshold: float):
        """
        Initialization of trade parameters.

        :param buy_threshold: (float) If larger than this value, buy.
        :param sell_threshold: (float) If smaller than this value, sell.
        """

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.frame = None

    def fit(self, frame: pd.DataFrame):
        """
        Sets the time series to be analyzed inside of the class object.

        :param frame: (pd.DataFrame) Time series to be analyzed.
        """

        self.frame = frame.copy().to_frame()

        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a side column to describe the signal that was detected at
        that point, based on the parameters given in the constructor.

        :param frame: (pd.DataFrame) Time series to be analyzed.
        :return: (pd.DataFrame) Time series augmented with the trade side
            information.
        """

        init_frame = self.frame

        buy_signal = init_frame.index.isin(
            frame[frame < self.buy_threshold].index)
        sell_signal = init_frame.index.isin(
            frame[frame > self.sell_threshold].index)

        init_frame['side'] = 0
        init_frame.loc[buy_signal, 'side'] = 1
        init_frame.loc[sell_signal, 'side'] = -1
        init_frame['side'] = init_frame['side'].shift(1)

        return init_frame

    def fit_transform(self, frame) -> pd.DataFrame:
        """
        Convenience method that executes the fit and transform method
        in sequence.

        :param frame: (pd.DataFrame) Time series to be analyzed.
        :return: (pd.DataFrame) Time series augmented with the trade side
            information.
        """

        return self.fit(frame).transform(frame)
