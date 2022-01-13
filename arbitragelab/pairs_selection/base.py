# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Abstract pair selector class.
"""

from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes


class AbstractPairsSelector(ABC):
    """
    This is an abstract class for pairs selectors objects. It has abstract method select_pairs() which needs to be
    implemented.
    """

    @abstractmethod
    def select_pairs(self):
        """
        Method which selects pairs based on some predefined criteria.
        """

        raise NotImplementedError('Must implement select_pairs() method.')

    @staticmethod
    def _convert_to_tuple(arr: np.array) -> tuple:
        """
        Returns a list converted to a tuple.

        :param arr: (np.array) Input array to be converted.
        :return: (tuple) List converted to tuple.
        """

        return tuple(i for i in arr)

    def describe_pairs_sectoral_info(self, leg_1: list, leg_2: list,
                                     sectoral_info_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns information on each pair selected.

        The following statistics are included - both legs of the pair, cointegration (t-value, p-value, hedge_ratio),
        hurst_exponent, half_life, no_mean_crossovers.

        :param leg_1: (list) Vector of asset names.
        :param leg_2: (list) Vector of asset names.
        :param sectoral_info_df: (pd.DataFrame) DataFrame with two columns [ticker, sector] to be used in the output.
        :return: (pd.DataFrame) DataFrame of pair sectoral statistics.
        """

        leg_1_info = self._loop_through_sectors(leg_1, sectoral_info_df)
        leg_2_info = self._loop_through_sectors(leg_2, sectoral_info_df)

        info_df = pd.concat([leg_1_info, leg_2_info], axis=1)
        info_df.columns = ['Leg 1 Ticker', 'Industry', 'Sector',
                           'Leg 2 Ticker', 'Industry', 'Sector']

        return info_df

    def _loop_through_sectors(self, tickers: list, sectoral_info_df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method that loops through sectoral info.

        :param tickers: (list) Vector of asset names.
        :param sectoral_info_df: (pd.DataFrame) DataFrame with two columns [ticker, sector] to be used in the output.
        :return: (pd.DataFrame) DataFrame of ticker sectoral statistics.
        """

        tck_info = []

        for tck in tickers:
            leg_sector_info = sectoral_info_df[sectoral_info_df['ticker'] == tck]
            if leg_sector_info.empty:
                tck_info.append(('', '', ''))
            else:
                info_as_tuple = self._convert_to_tuple(leg_sector_info.values[0])
                tck_info.append(info_as_tuple)

        return pd.DataFrame(tck_info)

    def plot_selected_pairs(self) -> list:
        """
        Plots the final selection of pairs.

        :return: (list) List of Axes objects.
        """

        if (self.final_pairs is None) or (len(self.final_pairs) == 0):
            raise Exception("The needed pairs have not been computed yet.",
                            "Please run criterion_selector() before this method.")

        if len(self.final_pairs) > 40:
            raise Exception("The amount of pairs to be plotted cannot exceed 40",
                            "without causing system instability.")

        _, axs = plt.subplots(len(self.final_pairs), figsize=(15, 3 * len(self.final_pairs)))

        for ax_object, frame in zip(axs, self.final_pairs.index.values):
            rets_asset_one = np.log(self.prices_df.loc[:, frame[0]]).diff()
            rets_asset_two = np.log(self.prices_df.loc[:, frame[1]]).diff()

            ax_object.plot(rets_asset_one.cumsum())
            ax_object.plot(rets_asset_two.cumsum())
            ax_object.legend([frame[0], frame[1]])

        return axs

    def plot_single_pair(self, pair: tuple) -> Axes:
        """
        Plots the given pair.

        :param pair: (tuple) Tuple of asset names.
        :return: (Axes) Axes object.
        """

        _, ax_object = plt.subplots(1, figsize=(15, 3))

        rets_asset_one = np.log(self.prices_df.loc[:, pair[0]]).diff()
        rets_asset_two = np.log(self.prices_df.loc[:, pair[1]]).diff()

        ax_object.plot(rets_asset_one.cumsum())
        ax_object.plot(rets_asset_two.cumsum())
        ax_object.legend([pair[0], pair[1]])

        return ax_object

    @staticmethod
    def _print_progress(iteration, max_iterations, prefix='', suffix='', decimals=1, bar_length=50):
        # pylint: disable=expression-not-assigned
        """
        Calls in a loop to create a terminal progress bar.
        https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

        :param iteration: (int) Current iteration.
        :param max_iterations: (int) Maximum number of iterations.
        :param prefix: (str) Prefix string.
        :param suffix: (str) Suffix string.
        :param decimals: (int) Positive number of decimals in percent completed.
        :param bar_length: (int) Character length of the bar.
        """

        str_format = "{0:." + str(decimals) + "f}"
        # Calculate the percent completed.
        percents = str_format.format(100 * (iteration / float(max_iterations)))
        # Calculate the length of bar.
        filled_length = int(round(bar_length * iteration / float(max_iterations)))
        # Fill the bar.
        block = '█' * filled_length + '-' * (bar_length - filled_length)
        # Print new line.
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, block, percents, '%', suffix)),

        if iteration == max_iterations:
            sys.stdout.write('\n')
        sys.stdout.flush()
