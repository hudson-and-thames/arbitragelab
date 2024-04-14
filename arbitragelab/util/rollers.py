"""
This module implements the (Crude Oil Future, NBP Future, RBOB Future, Grain Based Futures, Ethanol Future) Rollers.
"""

import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes

from arbitragelab.util.base_futures_roller import BaseFuturesRoller
from arbitragelab.util import segment

class CrudeOilFutureRoller(BaseFuturesRoller):
    """
    Rolls the contract data provided under the assumption
    trading terminates 3 business day prior to the 25th calendar day
    of the month prior to the contract month. If the 25th calendar
    day is not a business day, trading terminates 4 business days prior to
    the 25th calendar day of the month prior to the contract month.

    *NOTE*
    If you have reasons to believe that the termination dates / expiration
    method changed throughout the time series, direct your attention to:

    https://www.cmegroup.com/tools-information/advisory-archive.html
    """

    def _get_rolldates(self, dataset: pd.DataFrame) -> pd.Series:
        """
        The implementation method for the rolling procedure for the CL future.

        :param dataset: (pd.DataFrame) Future price data.
        :return: (pd.Series) Target roll dates.
        """

        # Get all monthly 25ths in the date range specified in the dataset
        cl_final_df = super().get_all_possible_static_dates_in(dataset.index, 25)

        # Remove duplicate dates from the list
        twnty_fives = cl_final_df['target_date'].drop_duplicates()

        working_frame = pd.DataFrame(twnty_fives)
        working_frame['is_in_original_index'] = pd.Series(twnty_fives.isin(cl_final_df['original_index']))

        futures_df_index = list(dataset.index)

        # Get all 25ths that have occured (ie. have price data), and get 2 days prior to them.
        dates_that_occurred_mask = np.equal(working_frame['is_in_original_index'], True)
        dates_that_occurred = working_frame[dates_that_occurred_mask]
        roll_over_dates_for_business_days = [futures_df_index.index(i)-2 for i in dates_that_occurred['target_date'].values]

        # Get all 25ths that did not occur (ie. they were holidays so no price data for that day).
        dates_not_occurred_mask = np.equal(working_frame['is_in_original_index'], False)
        dates_not_occurred = working_frame[dates_not_occurred_mask]
        roll_over_dates_on_holidays = dates_not_occurred['target_date']

        roll_over_dates_for_holidays = []

        # Get x business days prior to (non business day) 25th that are available in the dataset.
        for date in roll_over_dates_on_holidays:
            roll_over_dates_for_holidays.append(
                super().get_x_days_prior_missing_target_date(dataset, date.to_period('M'), 3))

        roll_over_dates_for_holidays = np.ravel(roll_over_dates_for_holidays)

        all_roll_overs = pd.concat([pd.Series(roll_over_dates_for_holidays),
                                    pd.Series(dataset.iloc[roll_over_dates_for_business_days].index.values)])

        segment.track('CrudeOilFutureRoller')

        return all_roll_overs.sort_values().values


class NBPFutureRoller(BaseFuturesRoller):
    """
    Rolls the contract data provided under the assumption that
    the termination date is the penultimate business day of the month.
    """

    def _get_rolldates(self, dataset: pd.DataFrame) -> pd.Series:
        """
        The implementation method for the rolling procedure for the NBP future.

        :param dataset: (pd.DataFrame) Future price data.
        :return: (pd.Series) Target roll dates.
        """

        target_dates = super().get_available_date_per_month_from(dataset)

        final_df = super().get_x_days_prior_available_target_date(1, dataset.index.to_series(),
                                                                  target_dates['target_date'])

        nbp_roll_dates = final_df['expiry'].drop_duplicates().dropna().values[:-1]

        segment.track('NBPFutureRoller')

        return nbp_roll_dates


class RBFutureRoller(BaseFuturesRoller):
    """
    Rolls the contract data provided under the assumption that
    the termination date is the last business day of the month.

    The following contracts can be used under this class.

    Refined Products
    - FO - Trading terminates on the last London business day of the contract month.
    - 7K - Trading terminates on the last business day of the contract month.
    - ME - Trading shall cease on the last business day of the contract month.
    - 1L - Trading terminates on the last business day of the contract month.
    - LT - Trading shall cease on the last business day of the contract month.
    - HO - Trading terminates on the last business day of the month prior to the contract month.
    - M1B - Trading terminates on the last business day of the contract month.
    - M35 - Trading terminates on the last business day of the contract month.
    - NYA - Trading terminates on the last business day of the contract month
    - NYR - Trading terminates on the last business day of the contract month
    - RBOB - Trading terminates on the last business day of the month prior to the contract month.

    Crude Oil
    - LLB - Trading terminates on the last business day of the contract month.
    - LWB - Trading terminates on the last business day of the contract month.
    - WJ - Trading terminates on the last business day of the contract month.
    - CS - Trading shall cease on the last business day of the contract month.
    - BK - Trading shall cease on the last business day of the contract month.

    BioFules
    - CU - Trading terminates on the last business day of the contract month.
    """

    def _get_rolldates(self, dataset: pd.DataFrame) -> pd.Series:
        """
        The implementation method for the rolling procedure for the RB future.

        :param dataset: (pd.DataFrame) Future price data.
        :return: (pd.Series) Target roll dates.
        """

        target_dates = super().get_available_date_per_month_from(dataset)

        rb_roll_dates = target_dates['target_date'].drop_duplicates()

        segment.track('RBFutureRoller')

        return rb_roll_dates.dropna().values[1:]


class GrainFutureRoller(BaseFuturesRoller):
    """
    Rolls the contract data provided under the assumption that
    the termination date is the 15th of each month.

    The following contracts can be used under this class.

    - S - Soybean
    - B0 - Soyoil
    - C - Corn
    """

    def _get_rolldates(self, dataset: pd.DataFrame) -> pd.Series:
        """
        The implementation method for the rolling procedure for Grain futures.

        :param dataset: (pd.DataFrame) Future price data.
        :return: (pd.Series) Target roll dates.
        """

        possible_dates_df = super().get_all_possible_static_dates_in(dataset.index, 15)

        # Remove duplicate dates from the list
        unique_selected_dates_df = possible_dates_df['target_date'].drop_duplicates()

        working_frame = pd.DataFrame(unique_selected_dates_df)
        working_frame['is_in_original_index'] = pd.Series(unique_selected_dates_df.isin(possible_dates_df['original_index']))

        futures_df_index = list(dataset.index)

        # Get all 15ths that have occured (ie. have price data), and get 1 day prior to them.
        dates_that_occurred_mask = np.equal(working_frame['is_in_original_index'], True)
        dates_that_occurred = working_frame[dates_that_occurred_mask]
        roll_over_dates_for_business_days = [futures_df_index.index(i)-1 for i in dates_that_occurred['target_date'].values]

        # Get all 15ths that did not occur (ie. they were holidays so no price data for that day).
        dates_not_occurred_mask = np.equal(working_frame['is_in_original_index'], False)
        dates_not_occurred = working_frame[dates_not_occurred_mask]
        roll_over_dates_on_holidays = dates_not_occurred['target_date']

        roll_over_dates_for_holidays = []

        # Get x business days prior to (non business day) 15th that are available in the dataset.
        for date in roll_over_dates_on_holidays:
            roll_over_dates_for_holidays.append(
                super().get_x_days_prior_missing_target_date(dataset, date.to_period('M'), 1, 15))

        roll_over_dates_for_holidays = np.ravel(roll_over_dates_for_holidays)

        all_roll_overs = pd.concat([pd.Series(roll_over_dates_for_holidays),
                                    pd.Series(dataset.iloc[roll_over_dates_for_business_days].index.values)])

        segment.track('GrainFutureRoller')

        return all_roll_overs.sort_values().values


class EthanolFutureRoller(BaseFuturesRoller):
    """
    Rolls the contract data provided under the assumption that
    the termination date is the 3rd business day of each month.
    """

    def _get_rolldates(self, dataset: pd.DataFrame) -> pd.Series:
        """
        The implementation method for the rolling procedure for the Ethanol future.

        :param dataset: (pd.DataFrame) Future price data.
        :return: (pd.Series) Target roll dates.
        """

        possible_dates_df = super().get_all_possible_static_dates_in(dataset.index, 3)

        # Remove duplicate dates from the list
        unique_selected_dates_df = possible_dates_df['target_date'].drop_duplicates()

        working_frame = pd.DataFrame(unique_selected_dates_df)
        working_frame['is_in_original_index'] = pd.Series(unique_selected_dates_df.isin(possible_dates_df['original_index']))

        futures_df_index = list(dataset.index)

        # Get all 3rds that have occured (ie. have price data), and get 1 days prior to them.
        dates_that_occurred_mask = np.equal(working_frame['is_in_original_index'], True)
        dates_that_occurred = working_frame[dates_that_occurred_mask]
        roll_over_dates_for_business_days = [futures_df_index.index(i) for i in dates_that_occurred['target_date'].values]

        # Get all 3rds that did not occur (ie. they were holidays so no price data for that day).
        dates_not_occurred_mask = np.equal(working_frame['is_in_original_index'], False)
        dates_not_occurred = working_frame[dates_not_occurred_mask]
        roll_over_dates_on_holidays = dates_not_occurred['target_date']

        roll_over_dates_for_holidays = []

        # Get x business days prior to (non business day) 3rd that are available in the dataset.
        for date in roll_over_dates_on_holidays:
            roll_over_dates_for_holidays.append(
                super().get_x_days_prior_missing_target_date(dataset, date.to_period('M'), 2, 3))

        roll_over_dates_for_holidays = np.hstack(roll_over_dates_for_holidays)

        all_roll_overs = pd.concat([pd.Series(roll_over_dates_for_holidays),
                                    pd.Series(dataset.iloc[roll_over_dates_for_business_days].index.values)])

        segment.track('EthanolFutureRoller')

        return all_roll_overs.sort_values().values

def plot_historical_future_slope_state(m1_last: pd.Series, m2_open: pd.Series) -> Axes:
    """
    Plots a historical plot of the contango/backwardation states between two
    contracts.

    :param m1_last: (pd.Series) The 'close' price vector for the month one contract.
    :param m2_last: (pd.Series) The 'open' price vector for the month two contract.
    :return: (Axes) Axes object.
    """

    premium = (m1_last - m2_open)
    perc_chg = ((premium/m1_last)*100)

    ax_object = perc_chg.plot(figsize=(15, 10), alpha=0)
    ax_object.fill_between(premium.index, perc_chg, where=perc_chg > 0, facecolor='green')
    ax_object.fill_between(premium.index, perc_chg, where=perc_chg < 0, facecolor='red')
    ax_object.legend(["", "Contango", "Backwardation"])

    segment.track('plot_historical_future_slope_state')

    return ax_object
