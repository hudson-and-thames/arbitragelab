# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the Crude Oil Futures Roller.
"""

import numpy as np
import pandas as pd
from arbitragelab.util.base_futures_roller import BaseFuturesRoller


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

        :param dataset: (pd.DataFrame)
        :return: (pd.Series)
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

        return all_roll_overs.sort_values().values
