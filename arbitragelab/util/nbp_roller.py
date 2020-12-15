# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the NBP Futures Roller.
"""

import pandas as pd
from arbitragelab.util.base_futures_roller import BaseFuturesRoller


class NBPFutureRoller(BaseFuturesRoller):
    """
    Rolls the contract data provided under the assumption that
    the termination date is the penultimate business day of the month.
    """

    def _get_rolldates(self, dataset: pd.DataFrame) -> pd.Series:
        """

        :param dataset: (pd.DataFrame)
        :return: (pd.Series)
        """

        target_dates = super().get_available_date_per_month_from(dataset)#, 'first')

        final_df = super().get_x_days_prior_available_target_date(1,
                                                                  dataset.index.to_series(), target_dates['target_date'])

        nbp_roll_dates = final_df['expiry'].drop_duplicates(
        ).dropna().values[:-1]

        return nbp_roll_dates
