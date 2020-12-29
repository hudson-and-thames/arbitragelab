# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the RBOB Futures Roller.
"""

import pandas as pd
from arbitragelab.util.base_futures_roller import BaseFuturesRoller


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

        :param dataset: (pd.DataFrame)
        :return: (pd.Series)
        """

        target_dates = super().get_available_date_per_month_from(dataset)

        rb_roll_dates = target_dates['target_date'].drop_duplicates()

        return rb_roll_dates.dropna().values[1:]
