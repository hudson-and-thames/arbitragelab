# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
This module is used to split datasets inot train and test parts.
"""

from typing import Tuple, Optional
import pandas as pd
import warnings


def train_test_split(price_df: pd.DataFrame, date_cutoff: Optional[pd.Timestamp] = None,
                     num_cutoff: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the price series into a training set to calculate the cointegration coefficient, beta,
    and a test set to simulate the trades to check trade frequency and PnL.

    Set both cutoff to none to perform an in-sample test.

    :param price_df: (pd.DataFrame) The source price dataframe for splitting.
    :param date_cutoff: (pd.Timestamp) If the price series has a date index then this will be used for split.
    :param num_cutoff: (int) Number of periods to include in the training set (could be used for any type of index).
    :return: (pd.DataFrame, pd.DataFrame) Training set price series; test set price series.
    """
    # If num_cutoff is not None, then date_cutoff should be ignored
    if num_cutoff is not None:
        warnings.warn("Already defined the number of data points included in training set. Date cutoff will be ignored.")
        train_series = price_df.iloc[:num_cutoff, :]
        test_series = price_df.iloc[num_cutoff:, :]
        return train_series, test_series

    # Both cutoff is None, do in-sample test. So training set and test set are the same.
    if date_cutoff is None:
        return price_df, price_df

    # Verify the index is indeed pd.DatetimeIndex
    assert price_df.index.is_all_dates, "Index is not of pd.DatetimeIndex type."

    # Make sure the split point is in between the time range of the data
    min_date = price_df.index.min()
    max_date = price_df.index.max()

    assert min_date < date_cutoff < max_date, "Date split point is not within time range of the data."

    train_series = price_df.loc[:date_cutoff]
    test_series = price_df.loc[date_cutoff:]

    return train_series, test_series
