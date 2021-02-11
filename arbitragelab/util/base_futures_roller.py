# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the base Futures Roller.
"""

import numpy as np
import pandas as pd

class BaseFuturesRoller:
    """
    Basic Futures Roller implementation.
    """

    def __init__(self, open_col: str = "PX_OPEN", close_col: str = "PX_LAST"):
        """
        Initialization of variables.

        :param open_col: (str) Name of the column with the open price.
        :param close_col: (str) Name of the column with the close price.
        """

        self.open_col = open_col
        self.close_col = close_col
        self.dataset = None
        self.diagnostic_frame = None

    def fit(self, dataset: pd.DataFrame):
        """
        Stores price data in the class object for later processing.

        :param dataset: (pd.DataFrame) Future price data.
        """

        self.dataset = dataset

        return self

    def diagnostic_summary(self) -> pd.DataFrame:
        """
        After the dataset has been transformed, a dataframe with all the dates
        of each roll operation is stored in the class object. This will return
        that dataframe.

        :return: (pd.DataFrame) Returns DataFrame with each roll and gap size.
        """

        return self.diagnostic_frame

    def transform(self, roll_forward: bool = True, handle_negative_roll: bool = False) -> pd.Series:
        """
        Processes the dataset provided with the set roll dates.

        :param roll_forward: (bool) The direction which the gaps should sum to.
        :param handle_negative_roll: (bool) Process to remove negative values from series.
        :return: (pd.Series) Series of gaps or Preprocessed rolled series.
        """

        roll_dates = self._get_rolldates(self.dataset)

        gaps_series, diagnostic_frame = self.roll(self.dataset, roll_dates, roll_forward)

        self.diagnostic_frame = diagnostic_frame

        if handle_negative_roll:
            rolled_series = self.dataset[self.close_col] - gaps_series

            return self.non_negativeize(rolled_series, self.dataset[self.close_col])

        return gaps_series

    def generate_diagnostic_frame(self, dataset: pd.DataFrame, termination_date_indexes: np.array,
                                  following_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Returns dataframe full of every roll operation and its gap size.

        :param dataset: (pd.DataFrame) Price data.
        :param termination_date_indexes: (np.array) Indexes of termination dates.
        :param following_dates: (pd.DateTimeIndex) Dates following the termination dates.
        :return: (pd.DataFrame) List of dates and their gap's size.
        """

        diag_df = pd.DataFrame({})
        diag_df['last_on_termination_date'] = dataset[self.close_col].iloc[termination_date_indexes].index
        diag_df['last_prices'] = dataset[self.close_col].iloc[termination_date_indexes].values
        diag_df['open_day_after_termination_date'] = following_dates
        diag_df['open_prices'] = dataset[self.open_col].loc[following_dates].values

        day_before_exp_vals = dataset[self.open_col].loc[following_dates].values

        day_after_exp_vals = dataset[self.close_col].iloc[termination_date_indexes].values

        diag_df['gap'] = day_before_exp_vals - day_after_exp_vals

        return diag_df

    @staticmethod
    def get_x_days_prior_available_target_date(n_days: int, dataset_datetime_index: pd.DatetimeIndex,
                                               target_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Gets x days prior to target date(that must be available in the dataset index).

        :param n_days: (int) Number of days to be shifted by
        :param dataset_datetime_index: (pd.DateTimeIndex) All dates that occur in the dataset
        :param target_dates: (pd.DateTimeIndex) Dates used as the start for the shift. Important(!)
            these dates need to exist in 'dataset_datetime_index'.
        :return: (pd.Series) Price series x days prior to target date.
        """

        price_series = pd.DataFrame()
        indexed_list = list(dataset_datetime_index)
        indexed_list = [indexed_list.index(i)-n_days for i in target_dates.dropna()]
        price_series['expiry'] = dataset_datetime_index.iloc[indexed_list]

        return price_series

    @staticmethod
    def get_available_date_per_month_from(price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Gets first available day per month from the dataset.

        :param price_df: (pd.DataFrame) Original prices dataframe.
        :return: (pd.DataFrame) Frame with first available days per month.
        """

        price_series = pd.DataFrame()
        price_series['dates'] = price_df.index
        price_series['expiry_month'] = price_df.index.to_period('M')
        price_series['day'] = price_df.index.to_period('D').day

        price_by_exp_month = price_series.groupby(by=['expiry_month'])
        price_series['target_day'] = price_by_exp_month['day'].transform(min)

        exp_month = price_series['expiry_month'].astype(str)
        exp_day = price_series['target_day'].astype(str)
        price_series['target_date'] = exp_month + "-" + exp_day

        price_series['target_date'] = pd.to_datetime(price_series['target_date'],
                                                     errors='coerce')

        return price_series

    @staticmethod
    def get_all_possible_static_dates_in(dataset_datetime_index: pd.DatetimeIndex,
                                         day_of_month: int = 25) -> pd.DataFrame:
        """
        Gets a series of static dates that could happen in a specific time range, described
        by the input 'dataset_datetime_index'. The dates returned from this method do not take
        into consideration the fact of if the specified date happened in the dataset or not.

        :param dataset_datetime_index: (pd.DateTimeIndex) All dates that occur in the dataset.
        :param day_of_month: (int) Day of month.
        :return: (pd.DataFrame) Dates.
        """

        price_series = pd.DataFrame()
        price_series['original_index'] = dataset_datetime_index
        price_series['expiry_month'] = dataset_datetime_index.to_period('M')
        price_series['day'] = day_of_month

        exp_month = price_series['expiry_month'].astype(str)
        exp_day = price_series['day'].astype(str)
        price_series['target_date'] = exp_month + "-" + exp_day

        price_series['target_date'] = pd.to_datetime(price_series['target_date'],
                                                     errors='coerce')

        return price_series[['original_index', 'target_date']]

    @staticmethod
    def get_x_days_prior_missing_target_date(dataset: pd.DataFrame, working_month_delta: pd.Period,
                                             days_prior: int = 3, target_day: int = 25):
        """
        This method will get x available day prior to a specific date, in the special case that that
        date doesn't exist in the dataset.

        :param dataset: (pd.DataFrame) Price data.
        :param working_month_delta: (pd.Period) specific Year-Month delta, ex. 1995-12
        :param days_prior: (int) Days prior the newly found target date.
        :param target_day: (int) The target day to be used as a starting point.
        :return: (np.array) Target Dates.
        """

        index_to_monthly = pd.to_datetime(dataset.index).to_period('M')

        # Use delta to get all days in that month from the original series.
        full_working_month = dataset.loc[index_to_monthly == working_month_delta]

        # Remove all days including and after target_date
        full_working_month = full_working_month[full_working_month.index.day < target_day]

        # Get Bottom x days and select the first one of them
        return full_working_month.last(str(days_prior) + 'd').first('d').index.values

    def roll(self, dataset: pd.DataFrame, roll_dates: pd.Series, match_end: bool = True):
        """
        Deals with single futures contracts. Forms a series of cumulative
        roll gaps and returns a diagnostic dataframe with dates of rolls for
        further analysis.

        :param dataset: (pd.DataFrame) Price data.
        :param roll_dates: (pd.Series) Specific dates that the roll will initialized on.
        :param match_end: (bool) Defines from where to match the gaps.
        :return: (pd.Series) (pd.DataFrame) Returns gaps and diagnostic frame.
        """

        gaps = dataset[self.close_col]*0

        iloc = list(dataset.index)

        iloc = [iloc.index(i)-1 for i in roll_dates]

        day_before_exp_vals = dataset[self.open_col].loc[roll_dates]

        day_after_exp_vals = dataset[self.close_col].iloc[iloc].values

        gaps.loc[roll_dates] = day_before_exp_vals - day_after_exp_vals

        gaps = gaps.cumsum().dropna()

        if match_end:
            gaps -= gaps.iloc[-1]

        return gaps, self.generate_diagnostic_frame(dataset, iloc, roll_dates)

    @staticmethod
    def non_negativeize(rolled_series: pd.Series, raw_series: pd.Series) -> pd.Series:
        """
        In general, non negative series are preferred over negative series, which could
        easily occur particularly if the contract sold off while in contango. The method
        employed here works as follows; the return as rolled price change is divided by
        the raw price and then a series is formed using those returns, ex (1+r).cumprod().

        This method has been described originally in De Prado, M.L., 2018. Advances in
        financial machine learning. John Wiley & Sons.

        :param rolled_series: (pd.Series)
        :param raw_series: (pd.Series)
        :return: (pd.Series)
        """

        new_prices_series = rolled_series.copy()
        new_prices_series = rolled_series.diff() / raw_series.shift(1)
        new_prices_series = (1+new_prices_series).cumprod()

        return new_prices_series
