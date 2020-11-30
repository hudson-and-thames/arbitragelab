from pandas import Timestamp
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.tseries.offsets import BDay

class BaseFuturesRoller:
    
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
        
        :return: Returns DataFrame with each roll and gap size.
        """
        
        return self.diagnostic_frame
    
    def transform(self, roll_forward: bool = True, handle_negative_roll: bool = False) -> pd.Series:
        """
        Processes the dataset provided with the set roll dates.
        
        :param roll_forward: (bool) The direction which the gaps should sum to.
        :param handle_negative_roll: (bool) Process to remove negative values from series.
        :return: Series of gaps or Preprocessed rolled series.
        """
        
        roll_dates = self._get_rolldates(self.dataset)
        
        gaps_series, diagnostic_frame = self.roll(self.dataset, roll_dates, roll_forward)
        
        self.diagnostic_frame = diagnostic_frame
        
        if handle_negative_roll:
            rolled_series = self.dataset['PX_LAST'] - gaps_series
            return self.non_negativeize(rolled_series, self.dataset['PX_LAST'])
        else:
            return gaps_series
        
    def _get_rolldates(self, dataset: pd.DataFrame) -> bool:
        """
        To be overrided as needed.
        
        :return: False.
        """
        
        return False # Not Implemented.

    def generate_diagnostic_frame(self, dataset: pd.DataFrame, termination_date_indexes: np.array,
                                  following_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Returns dataframe full of every roll operation and its gap size.

        :param dataset: (pd.DataFrame) Price data. Must have columns [PX_OPEN, PX_LAST].
        :param termination_date_indexes: (np.array) Indexes of termination dates.
        :param following_dates: (pd.DateTimeIndex) Dates following the termination dates.
        :return: (pd.DataFrame) List of dates and their gap's size.
        """

        diag_df = pd.DataFrame({})
        diag_df['last_on_termination_date'] = dataset['PX_LAST'].iloc[termination_date_indexes].index
        diag_df['last_prices'] = dataset['PX_LAST'].iloc[termination_date_indexes].values
        diag_df['open_day_after_termination_date'] = following_dates
        diag_df['open_prices'] = dataset['PX_OPEN'].loc[following_dates].values

        diag_df['gap'] = dataset['PX_OPEN'].loc[following_dates].values \
                                - dataset['PX_LAST'].iloc[termination_date_indexes].values

        return diag_df


    def get_x_days_prior_available_target_date(self, n_days: int, dataset_datetime_index: pd.DatetimeIndex,
                                               target_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Gets x days prior to target date(that must be available in the dataset index).

        :param n_days: (int) Number of days to be shifted by
        :param dataset_datetime_index: (pd.DateTimeIndex) All dates that occur in the dataset
        :param target_dates: (pd.DateTimeIndex) Dates used as the start for the shift. Important
            these dates need to exist in 'dataset_datetime_index'.
        :return: (pd.Series) 
        """

        price_series = pd.DataFrame()
        indexed_list = list(dataset_datetime_index)
        indexed_list = [indexed_list.index(i)-n_days for i in target_dates.dropna()]
        price_series['expiry'] = dataset_datetime_index.iloc[indexed_list]
        
        return price_series

    def get_available_date_per_month_from(self, price_df: pd.DataFrame, mode: str,
                                          nth_day: int = 0) -> pd.DataFrame:
        """

        :param price_df: (pd.DataFrame)
        :param mode: (str) Available modes; ["first", "last", "specific"]
        :param nth_day: (int)
        :return: (pd.DataFrame)
        """

        price_series = pd.DataFrame()
        price_series['dates'] = price_df.index
        price_series['expiry_month'] = price_df.index.to_period('M')  
        price_series['day'] = price_df.index.to_period('D').day

        if mode == "first":
            price_series['target_day'] = price_series.groupby(by=['expiry_month'])['day'].transform(min)
        elif mode == "last":
            price_series['target_day'] = price_series.groupby(by=['expiry_month'])['day'].transform(max)
        elif mode == "specifc":
            price_series['target_day'] = price_series.groupby(by=['expiry_month'])['day'].nth(nth_day)
        else:
            print("Please select a valid mode!")

        price_series['target_date'] = price_series['expiry_month'].astype(str) \
                                            + "-" + price_series['target_day'].astype(str)

        price_series['target_date'] = pd.to_datetime(price_series['target_date'], errors='coerce')
        
        return price_series

    def get_all_possible_static_dates_in(self, dataset_datetime_index: pd.DatetimeIndex,
                                         day_of_month: int = 25) -> pd.DataFrame:
        """
        Gets a series of static dates that could happen in a specific time range, described
        by the input 'dataset_datetime_index'. The dates returned from this method do not take 
        into consideration the fact of if the specified date happened in the dataset or not.

        :param dataset_datetime_index: (pd.DateTimeIndex) All dates that occur in the dataset
        :param day_of_month: (int) Day of month    
        :return: (pd.DataFrame) Dates
        """

        price_series = pd.DataFrame()
        price_series['original_index'] = dataset_datetime_index
        price_series['expiry_month'] = dataset_datetime_index.to_period('M')
        price_series['day'] = day_of_month
        price_series['target_date'] = price_series['expiry_month'].astype(str) \
                                            + "-" + price_series['day'].astype(str)

        price_series['target_date'] = pd.to_datetime(price_series['target_date'], errors='coerce')
        
        return price_series[['original_index', 'target_date']]

    def get_x_days_prior_missing_target_date(self, dataset: pd.DataFrame,
                                             working_month_delta: pd.Period, days_prior: int = 3,
                                             target_day: int = 25) -> str:
        """
        This method will get x available day prior to a specific date, in the special case that that
        date doesn't exist in the dataset.

        :param dataset: (pd.DataFrame)
        :param working_month_delta: (pd.Period) specific Year-Month delta, ex. 1995-12
        :param days_prior: (int)
        :param target_day: (int)
        :return: (Date)
        """

        # Use delta to get all days in that month from the original series.
        full_working_month = dataset.loc[pd.to_datetime(dataset.index).to_period('M') == working_month_delta]

        # Remove all days including and after target_date
        full_working_month = full_working_month[full_working_month.index.day < target_day]

        # Get Bottom x days and select the first one of them
        return full_working_month.last(str(days_prior) + 'd').first('d').index.values

    def roll(self, dataset: pd.DataFrame, roll_dates: pd.Series, match_end: bool = True):
        """
        
        :param dataset: (pd.DataFrame)
        :param roll_dates: (pd.Series)
        :param match_end: (bool)
        :return: (pd.Series) (pd.DataFrame)
        """
        gaps = dataset['PX_LAST']*0

        iloc=list(dataset.index)

        iloc = [iloc.index(i)-1 for i in roll_dates]

        gaps.loc[roll_dates] = dataset['PX_OPEN'].loc[roll_dates] - dataset['PX_LAST'].iloc[iloc].values

        gaps = gaps.cumsum().dropna()

        if match_end:
            gaps -= gaps.iloc[-1]

        return gaps, self.generate_diagnostic_frame(dataset, iloc, roll_dates)

    def non_negativeize(self, rolled_series: pd.Series, raw_series: pd.Series) -> pd.Series:
        """
        
        :param rolled_series: (pd.Series)
        :param raw_series: (pd.Series)
        :return: (pd.Series)        
        """
        new_prices_series = rolled_series.copy()
        new_prices_series = rolled_series.diff() / raw_series.shift(1)
        new_prices_series = (1+new_prices_series).cumprod()
        
        return new_prices_series