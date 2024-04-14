"""
Tests Spread Modeling filter functionality.
"""
import os
import unittest

import pandas as pd

from arbitragelab.ml_approach.filters import ThresholdFilter, CorrelationFilter, VolatilityFilter


class TestFilters(unittest.TestCase):
    """
    Tests Filtering Classes.
    """

    def setUp(self):
        """
        Loads futures price data.
        """

        project_path = os.path.dirname(__file__)

        # Load the needed contracts and calculate spread.
        cl_df = pd.read_csv(project_path + '/test_data/cl.csv',
                            parse_dates=True, index_col="Dates")['PX_LAST']
        rb_df = pd.read_csv(project_path + '/test_data/rb.csv',
                            parse_dates=True, index_col="Dates")['PX_LAST']
        df_spread = cl_df - rb_df

        # Concatenate everything for use cases that need all the data at the same time.
        working_df = pd.concat([cl_df, rb_df, df_spread], axis=1)
        working_df.columns = ["wti", "gasoline", "spread"]
        working_df.dropna(inplace=True)
        self.working_df = working_df

        # Calculate spread returns and std dev.
        spread_series = working_df['spread']
        self.spread_diff_series = spread_series.diff()
        self.spread_diff_std = self.spread_diff_series.std()

    def test_threshold_filter(self):
        """
        Tests the Threshold filter
        """

        # Initialize ThresholdFilter with 2 std dev band for buying and selling triggers.
        thres_filter = ThresholdFilter(buy_threshold=-self.spread_diff_std*2,
                                       sell_threshold=self.spread_diff_std*2)

        std_events = thres_filter.fit_transform(self.spread_diff_series)

        # Check that the correct amount of triggers have been set.
        self.assertEqual(std_events['side'].value_counts().values.tolist(),
                         [3817, 76, 58])

        thres_filter.plot()

    def test_correlation_filter(self):
        """
        Tests the Correlation filter.
        """

        # Initialize CorrelationFilter with +-0.05 correlation change to trigger buy/sell.
        corr_filter = CorrelationFilter(buy_threshold=0.05, sell_threshold=-0.05,
                                        lookback=30)
        corr_filter.fit(self.working_df[['wti', 'gasoline']])
        corr_events = corr_filter.transform(self.working_df[['wti', 'gasoline']])

        # Check that the correct amount of triggers have been set.
        self.assertEqual(corr_events['side'].value_counts().values.tolist(),
                         [3693, 130, 128])

        corr_filter.plot()

    def test_volatility_filter(self):
        """
        Tests the Volatility filter.
        """

        vol_filter = VolatilityFilter(lookback=80)

        vol_events = vol_filter.fit_transform(self.spread_diff_series)

        self.assertEqual(vol_events['regime'].value_counts().values.tolist(),
                         [1846, 1840, 80, 79, 13, 13])

        vol_filter.plot()
