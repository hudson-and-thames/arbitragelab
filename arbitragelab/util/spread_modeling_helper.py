# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the Spread Modeling Helper class.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from arbitragelab.ml_approach.feature_expander import FeatureExpander
from arbitragelab.ml_approach.filters import ThresholdFilter, CorrelationFilter, VolatilityFilter

# pylint: disable=R0902

class SpreadModelingHelper:
    """
    Dunis Approach.
    """

    def __init__(self, sprd: pd.Series, feat_expansion: bool = True, unique_sampling: bool = True):
        """
        The dataset returned in this module will consists of two major periods
        Insample Period - that will be broken down in two; the training period
        and the testing period. Out of Sample Period - the final stretch embragoed
        by a year from the insample period.

        :param sprd: (pd.Series)
        :param feat_expansion: (bool)
        :param unique_sampling: (bool)
        :return: (pd.DataFrame)
        """

        # Change spread to returns.
        sprd_rets = sprd.diff()

        # Generate Dataset, first columns is the target, the others are just lags of different amounts.
        dataset_df = pd.concat([sprd_rets, sprd_rets.shift(1), sprd_rets.shift(2),
                                sprd_rets.shift(3), sprd_rets.shift(4),
                                sprd_rets.shift(5)], axis=1).dropna()

        # Check if the dataset should be augmented with higher order terms or not.
        if feat_expansion:
            feat_expdr = FeatureExpander(methods=["product"], n_orders=2)
            feat_expdr.fit(dataset_df.loc[:, 1:].values)
            expanded_input_data = feat_expdr.transform()

            input_data = pd.DataFrame(data=expanded_input_data.values, index=dataset_df.index)
        else:
            input_data = dataset_df.loc[:, 1:]

        target_data = dataset_df.iloc[:, 0]

        insample_input, insample_target = input_data['2006':'2016'], target_data['2006':'2016']

        input_train, input_test, target_train, target_test = train_test_split(
            insample_input, insample_target, test_size=0.3, shuffle=False)

        input_scaler = MinMaxScaler().fit(input_train)

        target_scaler = MinMaxScaler().fit(target_train.values.reshape(-1, 1))

        input_oos, target_oos = input_data['2017':], target_data['2017':]

        if unique_sampling:
            # What is '.iloc[::N, N]' ?
            # - In most machine learning algorithm one of the assumptions is that the input data must be
            #     i.i.d 'independent and identically distributed'. When there is significant use of lagged
            #     variables, there will a lot of overlap thus breaking the independence assumption. To take
            #     care of this we sample the dataset every N rows.
            input_train = input_train.iloc[::6, :]
            target_train = target_train.iloc[::6]

        # input_train, input_test, input_oos, target_train, target_test, target_oos, ret_scaler
        self.input_train = input_scaler.transform(input_train)
        self.input_test = input_scaler.transform(input_test)
        self.input_oos = input_scaler.transform(input_oos)
        self.target_train = self._onedim_scaler_wrapper(target_scaler, target_train)
        self.target_test = self._onedim_scaler_wrapper(target_scaler, target_test)
        self.target_oos = self._onedim_scaler_wrapper(target_scaler, target_oos)
        self.ret_scaler = target_scaler

        # Initialization of variables needed post-model fitting.
        self.train_set = None
        self.train_pred = None
        self.test_set = None
        self.test_pred = None
        self.oos_set = None
        self.oos_pred = None

    @staticmethod
    def _onedim_scaler_wrapper(scaler_obj, original_data):
        """
        Converts data object to a proper scaled object.
        """

        reshaped_data = original_data.values.reshape(-1, 1)
        scaled_data = scaler_obj.transform(reshaped_data).reshape(-1)

        return pd.Series(scaled_data, index=original_data.index)

    @staticmethod
    def _wrap_threshold_filter(ytrue, ypred, std_dev_sample):
        """
        Wraps Threshold filter class.
        """

        thresh_filter = ThresholdFilter(-std_dev_sample, std_dev_sample)
        std_events = thresh_filter.fit_transform(ypred)

        std_events.columns = ["rets", "side"]
        std_events['rets'] = ytrue

        return std_events

    @staticmethod
    def _wrap_correlation_filter(ytrue, working_df):
        """
        Wraps Correlation filter class.
        """

        corr_filter = CorrelationFilter(buy_threshold=0.05,
                                        sell_threshold=-0.05,
                                        lookback=30)
        corr_filter.fit(working_df[['wti', 'gasoline']])

        corr_events = corr_filter.transform(ytrue.to_frame())
        corr_events.columns = ["rets", "side"]
        corr_events['rets'] = ytrue

        return corr_events

    @staticmethod
    def _wrap_vol_filter(ytrue, ypred, base_events):
        """
        Wraps Volatility filter class.
        """

        vol_filter = VolatilityFilter(lookback=80)
        vol_y = np.cumprod(1 + ypred) - 1
        vol_filter.fit(vol_y.astype('double'))

        vol_events = vol_filter.transform()
        vol_events.columns = ["rets", "regime", "leverage_multiplier"]
        vol_events['rets'] = ytrue * vol_events['leverage_multiplier']
        vol_events['side'] = base_events['side']

        return vol_events

    def get_filtering_results(self, ytrain, ypred, std_dev_sample, working_df, plot=True):
        """
        Executes and if needed plots the results from all the filter combinations
        included in the library.
        """

        unfiltered_events = self._wrap_threshold_filter(ytrain, ypred,
                                                        std_dev_sample.std()*0.01)
        unfiltered_events.name = "unfiltered"
        unfiltered_events.dropna(inplace=True)

        std_events = self._wrap_threshold_filter(ytrain, ypred,
                                                 std_dev_sample.std()*2)
        std_events.name = "std_threshold"
        std_events.dropna(inplace=True)

        corr_events = self._wrap_correlation_filter(ytrain, working_df)
        corr_events.name = "corr_filter"
        corr_events.dropna(inplace=True)

        std_vol_events = self._wrap_vol_filter(ytrain, ypred, std_events)
        std_vol_events.name = "std_vol_filter"
        std_vol_events.dropna(inplace=True)

        corr_vol_events = self._wrap_vol_filter(ytrain, ypred, corr_events)
        corr_vol_events.name = "corr_vol_filter"
        corr_vol_events.dropna(inplace=True)

        if plot:
            self.plot_trades(unfiltered_events, "Unfiltered")
            plt.show()
            self.plot_trades(std_events, "Threshold Filter")
            plt.show()
            self.plot_trades(corr_events, "Correlation Filter")
            plt.show()
            self.plot_trades(std_vol_events, "Threshold + Volatility Leverage Filter")
            plt.show()
            self.plot_trades(corr_vol_events, "Correlation + Volatility Leverage Filter")
            plt.show()

        return unfiltered_events, std_events, corr_events, std_vol_events, corr_vol_events

    def get_metrics(self, working_df):
        """
        Main handler for gathering all the financial metrics relating
        to each dataset slice.
        """

        train_filter_results = self.get_filtering_results(self.train_set, self.train_pred, self.train_pred, working_df, False)
        test_filter_results = self.get_filtering_results(self.test_set, self.test_pred, self.train_pred, working_df, False)
        oos_filter_results = self.get_filtering_results(self.oos_set, self.oos_pred, self.test_pred, working_df, False)

        train_filter_metrics = self.convert_filter_events_to_metrics(train_filter_results)
        train_filter_metrics['Set'] = 'train'

        test_filter_metrics = self.convert_filter_events_to_metrics(test_filter_results)
        test_filter_metrics['Set'] = 'test'

        oos_filter_metrics = self.convert_filter_events_to_metrics(oos_filter_results)
        oos_filter_metrics['Set'] = 'oos'

        return pd.concat([train_filter_metrics, test_filter_metrics, oos_filter_metrics])

    @staticmethod
    def convert_filter_events_to_metrics(filter_events):
        """
        Convert a set of returns to financial metrics.
        """

        metrics = []

        for fresult in filter_events:
            rets = fresult[fresult['side'] != 0]['rets']
            possible_days = len(pd.date_range(rets.index[0], rets.index[-1]))

            # THIS ANNUAL RETURN CALCULATION NEEDS TO BE REVISED.
            annual_ret = rets.add(1).prod() ** (252 / possible_days) - 1
            std_dev = fresult[fresult['side'] != 0]['rets'].std()
            metrics.append([fresult.name, annual_ret, std_dev])

        return pd.DataFrame(metrics, columns=['Filtering Method', 'Annual Returns', 'Std Dev'])

    def plot_model_results(self, model):
        """
        Inverts the scaling of the dataset and plots the regression results.
        """

        predicted_train_y = model.predict(self.input_train)
        predicted_test_y = model.predict(self.input_test)
        predicted_oos_y = model.predict(self.input_oos)

        y_train_inverted = pd.Series(self.ret_scaler.inverse_transform(
            self.target_train.values.reshape(-1, 1)).reshape(-1), index=self.target_train.index)
        y_test_inverted = pd.Series(self.ret_scaler.inverse_transform(
            self.target_test.values.reshape(-1, 1)).reshape(-1), index=self.target_test.index)
        y_oos_inverted = pd.Series(self.ret_scaler.inverse_transform(
            self.target_oos.values.reshape(-1, 1)).reshape(-1), index=self.target_oos.index)

        predicted_train_inverted = pd.Series(self.ret_scaler.inverse_transform(
            predicted_train_y.reshape(-1, 1)).reshape(-1), index=self.target_train.index)  # .reshape(-1)
        predicted_test_inverted = pd.Series(self.ret_scaler.inverse_transform(
            predicted_test_y.reshape(-1, 1)).reshape(-1), index=self.target_test.index)  # .reshape(-1)
        predicted_oos_inverted = pd.Series(self.ret_scaler.inverse_transform(
            predicted_oos_y.reshape(-1, 1)).reshape(-1), index=self.target_oos.index)  # .reshape(-1)

        self.plot_regression_results(y_train_inverted,
                                     predicted_train_inverted,
                                     "Predicting Training Set")
        plt.show()
        self.plot_regression_results(y_test_inverted,
                                     predicted_test_inverted,
                                     "Predicting Test Set")
        plt.show()
        self.plot_regression_results(y_oos_inverted,
                                     predicted_oos_inverted,
                                     "Predicting Out of Sample Set")

        self.train_set = y_train_inverted
        self.train_pred = predicted_train_inverted
        self.test_set = y_test_inverted
        self.test_pred = predicted_test_inverted
        self.oos_set = y_oos_inverted
        self.oos_pred = predicted_oos_inverted

        return y_train_inverted, predicted_train_inverted, y_test_inverted, predicted_test_inverted, y_oos_inverted, predicted_oos_inverted

    @staticmethod
    def plot_trades(events, title):
        """
        Plots long/short/all trades given a set of labeled returns.
        """

        long_trades = events[(events['side'] == 1)].iloc[:, 0]
        short_trades = -(events[(events['side'] == -1)].iloc[:, 0])
        all_trades = pd.concat([long_trades, short_trades]).sort_index()

        plt.plot(np.cumprod(1 + long_trades.values) - 1)

        plt.plot(np.cumprod(1 + short_trades.values) - 1)

        plt.plot(np.cumprod(1 + all_trades.values) - 1)

        plt.legend(['long trades', 'short trades', 'all trades'])

        plt.title(title)

    @staticmethod
    def plot_regression_results(y_true, y_pred, title):
        """
        Plots Regression results for visual comparison.
        """

        print(r2_score(y_true, y_pred))

        plt.figure(figsize=(15, 10))

        plt.plot(y_pred.values)
        plt.plot(y_true.values, alpha=0.5)

        plt.legend(["Predicted Y", "True Y"])
        plt.title(title)
