# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the Spread Modeling Helper class.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from arbitragelab.ml_approach.feature_expander import FeatureExpander
from arbitragelab.ml_approach.filters import ThresholdFilter, CorrelationFilter, VolatilityFilter

# pylint: disable=R0914

class SpreadModelingHelper:
    """
    This class basically wraps most of the framework's most
    repeatable functionality into two basic steps;
    - init - where the datasets given the spread are generated.
    - plot_model_results - given a model plot strategy results and metrics.
    """

    def __init__(self, sprd: pd.Series, insample_date_range: tuple,
                 oosample_date_range: tuple, feat_expansion: bool = True,
                 unique_sampling: bool = True):
        """
        This method will break the spread in two major periods:

        Insample Period - that will be broken down in two; the training period
        and the testing period.
        Out of Sample Period - the final stretch embargoed by a year from the
        insample period.

        The dataset generated here according to the original paper, is based
        on lagged returns. Returns are assumed to be normalized thus no
        scaling was employed.

        :param sprd: (pd.Series) A series consisting of the spread.
        :param insample_date_range: (tuple) Date range to be use for insample dataset; (from, to).
        :param oosample_date_range: (tuple) Date range to be use for oosample dataset; (from, to).
        :param feat_expansion: (bool) If enabled will deploy the feature expansion
            procedure based on what is documented in (Dunis et al. 2006)
        :param unique_sampling: (bool) If enabled will remove all the rows
            at every N lag.
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

        # Get the target 'y' variable from the dataset variable.
        target_data = dataset_df.iloc[:, 0]

        # Prepare the date slice we are going to use for the insample dataset.
        insample_sliced = slice(insample_date_range[0], insample_date_range[1])

        # Cut and set the input and target data for insample.
        insample_input, insample_target = input_data[insample_sliced], target_data[insample_sliced]

        # Split insample dataset into training and test sets.
        input_train, input_test, target_train, target_test = train_test_split(
            insample_input, insample_target, test_size=0.3, shuffle=False)

        # Prepare the date slice we are going to use for the oosample dataset.
        oosample_sliced = slice(oosample_date_range[0], oosample_date_range[1])

        # Cut and set the input and target data for oosample.
        input_oos, target_oos = input_data[oosample_sliced], target_data[oosample_sliced]

        if unique_sampling:
            # What is '.iloc[::N, N]' ?
            # - In most machine learning algorithm one of the assumptions is that the input data must be
            #     i.i.d 'independent and identically distributed'. When there is significant use of lagged
            #     variables, there will a lot of overlap thus breaking the independence assumption. To take
            #     care of this we sample the dataset every N rows.
            input_train = input_train.iloc[::6, :]
            target_train = target_train.iloc[::6]

        self.input_train = input_train
        self.input_test = input_test
        self.input_oos = input_oos
        self.target_train = target_train
        self.target_test = target_test
        self.target_oos = target_oos

        # Initialization of variables needed post-model fitting.
        self.train_pred = None
        self.test_pred = None
        self.oos_pred = None

    @staticmethod
    def _wrap_threshold_filter(ytrue: pd.DataFrame, ypred: pd.DataFrame, std_dev_sample: float) -> pd.DataFrame:
        """
        Wraps Threshold filter class.

        :param ytrue: (pd.DataFrame) The ground truth data.
        :param ypred: (pd.DataFrame) The predicted data.
        :param std_dev_sample: (float) The range for the buy/sell threshold.
        :return: (pd.DataFrame)
        """

        thresh_filter = ThresholdFilter(-std_dev_sample, std_dev_sample)
        std_events = thresh_filter.fit_transform(ypred)

        std_events.columns = ["rets", "side"]
        std_events['rets'] = ytrue

        return std_events

    @staticmethod
    def _wrap_correlation_filter(ytrue: pd.DataFrame, working_df: pd.DataFrame) -> pd.DataFrame:
        """
        Wraps Correlation filter class.

        :param ytrue: (pd.DataFrame) The ground truth data.
        :param working_df: (pd.DataFrame) DataFrame with both legs of the spread.
        :return: (pd.DataFrame)
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
    def _wrap_vol_filter(ytrue: pd.DataFrame, ypred: pd.DataFrame, base_events: pd.DataFrame) -> pd.DataFrame:
        """
        Wraps Volatility filter class.

        :param ytrue: (pd.DataFrame) The ground truth data.
        :param ypred: (pd.DataFrame) The predicted data.
        :param base_events: (pd.DataFrame) A DataFrame full of events to be filtered.
        :return: (pd.DataFrame)
        """

        vol_filter = VolatilityFilter(lookback=80)
        vol_y = np.cumprod(1 + ypred) - 1
        vol_filter.fit(vol_y.astype('double'))

        vol_events = vol_filter.transform()
        vol_events.columns = ["rets", "regime", "leverage_multiplier"]
        vol_events['rets'] = ytrue * vol_events['leverage_multiplier']
        vol_events['side'] = base_events['side']

        return vol_events

    def get_filtering_results(self, ytrain: pd.DataFrame, ypred: pd.DataFrame,
                              std_dev_sample: float, working_df: pd.DataFrame,
                              plot: bool = True, figsize: tuple = (15, 10)):
        """
        Executes and if needed plots the results from all the filter combinations
        included in the library.

        :param ytrue: (pd.DataFrame) The ground truth data.
        :param ypred: (pd.DataFrame) The predicted data.
        :param std_dev_sample: (float) The range for the buy/sell threshold.
        :param working_df: (pd.DataFrame) DataFrame with both legs of the spread.
        :param plot: (bool) Plot the trades of each filter.
        :param figsize: (tuple) Figure size as a tuple for plotting.
        :return: (tuple) Filter events.
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
            self.plot_trades(unfiltered_events, "Unfiltered", figsize=figsize)
            plt.show()
            self.plot_trades(std_events, "Threshold Filter", figsize=figsize)
            plt.show()
            self.plot_trades(corr_events, "Correlation Filter", figsize=figsize)
            plt.show()
            self.plot_trades(std_vol_events, "Threshold + Volatility Leverage Filter", figsize=figsize)
            plt.show()
            self.plot_trades(corr_vol_events, "Correlation + Volatility Leverage Filter", figsize=figsize)
            plt.show()

        return unfiltered_events, std_events, corr_events, std_vol_events, corr_vol_events

    def get_metrics(self, working_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main handler for gathering all the financial metrics relating
        to each dataset slice.

        :param working_df: (pd.DataFrame) DataFrame with both legs of the spread.
        :return: (pd.DataFrame) Metrics for all sets.
        """

        train_filter_results = self.get_filtering_results(self.target_train, self.train_pred,
                                                          self.train_pred, working_df, False)
        test_filter_results = self.get_filtering_results(self.target_test, self.test_pred,
                                                         self.train_pred, working_df, False)
        oos_filter_results = self.get_filtering_results(self.target_oos, self.oos_pred,
                                                        self.test_pred, working_df, False)

        train_filter_metrics = self.convert_filter_events_to_metrics(train_filter_results)
        train_filter_metrics['Set'] = 'train'

        test_filter_metrics = self.convert_filter_events_to_metrics(test_filter_results)
        test_filter_metrics['Set'] = 'test'

        oos_filter_metrics = self.convert_filter_events_to_metrics(oos_filter_results)
        oos_filter_metrics['Set'] = 'oos'

        return pd.concat([train_filter_metrics, test_filter_metrics, oos_filter_metrics])

    @staticmethod
    def convert_filter_events_to_metrics(filter_events: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a set of returns to financial metrics. The metrics
        include annual returns, annual volatility, max drawdown and
        sharpe ratio.

        :param filter_events: (pd.DataFrame) Trade events tagged with a
            'side' column with 1/0/-1 referring to long, no trade, short.
        :return: (pd.DataFrame) Financial metrics for each set.
        """

        metrics = []

        for fresult in filter_events:
            # Get all events with side 1 or -1; which means that have a long
            # or short signal.
            processed_freturns = fresult[fresult['side'] != 0]

            # Select the short only events as a mask.
            short_side_mask = (processed_freturns['side'] == -1)

            # Switch the sign of the returns of the short side.
            processed_freturns[short_side_mask] = -processed_freturns[short_side_mask]

            # And store the processed returns into one final variable.
            rets = processed_freturns['rets']

            # Get count of all possible days from start to end of returns index.
            possible_days = len(pd.date_range(rets.index[0], rets.index[-1]))

            returns_sign = np.sign(rets.add(1).prod())

            # Get average annual returns, based on a 365/day year.
            annual_ret = (abs(rets.add(1).prod()) ** (356 / possible_days)) - 1
            # Round and get the return out of 100.
            annual_ret = np.round(annual_ret*100, 2)*returns_sign

            # Calculate the std dev of all trades
            std_dev = rets.std()
            # Calculate volatility and annualize it.
            annual_vol = std_dev ** (1.0 / 2)
            annual_vol = np.round(annual_vol*100, 2)

            # Calculate the sharpe ratio.
            sharpe_ratio = np.round(rets.mean() / std_dev, 2)

            # Calculate overrall max drawdown.
            cum_returns = (np.cumprod(1 + rets.values))*100
            max_return = np.fmax.accumulate(cum_returns, axis=0)
            max_drawdown = np.min((cum_returns - max_return) / max_return, axis=0)
            max_drawdown = np.round(max_drawdown*100, 2)

            metrics.append([fresult.name, annual_ret,
                            annual_vol, max_drawdown,
                            sharpe_ratio])

        return pd.DataFrame(metrics, columns=['Filtering Method', 'Annual Returns',
                                              'Annual Volatility', 'Max Drawdown',
                                              'Sharpe Ratio'])

    def plot_model_results(self, model, figsize=(15, 10)):
        """
        Plots the regression results for the training, test and oos sets.

        :param model: (Object) ML model that has the method 'predict' implemented.
        :param figsize: (tuple) Figure size for plot.
        :return: (tuple) Predicted target values for all sets.
        """

        predicted_train_y = pd.Series(model.predict(self.input_train))
        predicted_train_y.index = self.target_train.index

        predicted_test_y = pd.Series(model.predict(self.input_test))
        predicted_test_y.index = self.target_test.index

        predicted_oos_y = pd.Series(model.predict(self.input_oos))
        predicted_oos_y.index = self.target_oos.index

        self.plot_regression_results(self.target_train,
                                     predicted_train_y,
                                     "Predicting Training Set",
                                     figsize)
        plt.show()
        self.plot_regression_results(self.target_test,
                                     predicted_test_y,
                                     "Predicting Test Set",
                                     figsize)
        plt.show()
        self.plot_regression_results(self.target_oos,
                                     predicted_oos_y,
                                     "Predicting Out of Sample Set",
                                     figsize)

        self.train_pred = predicted_train_y
        self.test_pred = predicted_test_y
        self.oos_pred = predicted_oos_y

        return predicted_train_y, predicted_test_y, predicted_oos_y

    @staticmethod
    def plot_trades(events: pd.DataFrame, title: str, figsize: tuple = (15, 10)):
        """
        Plots long/short/all trades given a set of labeled returns.

        :param events: (pd.DataFrame) Trade DataFrame with returns and side as columns.
        :param title: (str) Title for plot.
        :param figsize: (tuple) Figure size for plot.
        """

        plt.figure(figsize=figsize)

        long_trades = events[(events['side'] == 1)].iloc[:, 0]
        short_trades = -(events[(events['side'] == -1)].iloc[:, 0])
        all_trades = pd.concat([long_trades, short_trades]).sort_index()

        plt.plot(np.cumprod(1 + long_trades.values) - 1)
        plt.plot(np.cumprod(1 + short_trades.values) - 1)
        plt.plot(np.cumprod(1 + all_trades.values) - 1)

        plt.legend(['long trades', 'short trades', 'all trades'])
        plt.title(title)

    @staticmethod
    def plot_regression_results(y_true: pd.Series, y_pred: pd.Series, title: str, figsize: tuple = (15, 10)):
        """
        Plots Regression results (predicted vs ground) for visual comparison.

        :param ytrue: (pd.Series) The ground truth data.
        :param ypred: (pd.Series) The predicted data.
        :param title: (str) Title for plot.
        :param figsize: (tuple) Figure size for plot.
        """

        print("This is r^2 score for " + str(title) + ": " + str(r2_score(y_true, y_pred)))

        plt.figure(figsize=figsize)

        plt.plot(y_pred)
        plt.plot(y_true, alpha=0.5)

        plt.legend(["Predicted Y", "True Y"])
        plt.title(title)
