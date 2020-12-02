# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This module implements the quantile time series approach described in
Sarmento and Nuno Horta in `"A Machine Learning based Pairs Trading Investment Strategy." <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class QuantileTimeSeriesTradingStrategy:
    """
    The strategy which implements a quantile-based time series approach in mean-reversion trading. First, we define
    top quantile of positive spread (`y`) differences and bottom quantile of negative spread differences.
    Secondly, we use time series spread prediction `y_hat` (it can be user-specified prediction, ARIMA, ANN, RNN, etc.)
    We enter a position if y_hat - y <= bottom quantile or y_hat - y >= top quantile.
    """

    def __init__(self, long_quantile: float = 0.9, short_quantile: float = 0.1):
        """
        Class constructor.

        :param long_quantile: (float) Positive spread differences quantile used as long entry threshold.
        :param short_quantile: (float) Negative spread differences quantile used as short entry threshold.
        """

        self.long_quantile = long_quantile
        self.short_quantile = short_quantile
        self.long_diff_threshold = None
        self.short_diff_threshold = None

        self.positive_differences = None
        self.negative_differences = None

        self.positions = []  # Positions (-1, 0, 1) logs

    def fit_thresholds(self, spread_series: pd.Series):
        """
        Define quantile-based long/short difference thresholds from spread series.

        :param spread_series: (pd.Series) Spread series used to fit thresholds.
        """

        differences = spread_series.diff()
        self.positive_differences = differences[differences > 0]
        self.negative_differences = differences[differences < 0]

        self.long_diff_threshold = self.positive_differences.quantile(self.long_quantile)
        self.short_diff_threshold = self.negative_differences.quantile(self.short_quantile)

    def plot_thresholds(self):
        """
        Plot KDE-plots of positive and negative differences vs long/short thresholds.

        :return: (plt.axes) The KDE plot.
        """

        _, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

        # Positive differences plot
        sns.kdeplot(self.positive_differences, shade=True, color="green", ax=axes[0],
                    label='threshold: {}'.format(self.long_diff_threshold.round(4)))
        axes[0].axvline(self.long_diff_threshold, linestyle='--', color='black')
        axes[0].set_title('Positive differences KDE')

        # Negative differences plot
        sns.kdeplot(self.negative_differences, shade=True, color="red", ax=axes[1],
                    label='threshold: {}'.format(self.short_diff_threshold.round(4)))
        axes[1].axvline(self.short_diff_threshold, linestyle='--', color='black')
        axes[1].set_title('Negative differences KDE')

        return axes

    def get_allocation(self, predicted_difference: float, exit_threshold: float = 0) -> int:
        """
        Get target allocation (-1, 0, 1) based on current spread value, predicted value, and exit threshold. -1/1 means
        either to open a new short/long position or stay in a long/short trade (if the position has been already opened).
        0 means exit the position.

        :param predicted_difference: (float) Spread predicted value - current spread value
        :param exit_threshold: (float) Difference between predicted and current value threshold to close the trade.
        :return: (int) Trade signal: -1 (short), 0 (exit current position/stay in cash), 1(long).
        """

        # New position entry
        if predicted_difference >= self.long_diff_threshold:
            return_flag = 1
        elif predicted_difference <= self.short_diff_threshold:
            return_flag = -1
        elif len(self.positions) > 0 and self.positions[-1] == 1 and predicted_difference > exit_threshold:
            return_flag = 1
        elif len(self.positions) > 0 and self.positions[-1] == -1 and predicted_difference <= exit_threshold:
            return_flag = -1
        else:
            return_flag = 0

        self.positions.append(return_flag)

        return return_flag
