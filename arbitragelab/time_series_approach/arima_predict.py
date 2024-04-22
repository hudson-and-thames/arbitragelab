"""
The module implements the ARIMA forecast of any time series using the Auto-ARIMA approach.
"""
# pylint: disable=consider-using-f-string, broad-exception-raised

import warnings
import sys
from contextlib import nullcontext

import pandas as pd

from pmdarima.arima import auto_arima, ADFTest


def get_trend_order(y_train: pd.Series, max_order: int = 10) -> int:
    """
    Get trend order for a time series using 95% ADF test.

    :param y_train: (pd.Series) Series to test.
    :param max_order: (int) Highest order value that is being tested.
    :return: (int) Trend order, 0 means that `y_train` is already stationary.
    """

    # Setting the ADF class and the initial parameters
    adf_test = ADFTest(alpha=0.05)
    diff_order = 0
    order = 0
    stationarity_flag = False

    # Iterating through possible trend orders
    while stationarity_flag is False:
        test_series = y_train.copy()
        for _ in range(order):
            test_series = test_series.diff().dropna()

        # Testing if the trend order fits
        if bool(adf_test.should_diff(test_series)[1]) is False:
            diff_order = order
            stationarity_flag = True

        # Avoiding infinite loop
        if order >= max_order:
            stationarity_flag = True

        order += 1

    return diff_order


class AutoARIMAForecast:
    """
    Auto ARIMA forecast generator function.
    """

    def __init__(
        self, start_p: int = 0, start_q: int = 0, max_p: int = 5, max_q: int = 5
    ):
        """
        Init AutoARIMA (p, i, q)  prediction class.

        :param start_p: (int) Starting value of p (number of time lags) to search in auto ARIMA procedure.
        :param start_q: (int) Starting value of q (moving average window) to search in auto ARIMA procedure.
        :param max_p: (int) Maximum possible value of p.
        :param max_q: (int) Maximum possible value of q.
        """

        self.start_p = start_p
        self.start_q = start_q
        self.max_p = max_p
        self.max_q = max_q

        self.arima_model = None
        self.y_train = None

    def get_best_arima_model(
        self, y_train: pd.Series, verbose: bool = False, silence_warnings: bool = True
    ):
        """
        Using the AIC approach from pmdarima library, choose the best fit ARIMA(d, p, q) parameters.

        :param y_train: (pd.Series) Training series.
        :param verbose: (bool) Flag to print model fit logs.
        :param silence_warnings: (bool) Flag to silence warnings from the Auto ARIMA model - convergence warnings etc.
        """

        # Setting parameters
        trend_order = get_trend_order(y_train)
        self.y_train = y_train.copy()

        if silence_warnings:
            context = warnings.catch_warnings()
        else:
            context = nullcontext()

        with context:  # Silencing Warnings
            if silence_warnings:
                warnings.filterwarnings("ignore")

            # Fitting the ARIMA model without warnings
            self.arima_model = auto_arima(
                y=y_train,
                d=trend_order,
                start_p=self.start_p,
                start_q=self.start_q,
                max_p=self.max_p,
                max_q=self.max_q,
                max_order=self.max_q + self.max_p + trend_order,
                trace=verbose,
            )

    @staticmethod
    def _print_progress(
        iteration, max_iterations, prefix="", suffix="", decimals=1, bar_length=50
    ):
        # pylint: disable=expression-not-assigned
        """
        Calls in a loop to create a terminal progress bar.
        https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

        :param iteration: (int) Current iteration.
        :param max_iterations: (int) Maximum number of iterations.
        :param prefix: (str) Prefix string.
        :param suffix: (str) Suffix string.
        :param decimals: (int) Positive number of decimals in percent completed.
        :param bar_length: (int) Character length of the bar.
        """

        str_format = "{0:." + str(decimals) + "f}"
        # Calculate the percent completed
        percents = str_format.format(100 * (iteration / float(max_iterations)))
        # Calculate the length of bar
        filled_length = int(round(bar_length * iteration / float(max_iterations)))

        # Fill the bar
        block = "█" * filled_length + "-" * (bar_length - filled_length)
        # Print new line
        sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, block, percents, "%", suffix)),

        if iteration == max_iterations:
            sys.stdout.write("\n")
        sys.stdout.flush()

    # pylint: disable=invalid-name
    def predict(
        self,
        y: pd.Series,
        retrain_freq: int = 1,
        train_window: int = None,
        silence_warnings: bool = True,
    ) -> pd.Series:
        """
        Predict out-of-sample series using already fit ARIMA model. The algorithm retrains the model with `retrain_freq`
        either by appending new observations to train data (`train_window` = None) or by using the latest `train_window`
        observations + latest out-of-sample observations `y`.

        :param y: (pd.Series) Out-of-sample series (used to generate rolling forecast).
        :param retrain_freq: (int) Model retraining frequency. Model is fit on every `train_freq` step.
        :param train_window: (int) Number of data points from train dataset used in model retrain. If None, use all
            train set.
        :param silence_warnings: (bool) Flag to silence warnings from the Auto ARIMA model - convergence warnings etc.
        :return: (pd.Series) Series of forecasted values.
        """

        # Setting input parameters
        prediction = pd.Series(index=y.index, dtype=float)
        retrain_idx = 0

        if silence_warnings:
            context = warnings.catch_warnings()
        else:
            context = nullcontext()

        with context:  # Silencing Warnings
            if silence_warnings:
                warnings.filterwarnings("ignore")

            # Iterating through observations
            for i in range(1, len(y)):
                if retrain_idx >= retrain_freq:  # Retraining model
                    retrain_idx = 0

                    if (
                        train_window is None
                    ):  # If no training window, fit to all previous observations
                        # i-1 to avoid look-ahead bias.
                        out_of_sample_y_train = pd.concat(
                            [self.y_train, y.iloc[: i - 1]]
                        )
                        prediction.loc[y.index[i]] = self.arima_model.fit_predict(
                            out_of_sample_y_train, n_periods=1
                        ).values[0]

                    else:
                        out_of_sample_y_train = pd.concat(
                            [self.y_train.iloc[-1 * train_window :], y.iloc[: i - 1]]
                        )
                        prediction.loc[y.index[i]] = self.arima_model.fit_predict(
                            out_of_sample_y_train,
                            n_periods=1,
                        ).values[0]

                else:  # Using trained model
                    prediction.loc[y.index[i]] = self.arima_model.predict(
                        n_periods=1
                    ).values[0]

                retrain_idx += 1

                # Print progress to inform user
                self._print_progress(
                    i + 1, y.shape[0], prefix="Progress:", suffix="Complete"
                )

        return prediction
