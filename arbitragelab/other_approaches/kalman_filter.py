# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
This module implements Kalman filter logic in statistical arbitrage trading. Kalman trading
technique is applied to dynamically estimate a hedge ratio between two assets. The output of
the Kalman filter can be used either to build a mean-reverting portfolio and use Bollinger-bands
strategy on top of that, or use Kalman filter residuals and standard deviation to form trading
signals. This implementation is based on the one from the book by E.P Chan:
`"Algorithmic Trading: Winning Strategies and Their Rationale"
<https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146>`_,

Additional information can be found in the following sources:
`Quantitative Research and Trading <http://jonathankinlay.com/2018/10/etf-pairs-trading-kalman-filter/>`__.
`Quantopian <https://www.quantopian.com/posts/ernie-chans-ewa-slash-ewc-pair-trade-with-kalman-filter>`__.
`QuantStart <https://www.quantstart.com/articles/kalman-filter-based-pairs-trading-strategy-in-qstrader/>`__.
"""

import numpy as np
import pandas as pd


# pylint: disable=invalid-name
from arbitragelab.util import devadarsh


class KalmanFilterStrategy:
    """
    KalmanFilterStrategy implements a dynamic hedge ratio estimation between two assets using
    the Kalman filter. Kalman Filter is a state space model that assumes the system state evolves
    following some hidden and unobservable pattern. The goal of the state space model is to infer
    information about the states, given the observations, as new information arrives. The strategy
    has two important values to fit: observation covariance and transition covariance.

    There are two ways to fit them: using cross-validation technique or by applying
    Autocovariance Least Squares (ALS) algorithm. Kalman filter approach generalizes
    a rolling linear regression estimate.

    This class implements the Kalman Filter from the book by E.P Chan:
    `"Algorithmic Trading: Winning Strategies and Their Rationale"
    <https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146>`_,
    """

    def __init__(self, observation_covariance: float = 0.001, transition_covariance: float = 1e-4):
        """
        Init Kalman Filter strategy.

        Kalman filter has two important parameters which need to be set in advance or optimized: observation covariance
        and transition covariance.

        :param observation_covariance: (float) Observation covariance value.
        :param transition_covariance: (float) Transition covariance value.
        """

        self.observation_covariance = observation_covariance
        self.transition_covariance = transition_covariance * np.eye(2)

        self.hedge_ratios = []  # Hedge ratio is a slope of linear regression (beta(1))
        self.intercepts = []  # Kalman filter also provides the estimate of an intercept (beta(2))
        self.spread_series = []  # Series of forecast errors (e)
        self.spread_std_series = []  # Series of standard deviations of forecast errors (sqrt(Q))

        self.means_trace = []  # Series of betas - states
        self.covs_trace = []  # Series of P - state covariances

        # Helper variables from Kalman filter
        self.R = None  # Prediction variance-covariance
        self.beta = np.array([0, 0])  # Starting state prediction

        devadarsh.track('KalmanFilterStrategy')

    def update(self, x: float, y: float):
        """
        Update the hedge ratio based on the recent observation of two assets.

        By default, y is the observed variable and x is the hidden one. That is the hedge ratio for y is 1 and the
        hedge ratio for x is estimated by the Kalman filter.

        Mean-reverting portfolio series is formed by:

        y - self.hedge_ratios * x

        One can get spread series from self.spread_series and self.spread_std_series to trade the Bollinger Bands
        strategy.

        :param x: (float) X variable value (hidden).
        :param y: (float) Y variable value.
        """

        # Observation matrix F is 2-dimensional, containing x price and 1 (intercept)
        observation_matrix = np.array([[x, 1]])  # Matrix F

        # Setting variance-covariance prediction
        if self.R is not None:
            P = self.covs_trace[-1]
            self.R = P + self.transition_covariance
        else:
            self.R = np.zeros((2, 2))

        # Adding last beta to the list of betas
        self.means_trace.append(self.beta)

        # Updating hedge ratios and intercepts
        self.hedge_ratios.append(self.means_trace[-1][0])
        self.intercepts.append(self.means_trace[-1][1])

        # Helping calculation of x * R * x'
        xrx = observation_matrix.dot(self.R).dot(observation_matrix.transpose())[0][0]

        # Forecast errors and their standard deviation
        spread = y - observation_matrix.dot(self.means_trace[-1])[0]  # e
        spread_std = np.sqrt(xrx + self.observation_covariance)  # sqrt(Q)

        self.spread_series.append(spread)
        self.spread_std_series.append(spread_std)

        # Kalman gain calculation
        K = self.R.dot(observation_matrix.transpose()) / (spread_std ** 2)

        # New state covariance
        P = self.R - K.dot(observation_matrix).dot(self.R)
        self.covs_trace.append(P)

        # New state
        self.beta = self.beta + K.dot(spread).transpose()[0]

    def trading_signals(self, entry_std_score: float = 3, exit_std_score: float = -3) -> pd.DataFrame:
        """
        Generate trading signals based on existing data.

        This method uses recorded forecast errors and standard deviations of
        forecast errors to generate trading signals, as described in the book by E.P Chan
        "Algorithmic Trading: Winning Strategies and Their Rationale".

        The logic is to have a long position open from
        e(t) < -entry_std_score * sqrt(Q(t)) till e(t) >= -exit_std_score * sqrt(Q(t))
        And a short position from
        e(t) > entry_std_score * sqrt(Q(t)) till e(t) <= exit_std_score * sqrt(Q(t))

        where e(t) is a forecast error at time t, and sqrt(Q(t)) is the
        standard deviation of standard errors at time t.

        :param entry_std_score: (float) Number of st.d. values to enter (long or short) the position.
        :param exit_std_score: (float) Number of st.d. values to exit (long or short) the position.
        :return: (pd.DataFrame) Series with forecast errors and target allocation on each observation.
        """

        # Internal lists to series
        forecast_err = pd.Series(self.spread_series, name="errors")
        std_err = pd.Series(self.spread_std_series, name='error_std')

        # Base DataFrame
        results_df = forecast_err.to_frame()

        # Entry and exit signals
        long_entry_index = forecast_err[forecast_err < -entry_std_score * std_err].index
        long_exit_index = forecast_err[forecast_err >= -exit_std_score * std_err].index

        short_entry_index = forecast_err[forecast_err > entry_std_score * std_err].index
        short_exit_index = forecast_err[forecast_err <= exit_std_score * std_err].index

        results_df['long_units'] = np.nan
        results_df['short_units'] = np.nan
        results_df.iloc[0, results_df.columns.get_loc('long_units')] = 0
        results_df.iloc[0, results_df.columns.get_loc('short_units')] = 0

        results_df.loc[long_entry_index, 'long_units'] = 1
        results_df.loc[long_exit_index, 'long_units'] = 0
        results_df.loc[short_entry_index, 'short_units'] = -1
        results_df.loc[short_exit_index, 'short_units'] = 0

        results_df.fillna(method='pad', inplace=True)
        results_df['target_quantity'] = results_df['long_units'] + results_df['short_units']

        return results_df[['errors', 'target_quantity']]
