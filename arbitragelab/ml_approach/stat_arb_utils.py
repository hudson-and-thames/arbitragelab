# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module houses utility functions used by the PairsSelector.
"""

import sys
import numpy as np
import pandas as pd
from arbitragelab.cointegration_approach import EngleGrangerPortfolio, get_half_life_of_mean_reversion
from arbitragelab.hedge_ratios import get_tls_hedge_ratio, get_ols_hedge_ratio


def _print_progress(iteration, max_iterations, prefix='', suffix='', decimals=1, bar_length=50):
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
    # Calculate the percent completed.
    percents = str_format.format(100 * (iteration / float(max_iterations)))
    # Calculate the length of bar.
    filled_length = int(round(bar_length * iteration / float(max_iterations)))
    # Fill the bar.
    block = '█' * filled_length + '-' * (bar_length - filled_length)
    # Print new line.
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, block, percents, '%', suffix)),

    if iteration == max_iterations:
        sys.stdout.write('\n')
    sys.stdout.flush()


def _outer_ou_loop(spreads_df: pd.DataFrame, test_period: str,
                   cross_overs_per_delta: int, molecule: list) -> pd.DataFrame:
    # pylint: disable=too-many-locals
    """
    This function gets mean reversion calculations (half-life and number of
    mean cross overs) for each pair in the molecule. Uses the linear regression
    method to get the half-life, which is much lighter computationally wise
    compared to the version using the OrnsteinUhlenbeck class.

    Note that when mean reversion is expected, lambda / StdErr has a negative value.
    This result implies that the expected duration of mean reversion lambda is
    inversely proportional to the absolute value of lambda.

    :param spreads_df: (pd.DataFrame) Spreads Universe.
    :param test_period: (str) Time delta format, to be used as the time
        period where the mean crossovers will be calculated.
    :param cross_overs_per_delta: (int) Crossovers per time delta selected.
    :param molecule: (list) Indices of pairs.
    :return: (pd.DataFrame) Mean Reversion statistics.
    """

    ou_results = []

    for iteration, pair in enumerate(molecule):
        spread = spreads_df.loc[:, str(pair)]

        # Split the spread in two periods. The training data is used to
        # extract the long term mean of the spread. Then the mean is used
        # to find the the number of crossovers in the test period.
        test_df = spread.last(test_period)
        train_df = spread.iloc[: -len(test_df)]

        long_term_mean = np.mean(train_df)

        centered_series = test_df - long_term_mean

        # Set the spread to a mean of zero and classifies each value
        # based on their sign.
        cross_over_indices = np.where(np.diff(np.sign(centered_series)))[0]
        cross_overs_dates = spreads_df.index[cross_over_indices]

        # Resample the mean crossovers series to yearly index and count
        # each occurence in each year.
        cross_overs_counts = cross_overs_dates.to_frame().resample('Y').count()
        cross_overs_counts.columns = ['counts']

        # Check that the number of crossovers are in accordance with the given selection
        # criteria.
        if cross_overs_per_delta is not None:
            cross_overs = len(cross_overs_counts[cross_overs_counts['counts'] >= cross_overs_per_delta]) > 0
        else:
            cross_overs = True

        # Append half-life and number of cross overs.
        half_life = get_half_life_of_mean_reversion(data=spread)
        ou_results.append([half_life, cross_overs])

        _print_progress(iteration + 1, len(molecule), prefix='Outer OU Loop Progress:',
                        suffix='Complete')

    return pd.DataFrame(ou_results, index=molecule, columns=['hl', 'crossovers'])


def _linear_f(beta: np.array, x_variable: np.array) -> np.array:
    """
    This is the helper linear model that is going to be used in the Orthogonal Regression.

    :param beta: (np.array) Model beta coefficient.
    :param x_variable: (np.array) Model X vector.
    :return: (np.array) Vector result of equation calculation.
    """

    return beta[0] * x_variable + beta[1]


def _outer_cointegration_loop(prices_df: pd.DataFrame, molecule: list, hedge_ratio_calculation: str) -> pd.DataFrame:
    """
    This function calculates the Engle-Granger test for each pair in the molecule.

    :param prices_df: (pd.DataFrame) Price Universe.
    :param molecule: (list) Indices of pairs.
    :param hedge_ratio_calculation: (str) Defines how hedge ratio is calculated. Can be either 'OLS,
                                        'TLS' (Total Least Squares) or 'min_half_life'.
    :return: (pd.DataFrame) Cointegration statistics.
    """

    cointegration_results = []

    for iteration, pair in enumerate(molecule):
        eg_port = EngleGrangerPortfolio()
        if hedge_ratio_calculation == 'OLS':
            fit, _, _, residuals = get_ols_hedge_ratio(price_data=prices_df.loc[:, [pair[0], pair[1]]],
                                                       dependent_variable=pair[0])
            hedge_ratio = fit.coef_
        if hedge_ratio_calculation == 'TLS':
            fit, _, _, residuals = get_tls_hedge_ratio(price_data=prices_df.loc[:, [pair[0], pair[1]]],
                                                       dependent_variable=pair[0])
            hedge_ratio = fit.beta[0]

        constant = residuals.mean()
        eg_port._perform_eg_test(residuals)
        statistic_value = eg_port.adf_statistics.loc['statistic_value'].iloc[0]
        p_value_99 = eg_port.adf_statistics.loc['99%'].iloc[0]
        p_value_95 = eg_port.adf_statistics.loc['95%'].iloc[0]
        p_value_90 = eg_port.adf_statistics.loc['90%'].iloc[0]

        cointegration_results.append(
            [statistic_value, p_value_99, p_value_95, p_value_90, hedge_ratio,
             constant])
        _print_progress(iteration + 1, len(molecule), prefix='Outer Cointegration Loop Progress:',
                        suffix='Complete')

    return pd.DataFrame(cointegration_results,
                        index=molecule,
                        columns=['coint_t', 'p_value_99%', 'p_value_95%', 'p_value_90%', 'hedge_ratio', 'constant'])
