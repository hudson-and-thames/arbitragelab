# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Back end module that handles maximum likelihood related copula calculations.

Functions include:

    - Finding (marginal) cumulative distribution function from data.
    - Finding empirical cumulative distribution function from data with linear interpolation.
    - Maximum likelihood estimation of theta_hat (empirical theta) from data.
    - Calculating the sum log-likelihood given a copula and data.
    - Calculating SIC (Schwarz information criterion).
    - Calculating AIC (Akaike information criterion).
    - Calculating HQIC (Hannan-Quinn information criterion).
    - Fitting Student-t Copula.
    - SCAD penalty functions.
    - Adjust weights for mixed copulas for normality.

For more information about the SCAD penalty functions on fitting mixed copulas, please refer to
`Cai, Z. and Wang, X., 2014. Selection of mixed copula model via penalized likelihood. Journal of the American
Statistical Association, 109(506), pp.788-801.
<https://www.tandfonline.com/doi/pdf/10.1080/01621459.2013.873366?casa_token=sey8HrojSgYAAAAA:TEMBX8wLYdGFGyM78UXSYm6hXl1Qp_K6wiLgRJf6kPcqW4dYT8z3oA3I_odrAL48DNr3OSoqkQsEmQ>`__
"""
# pylint: disable = invalid-name
from typing import Callable, Tuple
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.covariance import EmpiricalCovariance

import arbitragelab.copula_approach.copula_generate as cg


def find_marginal_cdf(x: np.array, empirical: bool = True, **kwargs) -> Callable[[float], float]:
    """
    Find the cumulative density function (CDF). i.e., P(X<=x).

    User can choose between an empirical CDF or a CDF selected by maximum likelihood.

    :param x: (np.array) Data. Will be scaled to [0, 1].
    :param empirical: (bool) Whether to use empirical estimation for CDF.
    :param kwargs: (dict) Setting the floor and cap of probability.
        prob_floor: (float) Probability floor.
        prob_cap: (float) Probability cap.
    :return: (func) The cumulative density function from data.
    """

    # Make sure it is an np.array.
    x = np.array(x)

    prob_floor = kwargs.get('prob_floor', 0.00001)
    prob_cap = kwargs.get('prob_cap', 0.99999)

    if empirical:
        # Use empirical cumulative density function on data.
        fitted_cdf = lambda data: max(min(ECDF(x)(data), prob_cap), prob_floor)
        # Vectorize so it works on an np.array.
        v_fitted_cdf = np.vectorize(fitted_cdf)
        return v_fitted_cdf

    return None

def construct_ecdf_lin(train_data: np.array, upper_bound: float = 1-1e-5, lower_bound: float = 1e-5) -> Callable:
    """
    Construct an empirical cumulative density function with linear interpolation between data points.

    The function it returns agrees with the ECDF function from statsmodels in values, but also applies linear
    interpolation to fill the gap.
    Features include: Allowing training data to have nan values; Allowing the cumulative density output to have an
    upper and lower bound, to avoid singularities in some applications with probability 0 or 1.

    :param train_data: (np.array) The data to train the output ecdf function.
    :param upper_bound: (float) The upper bound value for the returned ecdf function.
    :param lower_bound: (float) The lower bound value for the returned ecdf function.
    :return: (Callable) The constructed ecdf function.
    """

    train_data_np = np.array(train_data)  # Convert to numpy array for the next step in case the input is not.
    train_data_np = train_data_np[~np.isnan(train_data_np)]  # Remove nan value from the array.

    step_ecdf = ECDF(train_data_np)  # train an ecdf on all training data.
    # Sorted unique elements. They are the places where slope changes for the cumulative density.
    slope_changes = np.unique(np.sort(train_data_np))
    # Calculate the ecdf at the points of slope change.
    sample_ecdf_at_slope_changes = np.array([step_ecdf(unique_value) for unique_value in slope_changes])
    # Linearly interpolate. Allowing extrapolation to catch data out of range.
    # x: unique elements in training data; y: the ecdf value for those training data.
    interp_ecdf = interp1d(slope_changes, sample_ecdf_at_slope_changes, assume_sorted=True, fill_value='extrapolate')

    # Implement the upper and lower bound the ecdf.
    def bounded_ecdf(x):
        if np.isnan(x):  # Map nan input to nan.
            result = np.NaN
        else:  # Apply the upper and lower bound.
            result = max(min(interp_ecdf(x), upper_bound), lower_bound)

        return result

    # Vectorize it to work with arrays.
    return np.vectorize(bounded_ecdf)

def to_quantile(data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Convert the data frame to quantile by row.

    Not in place. Also returns the marginal cdfs of each column. This can work with more than just 2 columns.

    The method returns:

        - quantile_data: (pd.DataFrame) The calculated quantile data in a data frame with the original indexing.
        - cdf_list: (list) The list of marginal cumulative density functions.

    :param data: (pd.DataFrame) The original data in DataFrame.
    :return: (tuple)
        quantile_data: (pd.DataFrame) The calculated quantile data in a data frame with the original indexing.
        cdf_list: (list) The list of marginal cumulative density functions.
    """

    column_count = len(data.columns)  # Number of columns.
    cdf_lst = [None] * column_count  # List to store all marginal cdf functions.
    quantile_data_lst = [None] * column_count  # List to store all quantile data in pd.Series.

    # Loop through all columns.
    for i in range(column_count):
        cdf_lst[i] = construct_ecdf_lin(data.iloc[:, i])
        quantile_data_lst[i] = data.iloc[:, i].map(cdf_lst[i])

    quantile_data = pd.concat(quantile_data_lst, axis=1)  # Form the quantile DataFrame.

    return quantile_data, cdf_lst


def sic(log_likelihood: float, n: int, k: int = 1) -> float:
    """
    Schwarz information criterion (SIC), aka Bayesian information criterion (BIC).

    :param log_likelihood: (float) Sum of log-likelihood of some data.
    :param n: (int) Number of instances.
    :param k: (int) Number of parameters estimated by max likelihood.
    :return: (float) Value of SIC.
    """

    sic_value = np.log(n)*k - 2*log_likelihood

    return sic_value


def aic(log_likelihood: float, n: int, k: int = 1) -> float:
    """
    Akaike information criterion.

    :param log_likelihood: (float) Sum of log-likelihood of some data.
    :param n: (int) Number of instances.
    :param k: (int) Number of parameters estimated by max likelihood.
    :return sic_value (float): Value of AIC.
    """

    aic_value = (2*n/(n-k-1))*k - 2*log_likelihood

    return aic_value


def hqic(log_likelihood: float, n: int, k: int = 1) -> float:
    """
    Hannan-Quinn information criterion.

    :param log_likelihood: (float) Sum of log-likelihood of some data.
    :param n: (int) Number of instances.
    :param k: (int) Number of parameters estimated by max likelihood.
    :return: (float) Value of HQIC.
    """

    hqic_value = 2*np.log(np.log(n))*k - 2*log_likelihood

    return hqic_value


def scad_penalty(x: float, gamma: float, a: float) -> float:
    """
    SCAD (smoothly clipped absolute deviation) penalty function.

    It encourages sparse solutions for fitting data to models. As a piecewise function, this implementation is
    branchless.

    :param x: (float) The variable.
    :param gamma: (float) One of the parameters in SCAD.
    :param a: (float) One of the parameters in SCAD.
    :return: (float) Evaluated result.
    """

    # Bool variables for branchless construction.
    is_linear = (np.abs(x) <= gamma)
    is_quadratic = np.logical_and(gamma < np.abs(x), np.abs(x) <= a * gamma)
    is_constant = (a * gamma) < np.abs(x)

    # Assembling parts.
    linear_part = gamma * np.abs(x) * is_linear
    quadratic_part = (2 * a * gamma * np.abs(x) - x**2 - gamma**2) / (2 * (a - 1)) * is_quadratic
    constant_part = (gamma**2 * (a + 1)) / 2 * is_constant

    return linear_part + quadratic_part + constant_part

def scad_derivative(x: float, gamma: float, a: float) -> float:
    """
    The derivative of SCAD (smoothly clipped absolute deviation) penalty function w.r.t x.

    It encourages sparse solutions for fitting data to models.

    :param x: (float) The variable.
    :param gamma: (float) One of the parameters in SCAD.
    :param a: (float) One of the parameters in SCAD.
    :return: (float) Evaluated result.
    """

    part_1 = gamma * (x <= gamma)
    part_2 = gamma * (a * gamma - x)*((a * gamma - x) > 0) / ((a - 1) * gamma) * (x > gamma)

    return part_1 + part_2

def adjust_weights(weights: np.array, threshold: float) -> np.array:
    """
    Adjust the weights of mixed copula components.

    Dropping weights smaller or equal to a given threshold, and redistribute the weight. For example, if we set the
    threshold to 0.02 and the original weight is [0.49, 0.02, 0.49], then it will be re-adjusted to [0.5, 0, 0.5].

    :param weights: (np.array) The original weights to be adjusted.
    :param threshold: (float) The threshold that a weight will be considered 0.
    :return: (np.array) The readjusted weight.
    """

    raw_weights = np.copy(weights)
    # Filter out components that have low weights. Low weights will be 0.
    filtered_weights = raw_weights * (raw_weights > threshold)
    # Normalize the filtered weights. Make the total weight a partition of [0, 1]
    scaler = np.sum(filtered_weights)
    adjusted_weights = filtered_weights / scaler

    return adjusted_weights
