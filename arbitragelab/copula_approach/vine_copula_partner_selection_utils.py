# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Utils for implementing partner selection approaches for vine copulas.
"""
# pylint: disable = invalid-name
from contextlib import suppress
import itertools
import numpy as np
import pandas as pd
import scipy

from statsmodels.distributions.empirical_distribution import ECDF

def get_quantiles_data(col: pd.Series):
    """
    Returns ranked quantiles from returns.
    :param col: (pd.Series) returns data for a single stock.
    :return: ranked returns from quantiles.
    """

    return ECDF(col)(col)


def get_sector_data(quadruple: list, constituents: pd.DataFrame):
    """
    Function returns Sector and Sub sector information for all tickers in quadruple.
    :param quadruple: (list) List of four tickers.
    :param constituents: (pd.DataFrame) Dataframe consisting of sector data for all tickers in universe.
    :return: Data corresponding to the quadruple.
    """

    with suppress(KeyError):
        return constituents.loc[quadruple, ['Security', 'GICS Sector', 'GICS Sub-Industry']]


def get_sum_correlations_vectorized(data_subset: pd.DataFrame, all_possible_combinations: np.array) -> tuple:
    """
    Helper function for traditional approach to partner selection.

    Calculates sum of pairwise correlations between all stocks in each quadruple.
    Returns the quadruple with the largest sum.

    :param data_subset: (pd.DataFrame) Dataset storing correlations data.
    :param all_possible_combinations: (np.array) Array of indices of 19600 quadruples.
    :return: (tuple) Final quadruple, corresponding measure.
    """

    # We use the combinations as an index
    corr_matrix = data_subset.values[:, all_possible_combinations]
    # corr_matrix has shape of (51, 19600, 4)
    # We now use np.take_along_axis to get the shape (4,19600,4), then we can sum the first and the last dimension
    corr_sums = np.sum(np.take_along_axis(corr_matrix, all_possible_combinations.T[..., np.newaxis], axis=0),
                       axis=(0, 2))  # Shape: (19600,1)

    d = all_possible_combinations.shape[-1]
    corr_sums = (corr_sums - d) / 2

    # Return the maximum index for the sums.
    max_index = np.argmax(corr_sums)
    final_quadruple = data_subset.columns[list(all_possible_combinations[max_index])].tolist()

    return final_quadruple, corr_sums[max_index]


def multivariate_rho_vectorized(data_subset: pd.DataFrame, all_possible_combinations: np.array) -> tuple:
    """
    Helper function for extended approach to partner selection.

    Calculates 3 proposed estimators for high dimensional generalization for Spearman's rho.
    These implementations are present in
    `Multivariate extensions of Spearman’s rho and related statistics. (2007)
    <https://wisostat.uni-koeln.de/fileadmin/sites/statistik/pdf_publikationen/SchmidSchmidtSpearmansRho.pdf>`__
    by Schmid, F., Schmidt, R.

    Returns the quadruple with the largest measure.

    :param data_subset: (pd.DataFrame) Dataset storing ranked returns data.
    :param all_possible_combinations: (np.array) Array of indices of 19600 quadruples.
    :return: (tuple) Final quadruple, corresponding measure.
    """

    quadruples_combinations_data = data_subset.values[:, all_possible_combinations]  # Shape: (n, 19600, d)

    n, _, d = quadruples_combinations_data.shape  # n : Number of samples, d : Number of stocks
    h_d = (d + 1) / (2 ** d - d - 1)

    # Calculating the first estimator of multivariate rho
    sum_1 = np.product(1 - quadruples_combinations_data, axis=-1).sum(axis=0)
    rho_1 = h_d * (-1 + (2 ** d / n) * sum_1)

    # Calculating the second estimator of multivariate rho
    sum_2 = np.product(quadruples_combinations_data, axis=-1).sum(axis=0)
    rho_2 = h_d * (-1 + (2 ** d / n) * sum_2)

    # Calculating the third estimator of multivariate rho
    pairs = np.array(list(itertools.combinations(range(d), 2)))
    k, l = pairs[:, 0], pairs[:, 1]
    sum_3 = ((1 - quadruples_combinations_data[:, :, k]) * (1 - quadruples_combinations_data[:, :, l])).sum(axis=(0, 2))
    dc2 = scipy.special.comb(d, 2, exact=True)
    rho_3 = -3 + (12 / (n * dc2)) * sum_3

    quadruples_scores = (rho_1 + rho_2 + rho_3) / 3
    # The quadruple scores have the shape of (19600,1) now
    max_index = np.argmax(quadruples_scores)

    final_quadruple = data_subset.columns[list(all_possible_combinations[max_index])].tolist()

    return final_quadruple, quadruples_scores[max_index]


def diagonal_measure_vectorized(data_subset: pd.DataFrame, all_possible_combinations: np.array) -> tuple:
    """
    Helper function for geometric approach to partner selection.

    Calculates the sum of Euclidean distances from the relative ranks to the (hyper-)diagonal
    in four dimensional space for each quadruple of a target stock.
    Returns the quadruple with the smallest measure.

    :param data_subset: (pd.DataFrame) Dataset storing ranked returns data.
    :param all_possible_combinations: (np.array) Array of indices of 19600 quadruples.
    :return: (tuple) Final quadruple, corresponding measure
    """

    quadruples_combinations_data = data_subset.values[:, all_possible_combinations]
    d = quadruples_combinations_data.shape[-1]  # Shape: (n, 19600, d) where n: Number of samples, d: Number of stocks

    line = np.ones(d)
    # Einsum is great for specifying which dimension to multiply together
    # this extends the distance method for all 19600 combinations
    pp = (np.einsum("ijk,k->ji", quadruples_combinations_data, line) / np.linalg.norm(line))
    pn = np.sqrt(np.einsum('ijk,ijk->ji', quadruples_combinations_data, quadruples_combinations_data))
    distance_scores = np.sqrt(pn ** 2 - pp ** 2).sum(axis=1)
    min_index = np.argmin(distance_scores)
    final_quadruple = data_subset.columns[list(all_possible_combinations[min_index])].tolist()

    return final_quadruple, distance_scores[min_index]


def extremal_measure(u: pd.DataFrame, co_variance_matrix: np.array):
    """
    Helper function to calculate chi-squared test statistic based on p-dimensional Nelsen copulas.

    Specifically, proposition 3.3 from `Mangold (2015) <https://www.statistik.rw.fau.de/files/2016/03/IWQW-10-2015.pdf>`__
    is implemented for 4 dimensions.

    :param u: (pd.DataFrame) Ranked returns of stocks in quadruple.
    :param co_variance_matrix: (np.array) Covariance matrix.
    :return: (float) Test statistic.
    """

    u = u.to_numpy()
    n = u.shape[0]

    # Calculating array T_(4,n) from proposition 3.3
    t = t_calc(u).mean(axis=1).reshape(-1, 1)  # Shape :(16, 1), Taking the mean w.r.t n.
    # Calculating the final test statistic.
    t_test_statistic = n * np.matmul(t.T, np.matmul(co_variance_matrix, t))

    return t_test_statistic[0, 0]


def get_co_variance_matrix(d: int):
    """
    Calculates 2**d x 2**d dimensional covariance matrix. Since the matrix is symmetric, only the integrals
    in the upper triangle are calculated. The remaining values are filled from the transpose.

    :param d: (int) Number of stocks.
    """

    args = [[1,2]] * d

    co_variance_matrix = np.zeros((2**d, 2**d))
    for i, l1 in enumerate(itertools.product(*args)):
        for j, l2 in enumerate(itertools.product(*args)):
            if j < i:
                # Integrals in lower triangle are skipped.
                continue

            # Numerical Integration of d dimensions.
            co_variance_matrix[i, j] = scipy.integrate.nquad(variance_integral_func, [(0, 1)] * d, args=(l1, l2))[0]

    inds = np.tri(2**d, k=-1, dtype=bool)  # Storing the indices of elements in lower triangle.
    co_variance_matrix[inds] = co_variance_matrix.T[inds]

    return np.linalg.inv(co_variance_matrix)


def t_calc(u):
    """
    Calculates T_(d,n) as seen in proposition 3.3. Each of the 2**d rows in the array are appended to output and
    returned as numpy array.

    :param u: (pd.DataFrame) Ranked returns of stocks in quadruple.
    :return: (np.array) Array of Shape (2**d, n).
    """

    d = u.shape[1]
    args = [[1,2]] * d
    output = []
    for l in itertools.product(*args):
        # Equation form for each one of u1,u2,u3,u4,... after partial differentials are calculated and multiplied
        # together
        res = 1
        for ind in range(d):
            res *= func(u[:, ind], l[ind])
        output.append(res)

    return np.array(output)  # Shape (2**d, n)


def func(t: np.array, value: int):
    """
    Function returns equation form of respective variable after partial differentiation.
    All variables in the differential equations are in one of two forms.

    :param t: (np.array) Variable.
    :param value: (int) Flag denoting equation form of variable.
    :return: (float) Differentiation result.
    """

    res = None
    if value == 1:
        res =  (t - 1) * (3 * t - 1)
    if value == 2:
        res =  t * (2 - 3 * t)

    return res


def variance_integral_func(*args):
    """
    Calculates Integrand for covariance matrix calculation.

    :param args: (list) Given parameters.
    :return: (float) Integrand value.
    """

    l1 = args[-2]
    l2 = args[-1]

    res = 1
    for ind in range(len(args[:-2])):
        res *= func(args[ind], l1[ind]) * func(args[ind], l2[ind])

    return res
