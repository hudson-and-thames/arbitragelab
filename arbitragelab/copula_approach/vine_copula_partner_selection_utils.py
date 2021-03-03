"""
Utils for implementing partner selection approaches for vine copulas.
"""
# pylint: disable = invalid-name
import itertools
import numpy as np
import pandas as pd
import scipy


def get_sector_data(quadruple, constituents):
    """
    Function returns Sector and Sub sector information from tickers.
    """
    try:
        return constituents.loc[quadruple, ['Security', 'GICS Sector', 'GICS Sub-Industry']]
    except KeyError:
        return None


def get_sum_correlations(corr_matrix, quadruple: list) -> float:
    """
    Helper function for traditional approach to partner selection.
    Calculates sum of pairwise correlations between all stocks in the quadruple.

    :param corr_matrix: Correlation Matrix
    :param quadruple: (list) : list of 4 stock tickers
    :return: (float) : sum of correlations
    """

    return (corr_matrix.loc[quadruple, quadruple].sum().sum() - len(quadruple)) / 2


def multivariate_rho(u: pd.DataFrame) -> float:
    """
    Helper function for extended approach to partner selection. Calculates 3 proposed estimators for
    high dimensional generalization for Spearman's rho. These implementations are present in
    Schmid, F., Schmidt, R., 2007. Multivariate extensions of Spearman’s rho and related statis-tics.

    :param u: (pd.DataFrame) : ranked returns of quadruple
    :return: (float) : mean of the three estimators of multivariate rho
    """

    n, d = u.shape  # n : Number of samples, d : Number of stocks
    h_d = (d + 1) / ((2 ** d) - d - 1)

    # # Calculating the first estimator of multivariate rho
    sum_1 = np.prod(1 - u, axis=1).sum()
    rho_1 = h_d * (-1 + (((2 ** d) / n) * sum_1))

    # # Calculating the second estimator of multivariate rho
    sum_2 = np.prod(u, axis=1).sum()
    rho_2 = h_d * (-1 + (((2 ** d) / n) * sum_2))

    # Calculating the third estimator of multivariate rho
    pairs = itertools.combinations(range(u.shape[-1]), 2)
    sum_3 = np.sum([(1 - u.iloc[:, k]) * (1 - u.iloc[:, l]) for (k, l) in pairs])
    dc2 = scipy.special.comb(d, 2, exact=True)
    rho_3 = -3 + (12 / (n * dc2)) * sum_3

    return (rho_1 + rho_2 + rho_3) / 3


def distance_calc(a, ba, x):
    """
    Helper function to calculate Euclidean distance between Point and diagonal used to calculate diagonal measure.
    :param a: Origin (0,0,0,0)
    :param ba: d-dimensional Diagonal
    :param x: list of points on d-dimensions
    :return:
    """
    pa = x - a  # Shape : (n,d), pa represents vector from point p to point a
    t = np.dot(pa, ba) / np.dot(ba.T, ba)
    return np.linalg.norm(pa - np.dot(t, ba.T), ord=2, axis=1)


def diagonal_measure(points) -> float:
    """
    Helper function for geometric approach to partner selection. Calculates the sum of Euclidean distances
    from the relative ranks to the (hyper-)diagonal in four dimensional space for a given target stock.

    Reference for calculating the euclidean distance from a point to diagonal
    https://in.mathworks.com/matlabcentral/answers/76727-how-can-i-calculate-distance-between-a-point-and-a-line-in-4d-or-space-higer-than-3d

    :param points: (pd.DataFrame) : ranked returns of 4 stock tickers
    :return total_distance: (float) : total euclidean distance
    """
    d = points.shape[-1]  # d : Number of stocks
    a = np.zeros((1, d))  # Point a denotes origin which is present on hyper-diagonal
    b = np.ones((1, d))  # Point b denotes point (1,1,1,1) on hyper-diagonal
    ba = (b - a).T  # ba represents the hyper-diagonal
    total_distance = distance_calc(a, ba, points).sum()
    return total_distance


def extremal_measure(u, co_variance_matrix):
    """
    Helper function to calculate chi-squared test statistic based on p-dimensional Nelsen copulas.
    Specifically, proposition 3.3 from Mangold (2015) is implemented for 4 dimensions.
    :param u: (pd.DataFrame) : ranked returns of stocks in quadruple.
    :param co_variance_matrix: (np.array) : Covariance matrix
    :return: test statistic
    """
    u = u.to_numpy()
    n = u.shape[0]

    # Calculating array T_(4,n) from proposition 3.3
    t = t_calc(u).mean(axis=1).reshape(-1, 1)  # Shape : (16, 1), Taking the mean w.r.t n
    # Calculating the final test statistic
    t_test_statistic = n * np.matmul(t.T, np.matmul(co_variance_matrix, t))
    return t_test_statistic[0, 0]


def get_co_variance_matrix():
    """
    Calculates 16x16 dimensional covariance matrix. Since the matrix is symmetric, only the integrals
    in the upper triangle are calculated. The remaining values are filled from the transpose.
    """
    co_variance_matrix = np.zeros((16, 16))
    for i, l1 in enumerate(itertools.product([1, 2], [1, 2], [1, 2], [1, 2])):
        for j, l2 in enumerate(itertools.product([1, 2], [1, 2], [1, 2], [1, 2])):
            if j < i:
                # Integrals in lower triangle are skipped.
                continue

            # Numerical Integration of 4 dimensions
            co_variance_matrix[i, j] = scipy.integrate.nquad(variance_integral_func, [(0, 1)] * 4, args=(l1, l2))[0]

    inds = np.tri(16, k=-1, dtype=bool)  # Storing the indices of elements in lower triangle.
    co_variance_matrix[inds] = co_variance_matrix.T[inds]
    return np.linalg.inv(co_variance_matrix)


def t_calc(u):
    """
    Calculates T_(4,n) as seen in proposition 3.3. Each of the 16 rows in the array are appended to output and
    returned as numpy array.
    :param u:
    :return: (np.array) of Shape (16, n)
    """
    output = []
    for l in itertools.product([1, 2], [1, 2], [1, 2], [1, 2]):
        # Equation form for each one of u1,u2,u3,u4 after partial differentials are calculated and multiplied
        # together.
        res = func(u[:, 0], l[0]) * func(u[:, 1], l[1]) * func(u[:, 2], l[2]) * func(u[:, 3], l[3])
        output.append(res)

    return np.array(output)  # Shape (16, n)


def func(t, value):
    """
    Function returns equation form of respective variable after partial differentiation.
    All variables in the differential equations are in one of two forms.
    :param t: (np.array) Variable
    :param value: (int) Flag denoting equation form of variable
    :return:
    """

    res = None
    if value == 1:
        res =  (t - 1) * (3 * t - 1)
    if value == 2:
        res =  t * (2 - 3 * t)

    return res


def variance_integral_func(u1, u2, u3, u4, l1, l2):
    """
    Calculates Integrand for covariance matrix calculation.
    """
    return func(u1, l1[0]) * func(u2, l1[1]) * func(u3, l1[2]) * func(u4, l1[3]) * \
           func(u1, l2[0]) * func(u2, l2[1]) * func(u3, l2[2]) * func(u4, l2[3])
