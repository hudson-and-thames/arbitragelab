# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Implementation of distance using the Generic Non-Parametric Representation approach from "Some contributions to the
clustering of financial time series and applications to credit default swaps" by Gautier Marti
https://www.researchgate.net/publication/322714557
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import ot


# pylint: disable=invalid-name
def spearmans_rho(x: np.array, y: np.array) -> float:
    """
    Calculates a statistical estimate of Spearman's rho - a copula-based dependence measure.

    Formula for calculation:
    rho = 1 - (6)/(T*(T^2-1)) * Sum((X_t-Y_t)^2)

    It is more robust to noise and can be defined if the variables have an infinite second moment.
    This statistic is described in more detail in the work by Gautier Marti
    https://www.researchgate.net/publication/322714557 (p.54)

    This method is a wrapper for the scipy spearmanr function. For more details about the function and its parameters,
    please visit scipy documentation
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html

    :param x: (np.array/pd.Series) X vector
    :param y: (np.array/pd.Series) Y vector (same number of observations as X)
    :return: (float) Spearman's rho statistical estimate
    """

    # Coefficient calculationS
    rho, _ = spearmanr(x, y)

    return rho


def gpr_distance(x: np.array, y: np.array, theta: float) -> float:
    """
    Calculates the distance between two Gaussians under the Generic Parametric Representation (GPR) approach.

    According to the original work https://www.researchgate.net/publication/322714557 (p.70):
    "This is a fast and good proxy for distance d_theta when the first two moments ... predominate". But it's not
    a good metric for heavy-tailed distributions.

    Parameter theta defines what type of information dependency is being tested:
    - for theta = 0 the distribution information is tested
    - for theta = 1 the dependence information is tested
    - for theta = 0.5 a mix of both information types is tested

    With theta in [0, 1] the distance lies in range [0, 1] and is a metric. (See original work for proof, p.71)

    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector (same number of observations as X).
    :param theta: (float) Type of information being tested. Falls in range [0, 1].
    :return: (float) Distance under GPR approach.
    """

    # Calculating the GPR distance
    distance = theta * (1 - spearmans_rho(x, y)) / 2 + \
               (1 - theta) * (1 - ((2 * x.std() * y.std()) / (x.std()**2 + y.std()**2))**(1/2) *
                              np.exp(- (1 / 4) * (x.mean() - y.mean())**2 / (x.std()**2 + y.std()**2)))

    return distance**(1/2)


def gnpr_distance(x: np.array, y: np.array, theta: float, n_bins: int = 50) -> float:
    """
    Calculates the empirical distance between two random variables under the Generic Non-Parametric Representation
    (GNPR) approach.

    Formula for the distance is taken from https://www.researchgate.net/publication/322714557 (p.72).

    Parameter theta defines what type of information dependency is being tested:
    - for theta = 0 the distribution information is tested
    - for theta = 1 the dependence information is tested
    - for theta = 0.5 a mix of both information types is tested

    With theta in [0, 1] the distance lies in the range [0, 1] and is a metric.
    (See original work for proof, p.71)

    This method is modified as it uses 1D Optimal Transport Distance to measure
    distribution distance. This solves the issue of defining support and choosing
    a number of bins. The number of bins can be given as an input to speed up calculations.
    Big numbers of bins can take a long time to calculate.

    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector (same number of observations as X).
    :param theta: (float) Type of information being tested. Falls in range [0, 1].
    :param n_bins: (int) Number of bins to use to split the X and Y vector observations.
        (100 by default)
    :return: (float) Distance under GNPR approach.
    """

    # Number of observations
    num_obs = x.shape[0]

    # Calculating the d_1 distance
    dist_1 = 3 / (num_obs * (num_obs ** 2 - 1)) * (np.power(x - y, 2).sum())

    # Binning observations
    x_binned = pd.Series(np.histogram(x, bins=n_bins)[0]) / num_obs
    y_binned = pd.Series(np.histogram(y, bins=n_bins)[0]) / num_obs

    # Bin positions
    bins = np.linspace(0, 1, n_bins)

    # Loss matrix
    loss_matrix = ot.dist(bins.reshape((n_bins, 1)), bins.reshape((n_bins, 1)))
    loss_matrix /= loss_matrix.max()

    # Optimal transportation matrix
    ot_matrix = ot.emd(x_binned.sort_values(), y_binned.sort_values(), loss_matrix)

    # Optimal transport distance
    dist_0 = np.trace(np.dot(np.transpose(ot_matrix), loss_matrix))

    # Calculating the GNPR distance
    distance = theta * dist_1 + (1 - theta) * dist_0

    return distance**(1/2)
