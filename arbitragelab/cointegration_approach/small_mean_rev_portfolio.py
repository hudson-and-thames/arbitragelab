# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module identifies a small mean-reverting portfolio out of multiple assets by sparse estimation of covariance
matrix.
"""

from typing import Tuple

import cvxpy as cp
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from sklearn.covariance import GraphicalLasso
from sklearn.linear_model import lasso_path, Lasso, MultiTaskLasso
from sklearn.preprocessing import normalize

from arbitragelab.optimal_mean_reversion import OrnsteinUhlenbeck


class SmallMeanRevPortfolio:
    """

    """
    def __init__(self, assets):
        """
        Constructor of the small mean-reverting portfolio identification module.

        :param assets: (pd.DataFrame) The price history of each asset.
        """
        self.__assets = assets
        self.__demeaned = assets - assets.mean(axis=0)

    @property
    def assets(self) -> pd.DataFrame:
        """
        Getter for the class attribute "assets".

        :return: (pd.DataFrame) The price history of each asset.
        """
        return self.__assets

    @property
    def demeaned(self) -> pd.DataFrame:
        """
        Getter for the class attribute "demeaned".

        :return: (pd.DataFrame) The processed price history of each asset with zero mean.
        """
        return self.__demeaned

    @staticmethod
    def mean_rev_coeff(weights: np.array, assets: pd.DataFrame, interval: str = 'D') -> Tuple[float, float]:
        """
        Calculate the mean reversion coefficient assuming that the portfolio follows an Ornstein-Uhlenbeck (OU) process.

        :param weights: (np.array) The weightings for each asset.
        :param assets: (pd.DataFrame) The price history of each asset.
        :param interval: (str) The time interval, or the frequency, of the price data.
        :return: (float, float) Mean reversion coefficient; half life of the OU process.
        """

        # Check if the shape of the weights and the assets match
        if weights.shape[0] != assets.shape[1]:
            raise np.linalg.LinAlgError("Dimensions do not match!")

        # From the portfolio by the weights
        portfolio = assets @ weights

        # Fit the OU model
        ou_model = OrnsteinUhlenbeck
        ou_model.fit(portfolio, interval, discount_rate=0., transaction_cost=0.)

        # Return the mean reversion coefficient and the half-life
        return ou_model.mu, ou_model.half_life()

    def least_square_VAR_fit(self):
        """
        Calculate the least square estimate of the VAR(1) matrix.

        :return: (np.array) Least square estimate of VAR(1) matrix.
        """

        # Fit VAR(1) model
        var_model = sm.tsa.VAR(self.demeaned)

        # The statsmodels package will give the least square estimate
        least_sq_est = np.squeeze(var_model.fit(1).coefs, axis=0)

        return least_sq_est

    def box_tiao(self) -> np.array:
        """
        Perform Box-Tiao canonical decomposition on the assets dataframe.

        :return: (np.array) The weighting of each asset in the portfolio. There will be N decompositions for N assets,
            where each column vector corresponds to one portfolio. The
        """

        # Calculate the least square estimate of the price with VAR(1) model
        least_sq_est = self.least_square_VAR_fit()

        # Construct the matrix from which the eigenvectors need to be computed
        covar = self.demeaned.cov()
        box_tiao_matrix = np.linalg.inv(covar) @ least_sq_est @ covar @ least_sq_est.T

        # Calculate the eigenvectors and sort by eigenvalue
        eigvals, eigvecs = np.linalg.eig(box_tiao_matrix)

        # Sort the eigenvectors by eigenvalues by descending order
        bt_eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]

        # Return the weights
        return bt_eigvecs

    @staticmethod
    def greedy_search(cardinality: int, matrix_A: np.array, matrix_B: np.array) -> np.array:
        """

        :param cardinality: (int) Number of assets to form the portfolio.
        :param matrix_A: (np.array) Matrix A.T @ Gamma @ A, where A is the estimated VAR(1) coefficient matrix, and
            Gamma is the estimated covariance matrix.
        :param matrix_B: (np.array) Matrix Gamma, where Gamma is the estimated covariance matrix.
        :return: (np.array) Weight of each selected assets
        """

        # Use a list to store all selected assets
        selected = []
        selected_weights = None

        # Use a set to record which assets have already been selected
        candidates = set(range(matrix_B.shape[0]))

        # Start greedy search
        for k in range(cardinality):
            # Record the maximum value of the target function of the generalized eigenvalue problem
            max_gen_eig_ratio = np.NINF

            # Start greedy search
            for support in list(candidates):
                cand = selected + [support]

                # Calculate the current matrix_A and matrix_B according to the support
                cur_matrix_A = matrix_A[cand, :][:, cand]
                cur_matrix_B = matrix_B[cand, :][:, cand]

                # Special case: when only one asset is selected, need to reshape the numpy array into a matrix
                if len(cand) == 1:
                    cur_matrix_A = cur_matrix_A.reshape(-1, 1)
                    cur_matrix_B = cur_matrix_B.reshape(-1, 1)

                # Solve the generalized eigenvalue problem
                eigval, eigvec = scipy.linalg.eigh(cur_matrix_A, cur_matrix_B)

                # Construct the weighting of the portfolio based on the current support
                weight = np.zeros((matrix_B.shape[0], 1))

                # Choose the last eigenvector due to scipy.linalg.eigh sorts eigenvectors by ascending eigenvalues
                # Need to normalized the eigenvector to norm one
                weight[cand, :] = normalize(eigvec[:, -1].reshape(-1, 1), axis=0, norm='l2')

                # Store the maximum and the support corresponds to the maximum
                gen_eig_ratio = np.squeeze((weight.T @ cur_matrix_A @ weight) / (weight.T @ cur_matrix_B @ weight))
                if gen_eig_ratio > max_gen_eig_ratio:
                    max_gen_eig_ratio = gen_eig_ratio
                    cur_support = support
                    selected_weights = weight

            # Now the best asset candidate have been included in the portfolio, remove it from the candidate list
            selected.append(cur_support)
            candidates.remove(cur_support)

        return selected_weights

    @staticmethod
    def sdp_relax(cardinality: int, matrix_A: np.array, matrix_B: np.array) -> np.array:
        """

        :param cardinality: (int) Number of assets to form the portfolio.
        :param matrix_A: (np.array) Matrix A.T @ Gamma @ A, where A is the estimated VAR(1) coefficient matrix, and
            Gamma is the estimated covariance matrix.
        :param matrix_B: (np.array) Matrix Gamma, where Gamma is the estimated covariance matrix.
        :return:
        """

        # Declare a symmetric matrix variable
        Y_dim = matrix_B.shape[0]
        Y = cp.Variable((Y_dim, Y_dim), symmetric=True)

        # Constraints for the semidefinite program (SDP)
        constraints = [
            cp.sum(cp.abs(Y)) <= cardinality * cp.trace(Y),
            cp.trace(Y) >= 0,
            cp.trace(matrix_B @ Y) == 1,
            Y >> 0
        ]

        # Solve the SDP
        cp.Problem(cp.Maximize(cp.trace(matrix_A @ Y)), constraints).solve()

        # Calculate the eigenvectors; np.linalg.eig will ensure that eigenvectors are normalized
        eigvals, eigvectors = np.linalg.eig(Y.value)

        # Get the eigenvector that corresponds to the largest eigvalue
        weights = eigvectors[:, np.argmax(eigvals)]

        return weights

    def LASSO_VAR_tuning(self, sparsity: float, multi_task_lasso: bool = False, alpha_min: float = -5.,
                         alpha_max: float = 0., n_alphas: int = 100, max_iter: int = 1000) -> float:
        """
        Tune the l1-regularization coefficient (alpha) of LASSO regression for a sparse estimate of the VAR(1) matrix.

        :param sparsity: (float) Percentage of zeros required in the VAR(1) matrix.
        :param multi_task_lasso: (bool) If True, use multi-task LASSO for sparse estimate, where the LASSO will yield
            full columns of zeros; otherwise, do LASSO column-wise.
        :param alpha_min: (float) Minimum l1-regularization coefficient.
        :param alpha_max: (float) Maximum l1-regularization coefficient.
        :param n_alphas: (int) Number of l1-regularization coefficient for the parameter search.
        :param max_iter: (int) Maximum number of iterations for LASSO regression.
        :return: (float) Minimum alpha that satisfies the sparsity requirement.
        """

        # The number of elements in the VAR(1) matrix is asset number squared
        coefs_nums = self.demeaned.shape[1] ** 2

        # Construct the current data and lag-1 data such that they have the same shape
        data_now = self.demeaned.iloc[1:]
        data_lag = self.demeaned.iloc[:-1]

        # Set up the parameter space for alpha
        alphas = np.logspace(alpha_max, alpha_min, n_alphas)

        # Set up the LASSO model and do a search on the alpha parameter space
        if multi_task_lasso:
            # Fit the multi-task LASSO model
            _, coefs_lasso, _ = lasso_path(data_lag, data_now, alphas=alphas, max_iter=max_iter)

            # Select the maximum alpha that satisfies the sparsity requirement
            non_zeros = np.count_nonzero(coefs_lasso, axis=(0, 1))
            good_alphas = alphas[non_zeros <= (1 - sparsity) * coefs_nums]

            # If no alpha satisfies the sparsity requirement, return NaN
            if good_alphas.shape == (0, ):
                best_alpha = np.Inf
            else:
                best_alpha = np.min(good_alphas)
        else:
            best_alpha = np.Inf

            # Fit the normal LASSO model
            for alpha in alphas:
                lasso_model = Lasso(alpha=alpha, max_iter=max_iter)

                # Get the coefficient column-wise
                coefs_lasso = np.array([lasso_model.fit(data_lag, y).coef_ for y in data_now.values.T])

                # Calculate the number of non-zero elements
                non_zeros = np.count_nonzero(coefs_lasso)
                if non_zeros <= (1 - sparsity) * coefs_nums:
                    # Store the current best alpha if the sparsity requirement is met
                    best_alpha = np.min([alpha, best_alpha])
                    continue

                # If the alpha is sufficiently small, stop searching
                break

        if np.isinf(best_alpha):
            raise ValueError("The l1-regularization coefficient (alpha) range selected cannot meet the "
                             "sparsity requirements. Please try larger alphas for a sparser estimate.")
        return best_alpha

    def LASSO_VAR_fit(self, alpha: float, multi_task_lasso: bool = True, max_iter: int = 1000) -> np.array:
        """
        Fit the LASSO model using the optimized alpha for a specific sparsity.

        :param alpha: (float) Optimized l1-regularization coefficient.
        :param multi_task_lasso: (bool) If True, use multi-task LASSO for sparse estimate, where the LASSO will yield
            full columns of zeros; otherwise, do LASSO column-wise.
        :param max_iter: (int) Maximum number of iterations of LASSO regression.
        :return: (np.array) Sparse estimate of VAR(1) matrix.
        """

        # Construct the current data and lag-1 data such that they have the same shape
        data_now = self.demeaned.iloc[1:]
        data_lag = self.demeaned.iloc[:-1]

        # Fit the model with the optimized alpha
        if multi_task_lasso:
            lasso_model = MultiTaskLasso(alpha=alpha, max_iter=max_iter)
        else:
            lasso_model = Lasso(alpha=alpha, max_iter=max_iter)
        VAR_estimate = lasso_model.fit(data_lag, data_now).coef

        # Return the best fit for sparse estimate
        return VAR_estimate

    def covar_sparse_tuning(self, max_iter: int = 1000, alpha_min: float = -5., alpha_max: float = 0.,
                            n_alphas: int = 100, clusters: int = 3) -> float:
        """

        :param max_iter:
        :param alpha_min:
        :param alpha_max:
        :param n_alphas:
        :param clusters: (int) Number of smaller clusters desired from the precision matrix.
            The higher the number, the larger the best alpha will be. This parameter cannot exceed the number of assets.
        :return: (float)
        """

        # Set up the parameter space for the regularization parameter
        alphas = np.logspace(alpha_min, alpha_max, n_alphas)

        # Fit the graphical LASSO model
        for alpha in alphas:
            edge_model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
            edge_model.fit(self.demeaned)

            # Retrieve the precision matrix (inverse of sparse covariance matrix) as the graph adjacency matrix representation
            adj_matrix = np.copy(edge_model.precision_)

            # Graph should have no self loop, so we need to replace the diagonal with zeros for adjacency matrix
            np.fill_diagonal(adj_matrix, 0)

            # Assign one to non-zero elements
            adj_matrix[adj_matrix != 0] = 1

            # Check if the graph formed by the sparse covariance estimate is chordal and has the desired amount of clusters
            graph = nx.from_numpy_array(adj_matrix)
            if nx.number_connected_components(graph) == clusters and nx.is_chordal(graph):
                return alpha

        # The procedure failed to find an optimal alpha, raise exception
        raise ValueError("The regularization coefficient (alpha) range selected cannot meet the "
                         "chordal graph requirement. Please try larger alphas for a sparser estimate.")

    def covar_sparse_fit(self, alpha: float, max_iter: int = 1000) -> np.array:
        """

        :param alpha:
        :param max_iter:
        :return:
        """

        # Fit graphical LASSO model
        edge_model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
        edge_model.fit(self.demeaned)

        # Return the sparse estimate of the covariance matrix
        return edge_model.covariance_

    def plot_box_tiao_portfolio(self):
        """
        Plot the portfolio value

        :return:
        """
        pass
