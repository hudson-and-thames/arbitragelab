# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module identifies small mean-reverting portfolios out of multiple assets. The price dynamics of all assets are
assumed to follow a VAR(1) process. The portfolio selection is then realized by estimating a sparse VAR(1) coefficient
matrix and a sparse covariance matrix from which the weight of each asset in the prospective mean-reverting portfolio is
determined. Consequently, the mean-reverting properties of the portfolio will be quantified with an Ornstein-Uhlenbeck
model.
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


# pylint: disable=invalid-name
class SmallMeanRevPortfolio:
    """
    This class implements small mean-reverting portfolio selection from a group of assets and provides the following
    functionality:

    1. Least square estimate of VAR(1) coefficient matrix;
    2. Box-Tiao canonical decomposition;
    3. LASSO regression estimate of a sparse VAR(1) coefficient matrix;
    4. Graphical LASSO regression estimate of a sparse covariance matrix;
    5. Graph representation of the sparse VAR(1) coefficient matrix and the inverse of the sparse covariance matrix;
    6. Greedy search algorithm for sparse decomposition;
    7. Semidefinite relaxation algorithm for sparse decomposition;
    8. Ornstein-Uhlenbeck mean reversion coefficient and half-life calculation;
    9. Visualization of the sparse covariance matrix estimation, the sparse VAR(1) coefficient matrix estimation, and
    the price of the selected portfolio.
    """

    def __init__(self, assets):
        """
        Constructor of the small mean-reverting portfolio identification module. The constructor will subtract the mean
        price of each asset from the original price such that the price processes have zero mean.

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
        ou_model = OrnsteinUhlenbeck()
        ou_model.fit(data=portfolio, data_frequency=interval, discount_rate=0., transaction_cost=0.)

        # Return the mean reversion coefficient and the half-life
        return ou_model.mu, ou_model.half_life()

    def least_square_VAR_fit(self) -> np.array:
        """
        Calculate the least square estimate of the VAR(1) matrix.

        :return: (np.array) Least square estimate of VAR(1) matrix.
        """

        # Fit VAR(1) model
        var_model = sm.tsa.VAR(self.demeaned)

        # The statsmodels package will give the least square estimate
        least_sq_est = np.squeeze(var_model.fit(1).coefs, axis=0)

        return least_sq_est

    def box_tiao(self, threshold: int = 7) -> np.array:
        """
        Perform Box-Tiao canonical decomposition on the assets dataframe.

        :param threshold: (int) Round precision cutoff threshold. For example, a threshold of n means that a number less
            than :math:`10^{-n}` will be treated as zero.
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
    def greedy_search(cardinality: int, matrix_A: np.array, matrix_B: np.array, threshold: int = 7) -> np.array:
        """
        Greedy search algorithm for sparse decomposition.

        :param cardinality: (int) Number of assets to include in the portfolio.
        :param matrix_A: (np.array) Matrix :math:`A^T \\Gamma A`, where A is the estimated VAR(1) coefficient matrix,
            where :math:`\\Gamma` is the estimated covariance matrix.
        :param matrix_B: (np.array) Matrix :math:`\\Gamma`, the estimated covariance matrix.
        :param threshold: (int) Round precision cutoff threshold. For example, a threshold of n means that a number less
            than :math:`10^{-n}` will be treated as zero.
        :return: (np.array) Weight of each selected assets.
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

        return np.around(selected_weights, threshold)

    @staticmethod
    def sdp_relax(cardinality: int, matrix_A: np.array, matrix_B: np.array, threshold: int = 7) -> np.array:
        """
        Semidefinite relaxation algorithm for sparse decomposition.

        :param cardinality: (int) Number of assets to form the portfolio.
        :param matrix_A: (np.array) Matrix :math:`A^T \\Gamma A`, where A is the estimated VAR(1) coefficient matrix,
            where :math:`\\Gamma` is the estimated covariance matrix.
        :param matrix_B: (np.array) Matrix :math:`\\Gamma`, the estimated covariance matrix.
        :param threshold: (int) Round precision cutoff threshold. For example, a threshold of n means that a number less
            than :math:`10^{-n}` will be treated as zero.
        :return: (np.array) Weight of each selected assets.
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

        return np.around(weights, threshold)

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
        print(coefs_nums)

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

    def LASSO_VAR_fit(self, alpha: float, multi_task_lasso: bool = True, max_iter: int = 1000,
                      threshold: int = 10) -> np.array:
        """
        Fit the LASSO model using the optimized alpha to yield a sparse VAR(1) coefficient matrix estimate.

        :param alpha: (float) Optimized l1-regularization coefficient.
        :param multi_task_lasso: (bool) If True, use multi-task LASSO for sparse estimate, where the LASSO will yield
            full columns of zeros; otherwise, do LASSO column-wise.
        :param max_iter: (int) Maximum number of iterations of LASSO regression.
        :param threshold: (int) Round precision cutoff threshold. For example, a threshold of n means that a number less
            than :math:`10^{-n}` will be treated as zero.
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
        VAR_estimate = lasso_model.fit(data_lag, data_now).coef_

        # Return the best fit for sparse estimate
        return np.around(VAR_estimate, threshold)

    def covar_sparse_tuning(self, max_iter: int = 1000, alpha_min: float = -5., alpha_max: float = 0.,
                            n_alphas: int = 100, clusters: int = 3) -> float:
        """
        Tune the regularization parameter (alpha) of the graphical LASSO model for a sparse estimate of the covariance
        matrix.

        :param max_iter: (int) Maximum number of iterations for graphical LASSO fit.
        :param alpha_min: (float) Minimum regularization parameter.
        :param alpha_max: (float) Maximum regularization parameter.
        :param n_alphas: (int) Number of regularization parameter for parameter search.
        :param clusters: (int) Number of smaller clusters desired from the precision matrix.
            The higher the number, the larger the best alpha will be. This parameter cannot exceed the number of assets.
        :return: (float) Optimal alpha to split the graph representation of the inverse covariance matrix into
            designated number of clusters.
        """
        # Check parameter validity
        if clusters > self.assets.shape[1]:
            raise ValueError("The number of clusters cannot exceed the number of assets.")

        # Set up the parameter space for the regularization parameter
        alphas = np.logspace(alpha_min, alpha_max, n_alphas)

        # Fit the graphical LASSO model
        for alpha in alphas:
            edge_model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
            edge_model.fit(self.demeaned)

            # Retrieve the precision matrix (inverse of sparse covariance matrix) as the graph adjacency matrix
            adj_matrix = np.copy(edge_model.precision_)

            # Graph should have no self loop, so we need to replace the diagonal with zeros for adjacency matrix
            np.fill_diagonal(adj_matrix, 0)

            # Assign one to non-zero elements
            adj_matrix[adj_matrix != 0] = 1

            # Check if the graph formed by the sparse covariance estimate has the desired amount of clusters
            graph = nx.from_numpy_array(adj_matrix)
            if nx.number_connected_components(graph) == clusters:
                return alpha

        # The procedure failed to find an optimal alpha, raise exception
        raise ValueError("The regularization coefficient (alpha) range selected cannot meet the "
                         "chordal graph requirement. Please try larger alphas for a sparser estimate.")

    def covar_sparse_fit(self, alpha: float, max_iter: int = 1000, threshold: int = 10) -> Tuple[np.array, np.array]:
        """
        Fit the graphical LASSO model using the optimized alpha to yield a sparse covariance matrix estimate.

        :param alpha: (float) Optimized regularization coefficient of graphical LASSO.
        :param max_iter: (int) Maximum number of iterations for graphical LASSO fit.
        :param threshold: (int) Round precision cutoff threshold. For example, a threshold of n means that a number less
            than :math:`10^{-n}` will be treated as zero.
        :return: (np.array, np.array) Sparse estimate of covariance matrix; inverse of the sparse covariance matrix,
            i.e. precision matrix as graph representation.
        """

        # Fit graphical LASSO model
        edge_model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
        edge_model.fit(self.demeaned)

        # Return the sparse estimate of the covariance matrix and its inverse
        return np.around(edge_model.covariance_, threshold), np.around(edge_model.precision_, threshold)

    def find_clusters(self, precision_matrix: np.array, var_estimate: np.array) -> nx.Graph:
        """
        Use the intersection of the graph :math:`\\Gamma^{-1}` and the graph :math:`A^T A` to pinpoint the clusters of
        assets to perform greedy search or semidefinite relaxation on.

        :param precision_matrix: (np.array) The inverse of the estimated sparse covariance matrix.
        :param var_estimate: (np.array) The sparse estimate of VAR(1) coefficient matrix.
        :return: (networkx.Graph) A graph representation of the clusters.
        """

        # Construct the graph based on covariance matrix estimate
        covar_graph = nx.from_numpy_array(precision_matrix)
        VAR_graph = nx.from_numpy_array(var_estimate)

        # Relabel the graph nodes with asset names
        mapping = {x: y.split()[0] for x, y in enumerate(list(self.assets.columns))}
        covar_graph = nx.relabel_nodes(covar_graph, mapping)
        VAR_graph = nx.relabel_nodes(VAR_graph, mapping)

        # Return the intersection of the two graph
        return nx.intersection(covar_graph, VAR_graph)

    @staticmethod
    def check_symmetric(matrix: np.array, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """
        Check if a matrix is symmetric.

        :param matrix: (np.array) The matrix under inspection.
        :param rtol: (float) Relative tolerance for np.allclose.
        :param atol: (float) Absolute tolerance for np.allclose.
        :return: (bool) True if the matrix symmetric, False otherwise.
        """
        return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

    @staticmethod
    def is_pos_def(matrix: np.array) -> bool:
        """
        Check if a matrix is positive definite.

        :param matrix: (np.array) The matrix under inspection.
        :return: (bool) True if the matrix is positive definite, False otherwise.
        """
        return np.all(np.linalg.eigvals(matrix + matrix.T) > 0)
