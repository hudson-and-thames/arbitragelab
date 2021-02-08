# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module selects sparse mean-reverting portfolios out of an asset universe. The methods implemented in this module
are based on d'Aspremont (2011) and Cuturi (2016) and include the following:

1. Box-Tiao canonical decomposition.
2. Greedy search.
3. Semidefinite relaxation.
4. Graphical LASSO regression for sparse covariance selection.
5. Column-wise LASSO and multi-task LASSO regression for sparse VAR(1) coefficient matrix estimation.
6. Semidefinite programming approach to predictability optimization under a minimum volatility constraint.
7. Semidefinite programming approach to portmanteau statistics optimization under a minimum volatility constraint.
8. Semidefinite programming approach to crossing statistics optimization under a minimum volatility constraint.
"""

from typing import Tuple
import warnings

import cvxpy as cp
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from sklearn.covariance import GraphicalLasso
from sklearn.linear_model import lasso_path, Lasso, MultiTaskLasso
from sklearn.preprocessing import normalize, StandardScaler

from arbitragelab.optimal_mean_reversion import OrnsteinUhlenbeck


class SparseMeanReversionPortfolio:
    """
    Module for sparse mean reversion portfolio selection.
    """

    def __init__(self, assets):
        """
        Constructor of the small mean-reverting portfolio identification module. The constructor will subtract the mean
        price of each asset from the original price such that the price processes have zero mean.

        :param assets: (pd.DataFrame) The price history of each asset.
        """
        self._assets = assets

        # Demeaned assets
        self._demeaned = assets - assets.mean(axis=0)

        # Zero mean and unit variance for each column
        scaler = StandardScaler()
        standard_data = pd.DataFrame(scaler.fit_transform(assets))
        standard_data.index = assets.index
        self._standardized = standard_data

    @property
    def assets(self) -> pd.DataFrame:
        """
        Getter for the asset price data.

        :return: (pd.DataFrame) The price history of each asset.
        """
        return self._assets

    @property
    def demeaned(self) -> pd.DataFrame:
        """
        Getter for the demeaned price data.

        :return: (pd.DataFrame) The processed price history of each asset with zero mean.
        """
        return self._demeaned

    @property
    def standardized(self) -> pd.DataFrame:
        """
        Getter for the standardized price data.

        :return: (pd.DataFrame) The stnadardized price history of each asset with zero mean and unit variance.
        """
        return self._standardized

    @staticmethod
    def mean_rev_coeff(weights: np.array, assets: pd.DataFrame, interval: str = 'D') -> Tuple[float, float]:
        """
        Calculate the Ornstein-Uhlenbeck model mean reversion speed and half-life.

        :param weights: (np.array) The weightings for each asset.
        :param assets: (pd.DataFrame) The price history of each asset.
        :param interval: (str) The time interval, or the frequency, of the price data.
        :return: (float, float) Mean reversion coefficient; half life of the OU process.
        """

        # Check if the shape of the weights and the assets match
        if weights.shape[0] != assets.shape[1]:
            raise np.linalg.LinAlgError("Dimensions do not match!")

        half_life_conversion = {
            'D': 252,
            'M': 12,
            'Y': 1
        }

        # From the portfolio by the weights
        portfolio = assets @ weights

        # Fit the OU model
        ou_model = OrnsteinUhlenbeck()
        ou_model.fit(data=portfolio, data_frequency=interval, discount_rate=0., transaction_cost=0.)

        # Return the mean reversion coefficient and the half-life
        return ou_model.mu, ou_model.half_life() * half_life_conversion[interval]

    def autocov(self, nlags: int, symmetrize: bool = True, use_standardized: bool = True) -> np.array:
        """
        Calculate the autocovariance matrix.

        :param nlags: Lag of autocovariance. If nlags = 0, return the covariance matrix.
        :param symmetrize: (bool) If true, symmetrize the autocovariance matrix :math:`\\frac{A^T + A}{2}`;
            otherwise, return the original autocovariance matrix.
        :param use_standardized: (bool) If True, use standardized data; otherwise, use demeaned data.
        :return: (np.array) Autocovariance or covariance matrix.
        """

        data = self.standardized if use_standardized else self.demeaned
        # Lag the data, match the shape, and convert dataframe into numpy array
        if nlags > 0:
            data_now = data.iloc[nlags:].values
            data_lag = data.iloc[:-nlags].values

            # Autocovariance matrix of lag n
            autocov_m = (data_lag.T @ data_now) / (data_lag.shape[0] - 1)

            # Symmetrize the autocovariance, i.e. (A + A^T) / 2
            if symmetrize:
                return (autocov_m + autocov_m.T) / 2
            else:
                return autocov_m

        # nlags = 0, return covariance matrix (which is always symmetric)
        return (data.values.T @ data.values) / (data.shape[0] - 1)

    def least_square_VAR_fit(self, use_standardized=False) -> np.array:
        """
        Calculate the least square estimate of the VAR(1) matrix.

        :param use_standardized: (bool) If true, use standardized data; otherwise, use demeaned data.
        :return: (np.array) Least square estimate of VAR(1) matrix.
        """

        # Fit VAR(1) model
        if use_standardized:
            var_model = sm.tsa.VAR(self.standardized)
        else:
            var_model = sm.tsa.VAR(self.demeaned)

        # The statsmodels package will give the least square estimate
        least_sq_est = np.squeeze(var_model.fit(1).coefs, axis=0)

        return least_sq_est

    def box_tiao(self, threshold: int = 7) -> np.array:
        """
        Perform Box-Tiao canonical decomposition on the assets dataframe.

        :param threshold: (int) Round precision cutoff threshold. For example, a threshold of n means that a number less
            than :math:`10^{-n}` will be rounded to zero.
        :return: (np.array) The weighting of each asset in the portfolio. There will be N decompositions for N assets,
            where each column vector corresponds to one portfolio. The order of the weightings correspond to the
            descending order of the eigenvalue.
        """

        # Calculate the least square estimate of the price with VAR(1) model
        least_sq_est = self.least_square_VAR_fit(use_standardized=False)

        # Construct the matrix from which the eigenvectors need to be computed
        covar = self.demeaned.cov()
        box_tiao_matrix = np.linalg.inv(covar) @ least_sq_est @ covar @ least_sq_est.T

        # Calculate the eigenvectors and sort by eigenvalue
        eigvals, eigvecs = np.linalg.eig(box_tiao_matrix)

        # Sort the eigenvectors by eigenvalues by descending order
        bt_eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]

        # Return the weights
        return np.around(bt_eigvecs, threshold)

    @staticmethod
    def greedy_search(cardinality: int, var_est: np.array, cov_est: np.array, threshold: int = 7,
                      maximize: bool = False) -> np.array:
        """
        Greedy search algorithm for sparse decomposition.

        :param cardinality: (int) Number of assets to include in the portfolio.
        :param var_est: (np.array) Estimated VAR(1) coefficient matrix.
        :param cov_est: (np.array) Estimated covariance matrix.
        :param threshold: (int) Round precision cutoff threshold. For example, a threshold of n means that a number less
            than :math:`10^{-n}` will be treated as zero.
        :param maximize: (bool) If true, maximize predictability; otherwise, minimize predictability.
        :return: (np.array) Weight of each selected assets.
        """

        matrix_A = var_est.T @ cov_est @ var_est
        matrix_B = cov_est

        # Use a list to store all selected assets
        selected = []
        selected_weights = None

        # Use a set to record which assets have already been selected
        candidates = set(range(matrix_B.shape[0]))

        # Start greedy search
        for _ in range(cardinality):
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
                _, eigvec = scipy.linalg.eigh(cur_matrix_A, cur_matrix_B)

                # Construct the weighting of the portfolio based on the current support
                weight = np.zeros((matrix_B.shape[0], 1))

                # Choose the last eigenvector due to scipy.linalg.eigh sorts eigenvectors by ascending eigenvalues
                # Need to normalized the eigenvector to norm one
                weight[cand, :] = normalize(eigvec[:, -1].reshape(-1, 1), axis=0, norm='l2')

                # Store the maximum and the support corresponds to the maximum
                gen_eig_ratio = np.squeeze((weight.T @ matrix_A @ weight) / (weight.T @ matrix_B @ weight))

                # Minimize predictability is equivalent to maximize the inverse of predictability
                if not maximize:
                    gen_eig_ratio = 1. / gen_eig_ratio

                # Greedy algorithm to maximize the objective function
                if gen_eig_ratio > max_gen_eig_ratio:
                    max_gen_eig_ratio = gen_eig_ratio
                    cur_support = support
                    selected_weights = weight

            # Now the best asset candidate have been included in the portfolio, remove it from the candidate list
            selected.append(cur_support)
            candidates.remove(cur_support)

        return np.around(selected_weights, threshold)

    def sdp_eigenvalue(self, cardinality: float, var_est: np.array, cov_est: np.array,
                       verbose: bool = True, max_iter: int = 10000, maximize: bool = False) -> np.array:
        r"""
        Semidefinite relaxation sparse decomposition following the formulation in d'Aspremont (2011).

        .. math::
            :nowrap:

            \begin{align*}
            \text{minimize } & \mathbf{Tr}(AY) \\
            \text{subject to } & \mathbf{1}^T \lvert Y \rvert \mathbf{1} \leq k \, \mathbf{Tr}(Y) \\
            & \mathbf{Tr}(Y) > 0 \\
            & \mathbf{Tr}(BY) = 1 \\
            & Y \succeq 0
            \end{align*}

        :param cardinality: (float) Cardinality constraint. A float value is allowed for fine-tuning.
        :param var_est: (np.array) Estimated VAR(1) coefficient matrix.
        :param cov_est: (np.array) Estimated covariance matrix.
        :param verbose: (bool) If True, print the SDP solver iteration details for debugging; otherwise, suppress the
            debug output.
        :param max_iter: (int) Set number of iterations for the SDP solver.
        :param maximize: (bool) If true, maximize predictability; otherwise, minimize predictability.
        :return: (np.array) The optimized matrix :math:`Y`.
        """

        matrix_A = var_est.T @ cov_est @ var_est
        matrix_B = cov_est

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
        if maximize:
            problem = cp.Maximize(cp.trace(matrix_A @ Y))
        else:
            problem = cp.Minimize(cp.trace(matrix_A @ Y))

        cp.Problem(problem, constraints).solve(verbose=verbose, solver='SCS', max_iters=max_iter)

        return Y.value

    def sdp_predictability_vol(self, rho: float, variance: float, use_standardized: bool = True,
                               verbose: bool = True, max_iter: int = 10000) -> np.array:
        r"""
        Semidefinite relaxation optimization of predictability with a volatility threshold following the formulation of
        Cuturi (2016).

        .. math::
            :nowrap:

            \begin{align*}
            \text{minimize } & \mathbf{Tr}(\gamma_1 \gamma_0^{-1} \gamma_1^T Y) + \rho \lVert Y \rVert_1 \\
            \text{subject to } & \mathbf{Tr}(\gamma_0 Y) >= V \\
            & \mathbf{Tr}(Y) = 1 \\
            & Y \succeq 0
            \end{align*}

        where :math:`\gamma_i` is the lag-:math:`k` sample autocovariance (when :math:`k=0`, it is the sample
        covariance). :math:`V` is the variance lower bound of the portfolio.

        :param rho: (float) Regularization parameter of the :math:`l_1`-norm in the objective function.
        :param variance: (float) Variance lower bound for the portfolio.
        :param verbose: (bool) If True, print the SDP solver iteration details for debugging; otherwise, suppress the
            debug output.
        :param use_standardized: (bool) If true, use standardized data for optimization; otherwise, use de-meaned data.
        :param max_iter: (int) Set number of iterations for the SDP solver.
        :return: (np.array) The optimized matrix :math:`Y`
        """
        # Construct the matrix M
        acov1 = self.autocov(1, symmetrize=True, use_standardized=use_standardized)
        cov = self.autocov(0, use_standardized=use_standardized)
        matrix_M = acov1 @ np.linalg.inv(cov) @ acov1.T

        # Set up the SDP variable
        Y_dim = matrix_M.shape[0]
        Y = cp.Variable((Y_dim, Y_dim), symmetric=True)

        # Constraints
        constraints = [
            cp.trace(cov @ Y) >= variance,
            cp.trace(Y) == 1,
            Y >> 0
        ]

        # Minimization objective
        problem = cp.trace(matrix_M @ Y + rho * cp.sum(cp.abs(Y)))

        # Solve the SDP
        cp.Problem(cp.Minimize(problem), constraints).solve(verbose=verbose, solver='SCS', max_iters=max_iter)

        return Y.value

    def sdp_portmanteau_vol(self, rho: float, variance: float, nlags: int = 3, use_standardized: bool = True,
                            verbose: bool = True, max_iter: int = 10000) -> np.array:
        r"""
        Semidefinite relaxation optimization of portmanteau statistic with a volatility threshold following the
        formulation of Cuturi (2016).

        .. math::
            :nowrap:

            \begin{align*}
            \text{minimize } & \sum_{i=1}^p \mathbf{Tr}(\gamma_i Y)^2 + \rho \lVert Y \rVert_1 \\
            \text{subject to } & \mathbf{Tr}(\gamma_0 Y) >= V \\
            & \mathbf{Tr}(Y) = 1 \\
            & Y \succeq 0
            \end{align*}

        where :math:`\gamma_i` is the lag-:math:`k` sample autocovariance (when :math:`k=0`, it is the sample
        covariance). :math:`V` is the variance lower bound of the portfolio.

        :param rho: (float) Regularization parameter of the :math:`l_1`-norm in the objective function.
        :param variance: (float) Variance lower bound for the portfolio.
        :param nlags: (int) Order of portmanteau statistic :math:`p`.
        :param verbose: (bool) If True, print the SDP solver iteration details for debugging; otherwise, suppress the
            debug output.
        :param use_standardized: (bool) If true, use standardized data for optimization; otherwise, use de-meaned data.
        :param max_iter: (int) Set number of iterations for the SDP solver.
        :return: (np.array) The optimized matrix :math:`Y`.
        """

        # Calculate the covariance and the autocovariance matrices
        cov = self.autocov(0, use_standardized=use_standardized)
        acovs = [self.autocov(i, use_standardized=use_standardized) for i in range(1, nlags + 1)]

        # Formulate the SDP problem
        Y_dim = cov.shape[0]
        Y = cp.Variable((Y_dim, Y_dim), symmetric=True)
        problem = rho * cp.sum(cp.abs(Y))
        for acov in acovs:
            problem += cp.square(cp.trace(acov @ Y))

        # Formulate the constraints
        constraints = [
            cp.trace(cov @ Y) >= variance,
            cp.trace(Y) == 1,
            Y >> 0
        ]

        # Solve the SDP
        cp.Problem(cp.Minimize(problem), constraints).solve(verbose=verbose, solver='SCS', max_iters=max_iter)

        return Y.value

    def sdp_crossing_vol(self, rho: float, mu: float, variance: float, nlags: int = 3, use_standardized: bool = True,
                         verbose: bool = True, max_iter: int = 10000) -> np.array:
        r"""
        Semidefinite relaxation optimization of crossing statistic with a volatility threshold following the
        formulation of Cuturi (2016).

        .. math::
            :nowrap:

            \begin{align*}
            \text{minimize } & \mathbf{Tr}(\gamma_1 Y) + \mu \sum_{i=2}^p \mathbf{Tr}(\gamma_i Y)^2 + \rho \lVert Y \rVert_1 \\
            \text{subject to } & \mathbf{Tr}(\gamma_0 Y) >= V \\
            & \mathbf{Tr}(Y) = 1 \\
            & Y \succeq 0
            \end{align*}

        where :math:`\gamma_i` is the lag-:math:`k` sample autocovariance (when :math:`k=0`, it is the sample
        covariance). :math:`V` is the variance lower bound of the portfolio.

        :param rho: (float) Regularization parameter of the :math:`l_1`-norm in the objective function.
        :param mu: (float) Regularization parameter of higher-order autocovariance.
        :param variance: (float) Variance lower bound for the portfolio.
        :param nlags: (int) Order of portmanteau statistic :math:`p`.
        :param verbose: (bool) If True, print the SDP solver iteration details for debugging; otherwise, suppress the
            debug output.
        :param use_standardized: (bool) If true, use standardized data for optimization; otherwise, use de-meaned data.
        :param max_iter: (int) Set number of iterations for the SDP solver.
        :return: (np.array) The optimized matrix :math:`Y`.
        """

        # Calculate the covariance and the autocovariance matrices
        cov = self.autocov(0, use_standardized=use_standardized)
        acovs = [self.autocov(i, use_standardized=use_standardized) for i in range(1, nlags + 1)]

        # Formulate the SDP problem
        Y_dim = cov.shape[0]
        Y = cp.Variable((Y_dim, Y_dim), symmetric=True)
        problem = rho * cp.sum(cp.abs(Y)) + mu * cp.trace(acovs[0] @ Y)
        for acov in acovs[1:]:
            problem += cp.square(cp.trace(acov @ Y))

        # Formulate the constraints
        constraints = [
            cp.trace(cov @ Y) >= variance,
            cp.trace(Y) == 1,
            Y >> 0
        ]

        # Solve the SDP
        cp.Problem(cp.Minimize(problem), constraints).solve(verbose=verbose, solver='SCS', max_iters=max_iter)

        return Y.value

    def LASSO_VAR_tuning(self, sparsity: float, multi_task_lasso: bool = False, alpha_min: float = -5.,
                         alpha_max: float = 0., n_alphas: int = 100, max_iter: int = 1000,
                         use_standardized: bool = True) -> float:
        """
        Tune the l1-regularization coefficient (alpha) of LASSO regression for a sparse estimate of the VAR(1) matrix.

        :param sparsity: (float) Percentage of zeros required in the VAR(1) matrix.
        :param multi_task_lasso: (bool) If True, use multi-task LASSO for sparse estimate, where the LASSO will yield
            full columns of zeros; otherwise, do LASSO column-wise.
        :param alpha_min: (float) Minimum l1-regularization coefficient.
        :param alpha_max: (float) Maximum l1-regularization coefficient.
        :param n_alphas: (int) Number of l1-regularization coefficient for the parameter search.
        :param max_iter: (int) Maximum number of iterations for LASSO regression.
        :param use_standardized: (bool) If true, use standardized data for optimization; otherwise, use de-meaned data.
        :return: (float) Minimum alpha that satisfies the sparsity requirement.
        """

        if use_standardized:
            data = self.standardized
        else:
            data = self.demeaned

        # The number of elements in the VAR(1) matrix is asset number squared
        coefs_nums = data.shape[1] ** 2

        # Construct the current data and lag-1 data such that they have the same shape
        data_now = data.iloc[1:]
        data_lag = data.iloc[:-1]

        # Set up the parameter space for alpha
        alphas = np.linspace(alpha_min, alpha_max, n_alphas)

        # Set up the LASSO model and do a search on the alpha parameter space
        if multi_task_lasso:
            # Fit the multi-task LASSO model
            _, coefs_lasso, _ = lasso_path(data_lag, data_now, alphas=alphas, max_iter=max_iter)
            # Select the maximum alpha that satisfies the sparsity requirement
            non_zeros = np.count_nonzero(coefs_lasso, axis=(0, 1))

            # Find the index of the best alpha
            best_alpha_index = np.searchsorted(non_zeros, (1 - sparsity) * coefs_nums)

            # Retrieve the best alpha
            if best_alpha_index in [0, len(non_zeros)]:
                best_alpha = np.Inf
            else:
                best_alpha = alphas[::-1][best_alpha_index]

        else:
            best_alpha = np.Inf
            # Fit the normal LASSO model
            for alpha in alphas:
                lasso_model = Lasso(alpha=alpha, max_iter=max_iter)

                # Get the coefficient column-wise
                coefs_lasso = np.array([lasso_model.fit(data_lag, y).coef_ for y in data_now.values.T])

                # Calculate the number of non-zero elements
                non_zeros = np.count_nonzero(coefs_lasso)

                # Check if the number of non-zeros satisfies the requirements
                if non_zeros <= (1 - sparsity) * coefs_nums:
                    best_alpha = np.min([best_alpha, alpha])
                    break

        if np.isinf(best_alpha):
            raise ValueError("The l1-regularization coefficient (alpha) range selected cannot meet the "
                             "sparsity requirements. Please try another alpha range for a sparser estimate.")
        return best_alpha

    def LASSO_VAR_fit(self, alpha: float, multi_task_lasso: bool = True, max_iter: int = 1000,
                      threshold: int = 10, use_standardized: bool = True) -> np.array:
        """
        Fit the LASSO model with the designated alpha for a sparse VAR(1) coefficient matrix estimate.

        :param alpha: (float) Optimized l1-regularization coefficient.
        :param multi_task_lasso: (bool) If True, use multi-task LASSO for sparse estimate, where the LASSO will yield
            full columns of zeros; otherwise, do LASSO column-wise.
        :param max_iter: (int) Maximum number of iterations of LASSO regression.
        :param threshold: (int) Round precision cutoff threshold. For example, a threshold of n means that a number less
            than :math:`10^{-n}` will be treated as zero.
        :param use_standardized: (bool) If true, use standardized data for optimization; otherwise, use de-meaned data.
        :return: (np.array) Sparse estimate of VAR(1) matrix.
        """

        # Construct the current data and lag-1 data such that they have the same shape
        if use_standardized:
            data_now = self.standardized.iloc[1:]
            data_lag = self.standardized.iloc[:-1]
        else:
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

    def covar_sparse_tuning(self, max_iter: int = 1000, alpha_min: float = 0., alpha_max: float = 1.,
                            n_alphas: int = 100, clusters: int = 3, use_standardized: bool = True) -> float:
        """
        Tune the regularization parameter (alpha) of the graphical LASSO model for a sparse estimate of the covariance
        matrix.

        :param max_iter: (int) Maximum number of iterations for graphical LASSO fit.
        :param alpha_min: (float) Minimum regularization parameter.
        :param alpha_max: (float) Maximum regularization parameter.
        :param n_alphas: (int) Number of regularization parameter for parameter search.
        :param clusters: (int) Number of smaller clusters desired from the precision matrix.
            The higher the number, the larger the best alpha will be. This parameter cannot exceed the number of assets.
        :param use_standardized: (bool) If true, use standardized data for optimization; otherwise, use de-meaned data.
        :return: (float) Optimal alpha to split the graph representation of the inverse covariance matrix into
            designated number of clusters.
        """
        # Check parameter validity
        if clusters > self.assets.shape[1]:
            raise ValueError("The number of clusters cannot exceed the number of assets.")

        # Set up the parameter space for the regularization parameter
        alphas = np.linspace(alpha_min, alpha_max, n_alphas)

        # Fit the graphical LASSO model
        for alpha in alphas:
            edge_model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
            if use_standardized:
                edge_model.fit(self.standardized)
            else:
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
        raise ValueError("The regularization coefficient range selected cannot meet the number of connected components "
                         "requirements. Please try larger alphas for a sparser estimate.")

    def covar_sparse_fit(self, alpha: float, max_iter: int = 1000, threshold: int = 10,
                         use_standardized: bool = True) -> Tuple[np.array, np.array]:
        """
        Fit the graphical LASSO model using the optimized alpha for a sparse covariance matrix estimate.

        :param alpha: (float) Optimized regularization coefficient of graphical LASSO.
        :param max_iter: (int) Maximum number of iterations for graphical LASSO fit.
        :param threshold: (int) Round precision cutoff threshold. For example, a threshold of n means that a number less
            than :math:`10^{-n}` will be treated as zero.
        :param use_standardized: (bool) If true, use standardized data for optimization; otherwise, use de-meaned data.
        :return: (np.array, np.array) Sparse estimate of covariance matrix; inverse of the sparse covariance matrix,
            i.e. precision matrix as graph representation.
        """

        # Fit graphical LASSO model
        edge_model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
        if use_standardized:
            edge_model.fit(self.standardized)
        else:
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

    def sparse_eigen_deflate(self, sdp_result: np.array, cardinality: int, tol: float = 1e-6, max_iter: int = 100,
                             verbose: bool = True) -> np.array:
        """
        Calculate the leading **sparse** eigenvector of the SDP result. Deflate the original
        leading eigenvector to the input cardinality using Truncated Power method (Yuan and Zhang, 2013).

        The Truncated Power method is ported from the Matlab code provided by the original authors.

        :param sdp_result: (np.array) The optimization result from semidefinite relaxation.
        :param cardinality: (int) Desired cardinality of the sparse eigenvector.
        :param tol: (float) Convergence tolerance of the Truncated Power method.
        :param max_iter: (int) Maximum number of iterations for Truncated Power method.
        :param verbose: (bool) If True, print the Truncated Power method iteration details; otherwise, suppress the
            debug output.
        :return: (np.array) Leading sparse eigenvector of the SDP result.
        """

        # Abnormal input handling
        if cardinality > sdp_result.shape[0]:
            raise ValueError("Desired cardinality cannot exceed the number of assets!")

        if cardinality < 1:
            raise ValueError("Desired cardinality must be positive integers!")

        if self.is_semi_pos_def(sdp_result):
            warnings.warn("The SDP result is not positive semidefinite due to numerical issues. Please double check"
                          "if the negative eigenvalues are sufficiently small that they can be approximated as 0.")

        # Implement the TPower method by Yuan and Zhang (2013)
        # Initialize the leading sparse vector
        non_zero_index = np.argsort(np.diag(sdp_result))
        initial = np.zeros((sdp_result.shape[0],))

        # Assign 1 to the corresponding indices of the largest k diagonal elements; normalize to norm one
        initial[non_zero_index[-cardinality:]] = 1
        initial /= np.linalg.norm(initial)

        iteration = 1

        # Power step
        x = initial.reshape(-1, 1)
        s = np.squeeze(sdp_result @ x)
        g = 2 * s
        f = np.squeeze(x.T @ s)

        # Truncate step
        x = self.truncate(g, cardinality)
        f_old = f

        # TPower iteration loop
        while iteration <= max_iter:
            if verbose:
                print("Iteration: {}, Objective function value: {}".format(iteration, f_old))
            # Power step
            s = np.squeeze(sdp_result @ x)
            g = 2 * s

            # Truncate step
            x = self.truncate(g, cardinality)
            f = np.squeeze(x.T @ s)

            if np.abs(f - f_old) < tol:
                break

            # Keep record of the target function value
            f_old = f
            iteration += 1

        return x

    @staticmethod
    def truncate(vector: np.array, cardinality: int) -> np.array:
        """
        Helper function of Truncated Power method.

        :param vector: (np.array) The placeholder vector for the sparse eigenvector.
        :param cardinality: (int) The desired cardinality.
        :return: (np.array) Truncated placeholder vector.
        """

        # Sort the vector elements by their absolute value
        u = np.zeros((vector.shape[0], ))
        non_zero_index = np.argsort(np.abs(vector))

        # Select the largest k elements and discard the remainder; normalize to norm one
        v_truncated = vector[non_zero_index[-cardinality:]]
        u[non_zero_index[-cardinality:]] = v_truncated / np.linalg.norm(v_truncated)

        return u.reshape(-1, 1)

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
    def is_semi_pos_def(matrix: np.array) -> bool:
        """
        Check if a matrix is positive definite.

        :param matrix: (np.array) The matrix under inspection.
        :return: (bool) True if the matrix is positive definite, False otherwise.
        """
        return np.all(np.linalg.eigvals(matrix + matrix.T) >= 0)