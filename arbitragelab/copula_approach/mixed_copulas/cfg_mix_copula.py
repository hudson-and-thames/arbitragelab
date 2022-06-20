# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module that uses copula for trading strategy based method described in the following article.

`B Sabino da Silva, F., Ziegelman, F. and Caldeira, J., 2017. Mixed Copula Pairs Trading Strategy on the S&P 500.
Flávio and Caldeira, João, Mixed Copula Pairs Trading Strategy on the S&P, 500.
<https://www.researchgate.net/profile/Fernando_Sabino_Da_Silva/publication/315878098_Mixed_Copula_Pairs_Trading_Strategy_on_the_SP_500/links/5c6f080b92851c695036785f/Mixed-Copula-Pairs-Trading-Strategy-on-the-S-P-500.pdf>`__

Note: Algorithm for fitting mixed copula was adapted from

`Cai, Z. and Wang, X., 2014. Selection of mixed copula model via penalized likelihood. Journal of the American
Statistical Association, 109(506), pp.788-801.
<https://www.tandfonline.com/doi/pdf/10.1080/01621459.2013.873366?casa_token=sey8HrojSgYAAAAA:TEMBX8wLYdGFGyM78UXSYm6hXl1Qp_K6wiLgRJf6kPcqW4dYT8z3oA3I_odrAL48DNr3OSoqkQsEmQ>`__
"""

# pylint: disable = invalid-name, too-many-locals, arguments-differ
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import arbitragelab.copula_approach.copula_calculation as ccalc
from arbitragelab.copula_approach.archimedean import Gumbel, Clayton, Frank
from arbitragelab.copula_approach.mixed_copulas.base import MixedCopula
from arbitragelab.util import segment


class CFGMixCop(MixedCopula):
    """
    Clayton, Frank and Gumbel mixed copula.
    """

    def __init__(self, cop_params: list = None, weights: list = None):
        """
        Initiate Clayton, Frank and Gumbel (CFG) mixed copula.

        :param cop_params: (list) (3, ) size. Copula parameters for Clayton, Frank and Gumbel respectively.
        :param weights: (list) (3, ) size. Copulas weights for Clayton, Frank and Gumbel respectively. Need to be
            positive and sum up to 1.
        """
        super().__init__('CFGMixCop')
        self.cop_params = cop_params
        self.weights = weights
        self.clayton_cop, self.frank_cop, self.gumbel_cop = None, None, None
        # Initiate component copulas if they are given cop_params and weights
        if cop_params is not None:
            self.clayton_cop = Clayton(theta=self.cop_params[0])
            self.frank_cop = Frank(theta=self.cop_params[1])
            self.gumbel_cop = Gumbel(theta=self.cop_params[2])

        self.copulas = [self.clayton_cop, self.frank_cop, self.gumbel_cop]

        segment.track('CFGMixCop')

    def fit(self, data: pd.DataFrame, max_iter: int = 25, gamma_scad: float = 0.6, a_scad: float = 6,
            weight_margin: float = 1e-2) -> float:
        """
        Fitting cop_params and weights by expectation maximization (EM) from real data.

        Changes the mix copulas weights and copula parameters internally. Also returns the sum of log likelihood. The
        data will be converted to quantile by empirical cumulative distribution function.

        Implementation of EM method based on a non-parametric adaptation of the article:
        `Cai, Z. and Wang, X., 2014. Selection of mixed copula model via penalized likelihood. Journal of the American
        Statistical Association, 109(506), pp.788-801.
        <https://www.tandfonline.com/doi/pdf/10.1080/01621459.2013.873366?casa_token=sey8HrojSgYAAAAA:TEMBX8wLYdGFGyM78UXSYm6hXl1Qp_K6wiLgRJf6kPcqW4dYT8z3oA3I_odrAL48DNr3OSoqkQsEmQ>`__

        It contains the following procedure:

        1. Expectation step computes and updates the weights conditional on the copula parameters, using an iterative
           method.
        2. Maximization step maximizes an adapted log-likelihood function Q with penalty terms given the weights, using
           a Truncated Newton method, by minimizing Q over cop_params.

        Note: For the tuning parameters gamma_scad and a_scad, the final result is relatively sensitive based on their
        value. The default values were tested on limited data sets using stocks price series and returns series.
        However, the user is expected to tune them when necessary. Another approach is to estimate them using cross
        validation by the user. It is always a good practice to plot the sampling with the actual data for a sanity
        check.

        :param data: (pd.DataFrame) Data in (n, 2) pd.DataFrame used to fit the mixed copula.
        :param max_iter: (int) Optional. Maximum iteration for the EM method. The class default value 25 is just an
            empirical estimation and the user is expected to change it when needed.
        :param gamma_scad: (float) Optional. Tuning parameter for the SCAD penalty term. Defaults to 0.6.
        :param a_scad: (float) Optional. Tuning parameter for the SCAD penalty term. Defaults to 6.
        :param weight_margin: (float) Optional. A small number such that if below this threshold, the weight will be
            considered 0. Defaults to 1e-2.
        :return: (float) Sum of log likelihood for the fit.
        """

        # Make a quantile_data DataFrame by mapping original data with marginal cdfs.
        quantile_data = data.multiply(0)
        cdf1 = ccalc.construct_ecdf_lin(data.iloc[:, 0])
        cdf2 = ccalc.construct_ecdf_lin(data.iloc[:, 1])
        quantile_data.iloc[:, 0] = data.iloc[:, 0].map(cdf1)
        quantile_data.iloc[:, 1] = data.iloc[:, 1].map(cdf2)
        # Fit the quantile data.
        weights, cop_params = self._fit_quantile_em(quantile_data, max_iter, gamma_scad, a_scad)
        # Post processing weights. Abandon weights that are too small.
        weights = ccalc.adjust_weights(weights, threshold=weight_margin)
        # Internally construct the parameters and weights from the fit result.
        self.weights = weights
        self.cop_params = cop_params
        # Update the copulas with the updated params.
        self.clayton_cop = Clayton(theta=self.cop_params[0])
        self.frank_cop = Frank(theta=self.cop_params[1])
        self.gumbel_cop = Gumbel(theta=self.cop_params[2])
        # List used for the MixedCopula superclass
        self.copulas = [self.clayton_cop, self.frank_cop, self.gumbel_cop]

        # Calculate sum of log likelihood as the result of fit
        u1 = quantile_data.iloc[:, 0].to_numpy()
        u2 = quantile_data.iloc[:, 1].to_numpy()
        sum_log_likelihood = self._ml_qfunc(u1, u2, cop_params, weights, gamma_scad, a_scad, if_penalty=False)

        return sum_log_likelihood

    def _fit_quantile_em(self, quantile_data: pd.DataFrame, max_iter: int, gamma_scad: float,
                         a_scad: float) -> (np.array, np.array):
        """
        Fitting cop_params and weights by expectation maximization (EM) from quantile-data.

        Implementation of EM method based on a non-parametric adaptation of the article:
        `Cai, Z. and Wang, X., 2014. Selection of mixed copula model via penalized likelihood. Journal of the American
        Statistical Association, 109(506), pp.788-801.
        <https://www.tandfonline.com/doi/pdf/10.1080/01621459.2013.873366?casa_token=sey8HrojSgYAAAAA:TEMBX8wLYdGFGyM78UXSYm6hXl1Qp_K6wiLgRJf6kPcqW4dYT8z3oA3I_odrAL48DNr3OSoqkQsEmQ>`__

        It contains the following procedure:

        1. Expectation step computes and updates the weights conditional on the copula parameters, using an iterative
           method.
        2. Maximization step maximizes an adapted log-likelihood function Q with penalty terms given the weights, using
           a Truncated Newton method, by minimizing Q over cop_params.

        Note: For the tuning parameters gamma_scad and a_scad, the final result is relatively sensitive based on their
        value. The default values were tested on limited data sets using stocks price series and returns series.
        However, the user is expected to tune them when necessary. Another approach is to estimate them using cross
        validation by the user. It is always a good practice to plot the sampling with the actual data for a sanity
        check.

        :param quantile_data: (pd.DataFrame) The quantile data to be used for fitting.
        :param max_iter: (int) Optional. Maximum iteration for the EM method. The class default value 25 is just an
            empirical estimation and the user is expected to change it when needed.
        :param gamma_scad: (float) Tuning parameter for the SCAD penalty term.
        :param a_scad: (float) Tuning parameter for the SCAD penalty term.
        :return: (tuple) The fitted weights in (3, ) np.array and the fitted cop_params in (3, ) np.array.
        """

        # Initial guesses of weights and copula parameters
        init_weights = [0.33, 0.33, 1 - 0.33 - 0.33]
        init_cop_params = [3, 4, 5]
        # Initial round of calculation using guesses.
        weights = self._expectation_step(quantile_data, gamma_scad, a_scad, init_cop_params, init_weights)
        cop_params = self._maximization_step(quantile_data, gamma_scad, a_scad, init_cop_params, weights)
        # Full parameters, including weights and cop_params we aim to optimize.
        old_full_params = np.concatenate([init_weights, init_cop_params], axis=None)
        new_full_params = np.concatenate([weights, cop_params], axis=None)

        # Initiate while loop conditions.
        l1_diff = np.linalg.norm(old_full_params - new_full_params, ord=1)
        i = 1
        # Terminate when reached max iteration or l1 difference is small enough
        while i < max_iter and l1_diff > 5 * 1e-2:
            # Update for the old parameters
            old_full_params = np.concatenate([weights, cop_params], axis=None)
            # 1. Expectation step
            weights = self._expectation_step(quantile_data, gamma_scad, a_scad, cop_params, weights)
            # 2. Maximization step
            cop_params = self._maximization_step(quantile_data, gamma_scad, a_scad, cop_params, weights)
            # Update for the new parameters
            new_full_params = np.concatenate([weights, cop_params], axis=None)
            # Update the l1 difference norm and also the counter
            l1_diff = np.linalg.norm(old_full_params - new_full_params, ord=1)
            i += 1

        return weights, cop_params

    @staticmethod
    def _expectation_step(quantile_data: pd.DataFrame, gamma_scad: float, a_scad: float,
                          cop_params: list, weights: list) -> np.array:
        """
        The expectation step for EM approach on fitting mixed copula.

        This step updates the weights iteratively given cop_params, to optimize the conditional (on cop_params)
        expectation. Due to the SCAD penalty term, it tends to drive small weight(s) to 0. The algorithm is adapted from
        (Cai et al. 2014) as a non-parametric marginal version.

        :param quantile_data: (pd.DataFrame) The quantile data to be used for fitting.
        :param gamma_scad: (float) Tuning parameter for the SCAD penalty term.
        :param a_scad: (float) Tuning parameter for the SCAD penalty term.
        :param cop_params: (list) Shape (3, ), copula parameters for dependency. This is its initial guess.
        :param weights: (list) Shape (3, ), copula weights for the mix copula.
        :return: (np.array) Shape (3, ), the updated weights in np.array form.
        """

        num = len(quantile_data)  # Number of data Points.

        # Lower level density calculations were implemented using numpy arrays.
        u1 = quantile_data.iloc[:, 0].to_numpy()
        u2 = quantile_data.iloc[:, 1].to_numpy()

        # Iterative steps to find weights.
        # Initiate parameters for iterative computation.
        diff = 1  # Difference of the updated weight compared to the previous step.
        tol_weight = 1e-2  # When weight difference <= tol_weight, end the loop.
        iteration = 0
        # Initiate copulas with given cop_params. (cop_params does not change throughout the method)
        local_copulas = [Clayton(theta=cop_params[0]),
                         Frank(theta=cop_params[1]),
                         Gumbel(theta=cop_params[2])]

        # When difference <= tolerance OR iteration > 10, break the loop.
        while diff > tol_weight and iteration < 10:  # Small iterations to avoid overfitting
            new_weights = np.array([np.nan] * 3)
            iteration += 1
            for i in range(3):  # For each component of the mixed copula.
                # The rest of the for loop are just calculating components of the formula in (Cai et al. 2014)
                sum_ml_lst = np.zeros_like(u1)
                for t in range(num):  # For each data point.
                    sum_ml_lst[t] = (weights[i] * local_copulas[i].get_cop_density(u=u1[t], v=u2[t]) /
                                     np.sum([weights[j] * local_copulas[j].get_cop_density(u=u1[t], v=u2[t])
                                             for j in range(3)]))

                sum_ml = np.sum(sum_ml_lst)
                numerator = weights[i] * ccalc.scad_derivative(weights[i], gamma_scad, a_scad) - sum_ml / num
                denominator = np.sum([weight * ccalc.scad_derivative(weight, gamma_scad, a_scad)
                                      for weight in weights]) - 1
                new_weights[i] = numerator / denominator

            # Difference is defined in l1 norm.
            diff = np.sum(np.abs(weights - new_weights))
            weights = np.copy(new_weights)  # Only take values.

        return weights

    def _maximization_step(self, quantile_data: pd.DataFrame, gamma_scad: float, a_scad: float, cop_params: list,
                           weights: list) -> np.array:
        """
        The maximization step for EM approach on fitting mixed copula.

        This step uses a given weight, and updates the cop_params such that it maximizes Q. The authors (Cai et al.
        2014) used an iterative Newton-Raphson approach on |dQ(cop_params)/d cop_params| = 0 to find cop_params. However
        it is not guaranteed that such root exits. Hence we simply use 'TNC' (Truncated Newton) on minimizing -Q for
        practicality.

        :param quantile_data: (pd.DataFrame) The quantile data to be used for fitting.
        :param gamma_scad: (float) Tuning parameter for the SCAD penalty term.
        :param a_scad: (float) Tuning parameter for the SCAD penalty term.
        :param cop_params: (list) Shape (3, ), copula parameters for dependency. This is its initial guess.
        :param weights: (list) Shape (3, ), copula weights for the mix copula.
        :return: (np.array) Shape (3, ), the updated copula parameters in np.array form.
        """

        # Lower level density calculations were implemented using numpy arrays.
        u1 = quantile_data.iloc[:, 0].to_numpy()
        u2 = quantile_data.iloc[:, 1].to_numpy()
        eps = 1e-3  # Minimization tolerance.
        cop_params = np.array(cop_params)  # Cast to numpy array.

        # Define objection function -Q.
        def q_func(my_cop_params):
            # Compute the numerical gradient w.r.t. cop_params
            result = self._ml_qfunc(u1, u2, my_cop_params, weights, gamma_scad, a_scad, multiplier=-1)

            return result  # - (max_likelihood - penalty)

        # Use TNC to find weights such that norm_grad is minimized (hopefully 0)
        init_cop_params = cop_params  # Initial guess on cop_params.
        # Bounds: theta_c in [-1, 100]. theta_f in [-50, 50]. theta_g in [1, 100]
        bnds = ((-1, 100), (-50, 50), (1, 100))

        res = minimize(q_func, x0=init_cop_params, method='TNC', bounds=bnds, options={'disp': False}, tol=eps)

        return res.x  # Return the updated copula parameters.

    @staticmethod
    def _ml_qfunc(u1: np.array, u2: np.array, cop_params: list, weights: list,
                  gamma_scad: float, a_scad: float, if_penalty: bool = True, multiplier: float = 1) -> float:
        """
        The object function to minimize for EM method. Usually denoted as Q in literature.

        It is log_likelihood - SCAD penalty. The SCAD penalty drives small copula weights to 0 for better modeling.
        However, the exact parameters may require tuning to get a good result.

        :param u1: (np.array) 1D vector data. Need to be uniformly distributed in [0, 1].
        :param u2: (np.array) 1D vector data. Need to be uniformly distributed in [0, 1].
        :param cop_params: (list) Shape (3, ), copula parameters for dependency.
        :param weights: (list) Shape (3, ), copula weights for the mix copula.
        :param gamma_scad: (float) Tuning parameter for the SCAD penalty term.
        :param a_scad: (float) Tuning parameter for the SCAD penalty term.
        :param if_penalty: (bool) Optional. If adding SCAD penalty term. Without the penalty term it is just sum of
            log likelihood. Defaults to True.
        :param multiplier: (float) Optional. Multiply the calculated result by a number. -1 is usually used when an
            optimization algorithm searches minimum instead of maximum. Defaults to 1.
        :return: (float) The value of the objective function.
        """

        num = len(u1)
        # Reassign copula parameters and weights for local variables for readability.
        theta_c, theta_f, theta_g = cop_params
        weight_c, weight_f, _ = weights

        # Create local copulas for density calculation.
        clayton_cop = Clayton(theta=theta_c)
        frank_cop = Frank(theta=theta_f)
        gumbel_cop = Gumbel(theta=theta_g)

        # Calculate mixed copula's log-likelihood over data.
        likelihood_list_clayton = np.array([clayton_cop.get_cop_density(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
        likelihood_list_frank = np.array([frank_cop.get_cop_density(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
        likelihood_list_gumbel = np.array([gumbel_cop.get_cop_density(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
        likelihood_list_mix = (weight_c * likelihood_list_clayton + weight_f * likelihood_list_frank
                               + (1 - weight_c - weight_f) * likelihood_list_gumbel)
        log_likelihood_sum = np.sum(np.log(likelihood_list_mix))

        # Calculate the penalty term.
        penalty = num * np.sum([ccalc.scad_penalty(weights[k], gamma=gamma_scad, a=a_scad) for k in range(3)])

        return (log_likelihood_sum - penalty * int(if_penalty)) * multiplier

    def _get_param(self) -> dict:
        """
        Get the name and parameter(s) for this mixed copula instance.

        :return: (dict) Name and parameters for this copula.
        """

        descriptive_name = 'Bivariate Clayton-Frank-Gumbel Mixed Copula'
        class_name = 'CFGMixCop'
        cop_params = self.cop_params
        weights = self.weights
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'Clayton theta': cop_params[0], 'Frank theta': cop_params[1], 'Gumbel theta': cop_params[2],
                     'Clayton weight': weights[0], 'Frank weight': weights[1], 'Gumbel weight': weights[2]}

        return info_dict
