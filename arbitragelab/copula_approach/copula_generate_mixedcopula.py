# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module that uses copula for trading strategy based method described in the following article.

B Sabino da Silva, F., Ziegelman, F. and Caldeira, J., 2017. Mixed Copula Pairs Trading Strategy on the S&P 500. 
Flávio and Caldeira, João, Mixed Copula Pairs Trading Strategy on the S&P, 500.
https://www.researchgate.net/profile/Fernando_Sabino_Da_Silva/publication/315878098_Mixed_Copula_Pairs_Trading_Strategy_on_the_SP_500/links/5c6f080b92851c695036785f/Mixed-Copula-Pairs-Trading-Strategy-on-the-S-P-500.pdf
"""
# pylint: disable = invalid-name, too-many-locals
from abc import ABC
from scipy.optimize import minimize, OptimizeResult
from typing import Callable
import numpy as np
import pandas as pd
from arbitragelab.copula_approach.copula_strategy_mpi import CopulaStrategyMPI
import arbitragelab.copula_approach.copula_generate as cg

# class CopulaStrategyMixCop(CopulaStrategyMPI):
#     """
#     Copula trading strategy based on [da Silva et al. (2016)], involving MPI and mixed copulas.
    
#     This strategy uses mispricing indices and forms flag series. However, as compared to the approach described in
#     [Xie et al. 2014, Pairs Trading with Copulas] and thus in class CopulaStrategyMPI, this strategy does NOT reset
#     the flag series once reached a exit trigger.
#     """
#     def __init__(self):
#         pass

class MixedCopula(ABC):
    """
    Class template for mixed copulas.
    """
    def __init__(self):
        """
        Initiate the MixedCopula class.
        """
        
    def describe(self) -> dict:
        """
        Describe the components and coefficients of the mixed copula.
        """
        description = pd.Series(self._get_param())

        return description

    def c(self, u: float, v: float) -> float:
        """
        Calculate probability density of the bivariate copula: P(U=u, V=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The probability density (aka copula density).
        """

        # linear combo w.r.t. weights for each copula in the mix
        pdf = np.sum([self.weights[i] * cop.c(u, v) for i, cop in enumerate(self.copulas)])

        return pdf
    
    def C(self, u: float, v: float) -> float:
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).

        Result is analytical.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The culumative density.
        """

        # linear combo w.r.t. weights for each copula in the mix
        cdf = np.sum([self.weights[i] * cop.C(u, v) for i, cop in enumerate(self.copulas)])
        
        return cdf

    def condi_cdf(self, u: float, v: float) -> float:
        """
        Calculate conditional probability function: P(U<=u | V=v).

        Result is analytical.

        Note: This probability is symmetric about (u, v).

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        :return: (float) The conditional probability.
        """

        # linear combo w.r.t. weights for each copula in the mix
        result = np.sum([self.weights[i] * cop.condi_cdf(u, v) for i, cop in enumerate(self.copulas)])

        return result

    @staticmethod
    def _away_from_0(x: float, lower_limit: float = -1e-5, upper_limit: float = 1e-5) -> float:
        """
        Keep the parameter x away from 0 but still retain the sign.
        
        0 is remapped to the upper_limit.

        :param x: (float) The numbe to be remapped.
        :param lower_limit: (float) The lower limit to be considered a close enough to 0.
        :param upper_limit: (float) The upper limit to be considered a close enough to 0.
        :return: (float) The remapped parameter.
        """
        small_pos_bool = (0 <= x < upper_limit)  # Whether it is a small positive number
        small_neg_bool = (lower_limit < x < 0)  # Whether it is a small negative number
        small_bool = small_pos_bool or small_neg_bool  # Whether it is a small number
        # If not small, then return the param
        # If small, then return the corresponding limit.
        remapped_param = x * int(not small_bool) \
            + upper_limit * int(small_pos_bool) + lower_limit * int(small_neg_bool)

        return remapped_param

class CTGMixCop(MixedCopula):
    """
    Clayton, Student-t and Gumbel mixed copula.
    """
    def __init__(self, cop_params: tuple = None, weights: tuple = None):
        """
        Initiate Clayton, Student-t and Gumbel (CFG) mixed copula.
        
        :param cop_params: (tuple) (4, ) size. Copula parameters for Clayton, Student-t and Gumbel respectively. 
            Format is cop_params = (theta_clayton, rho_t, nu_t, theta_gumbel)
        :param weights: (tuple) (3, ) size. Copulas weights for Clayton, Frank and Gumbel respectively. Needs to be positive and
            sum up to 1.
        """
        super().__init__()
        self.cop_params = cop_params
        self.weights = weights
        self.clayton_cop, self.t_cop, self.gumbel_cop = None, None, None
        # Initiate component copulas if they are given cop_params and weights
        if cop_params is not None:
            self.clayton_cop = cg.Clayton(theta=self.cop_params[0])
            corr = [[1, cop_params[1]], [cop_params[1], 1]]
            self.t_cop = cg.Student(cov=corr, nu=cop_params[2])
            self.gumbel_cop = cg.Gumbel(theta=self.cop_params[3])

        self.copulas = [self.clayton_cop, self.t_cop, self.gumbel_cop]

    def fit(self, data: pd.DataFrame, cdf1: Callable[[float], float], cdf2: Callable[[float], float],
            tol: float = 0.01, method: str = 'TNC') -> float:
        """
        Fit copula params (cop_params) and weights internally. Returns max log likelihood sum.
        
        The fit is by using max likelihood of each copula's parameter and the weights. In total 5 params (3 copula
        params and 2 weights) are fitted by 'TNC' (Truncated Newton) method by default from scipy.minimize package. 
        Note that the result is relatively sensitive to tol value. By making tol larger, the fit will be faster, 
        however the result may stay relatively far from true optimal due to its truncating nature. By making tol too
        small, the result may not converge. 0.01 seems to be a good balancing point but the user needs to prepare to
        tune it when necessary.
        
        Tested choices for numerical methods are 'TNC' and 'L-BFGS-B'. For our case, TNC tends to get a better fit,
        although takes a bit longer. Please DO NOT use other methods as they are tested as invalid for this framework.

        :param data: (pd.DataFrame) Data in (n, 2) pd.DataFrame used to fit the mixed copula.
        :param cdf1: (func) Cumulative density function trained, for the 0th column in data.
        :param cdf2: (func) Cumulative density function trained, for the 1st column in data.
        :param tol: (float) Optional. Tolerance for termination for the numerical optimizer. The user is expected to
            tune this parameter for potentially better fit on different data. Defaults to 0.01.
        :param method: (str) Optional. Numerical algorithm used for optimization. By default it uses TNC (Truncated
            Newton) for better performance. Also the user can use 'L-BFGS-B' for faster calculation, but the fit is
            generally not as good as 'TNC'. Please DO NOT use other methods as they are tested as invalid for this
            framework.
        :return: (float) Sum of log likelihood for the fit.
        """
        # Make a quantile_data DataFrame by mapping original data with marginal cdfs.
        quantile_data = data * 0
        quantile_data.iloc[:, 0] = cdf1(data.iloc[:, 0])
        quantile_data.iloc[:, 1] = cdf1(data.iloc[:, 1])
        # Fit the quantile data.
        fit_res = self._fit_quantile(quantile_data, tol, method)
        # Assign sum of log likelihood.
        log_likelihood = fit_res.fun
        # Internally construct the parameters and weights from the fit result.
        self.cop_params = tuple(fit_res.x[0: 4])
        self.weights = (fit_res.x[3], fit_res.x[4], 1 - fit_res.x[3] - fit_res.x[4])
        self.clayton_cop = cg.Clayton(theta=self.cop_params[0])
        corr = [[1, self.cop_params[1]], [self.cop_params[1], 1]]
        self.t_cop = cg.Student(cov=corr, nu=self.cop_params[2])
        self.gumbel_cop = cg.Gumbel(theta=self.cop_params[2])
        # List used for the MixedCopula superclass
        self.copulas = [self.clayton_cop, self.t_cop, self.gumbel_cop]

        return log_likelihood, fit_res
    
    # This is probably wrong
    # def generate_pairs(self, num: int) -> np.array:
    #     r"""
    #     Generate pairs according to P.D.F., stored in a 2D np.array.

    #     :param num: (int) Number of points to generate.
    #     :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
    #     """
    #     # Generate pairs of indep uniform dist vectors. Use numpy to generate.
    #     unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))
    #     # Generate samples from each copula and multiply by weights. Stored in a list.
    #     samples_list = [self.weights[i] * copula.generate_pairs(unif_vec=unif_vec)
    #                     for (i, copula) in enumerate(self.copulas)]
    #     # Add up each component for the mixed copula sample.
    #     sample_pairs = samples_list[0] + samples_list[1] + samples_list[2]

    #     return sample_pairs


    def _fit_quantile(self, quantile_data: pd.DataFrame, tol: float, method: str) -> (OptimizeResult):
        """
        Fitting cop_params and weights by max likelihood from data.

        System default param: tol = 0.01, method = 'TNC'. Quantile data needs to be in [0, 1].
        :param quantile_data: (pd.DataFrame) The quantile data to be used for fitting.
        :param tol: (float) Tolerance for termination for the numerical optimizer. The user is expected to
            tune this parameter for potentially better fit on different data. Defaults to 0.01 in the class.
        :param method: (str) Numerical algorithm used for optimization. By default it uses TNC (Truncated
            Newton) for better performance in the class. Also the user can use 'L-BFGS-B' for faster calculation, but
            the fit is generally not as good as 'TNC'. Please DO NOT use other methods as they are tested as invalid
            for this framework.
        :return (OptimizeResult): The optimized result from the numerical algorithm.
        """
        u1 = quantile_data.iloc[:, 0].to_numpy()
        u2 = quantile_data.iloc[:, 1].to_numpy()
        # Define the objective function.
        def neg_log_likelihood_mixcop(params):
            theta_c, rho_t, nu_t, theta_g, weight_c, weight_t = params
            # Edge case for frank and clayton. Turn them away from 0 but retain the sign.
            theta_c = self._away_from_0(theta_c)
            rho_t = self._away_from_0(rho_t)
            # Initiate 3 copulas with their own copula parameter respectively.
            clayton_cop = cg.Clayton(theta=theta_c)
            corr = [[1, rho_t], [rho_t, 1]]
            t_cop = cg.Student(cov=corr, nu=nu_t)
            gumbel_cop = cg.Gumbel(theta=theta_g)
            # Calculate log-likelihood respectively for each copula.
            likelihood_list_clayton = np.array([clayton_cop.c(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
            likelihood_list_t = np.array([t_cop.c(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
            likelihood_list_gumbel = np.array([gumbel_cop.c(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
            # Mix according to weights.
            likelihood_list_mix = weight_c * likelihood_list_clayton + weight_t * likelihood_list_t \
                + (1 - weight_c - weight_t) * likelihood_list_gumbel
            # Calculate sum of log likelihood respectively.
            log_likelihood_sum = np.sum(np.log(likelihood_list_mix))
            
            return -log_likelihood_sum  # Minimizing the negative of log likelihood.
        
        # ML fit for obj func neg_log_likelihood_mixcop(theta_c, theta_f, theta_g, weight_c, weight_f)
        params = np.array([3, 3, 0.8, 3, 0.33, 0.33])  # Initial guess
        # Constraint: theta_c in [-1, 100]. rho_t in [0.01, 0.99]. nu_t in [2, 15]. theta_g in [1, 100]
        # weight_c in [0, 1]. weight_f in [0, 1]
        bnds = ((-1, 100), (0.01, 0.99), (2, 10), (1, 100), (0, 1), (0, 1))
    
        res = minimize(neg_log_likelihood_mixcop, params, method=method, bounds=bnds,
                       options={'disp': False}, tol=tol)

        return res
    
    def _get_param(self):
        """
        Get the name and parameter(s) for this mixed copula instance.

        :return: (dict) Name and parameters for this copula.
        """

        descriptive_name = 'Bivariate Clayton-Frank-Gumbel Copula'
        class_name = 'CDFMixCop'
        cop_params = self.cop_params
        weights = self.weights
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'Clayton theta': cop_params[0], 'Student rho': cop_params[1], 'Student nu': cop_params[2],
                     'Gumbel theta': cop_params[3],
                     'Clayton weight': weights[0], 'Frank weight': weights[1],'Gumbel weight': weights[2]}

        return info_dict

class CFGMixCop(MixedCopula):
    """
    Clayton, Frank and Gumbel mixed copula.
    """
    def __init__(self, cop_params: tuple = None, weights: tuple = None):
        """
        Initiate Clayton, Frank and Gumbel (CFG) mixed copula.
        
        :param cop_params: (tuple) (3, ) size. Copula parameters for Clayton, Frank and Gumbel respectively.
        :param weights: (tuple) (3, ) size. Copulas weights for Clayton, Frank and Gumbel respectively. Needs to be
            positive and sum up to 1.
        """
        super().__init__()
        self.cop_params = cop_params
        self.weights = weights
        self.clayton_cop, self.frank_cop, self.gumbel_cop = None, None, None
        # Initiate component copulas if they are given cop_params and weights
        if cop_params is not None:
            self.clayton_cop = cg.Clayton(theta=self.cop_params[0])
            self.frank_cop = cg.Frank(theta=self.cop_params[1])
            self.gumbel_cop = cg.Gumbel(theta=self.cop_params[2])

        self.copulas = [self.clayton_cop, self.frank_cop, self.gumbel_cop]

    def fit(self, data: pd.DataFrame, cdf1: Callable[[float], float], cdf2: Callable[[float], float],
            tol: float = 0.01, method: str = 'TNC') -> float:
        """
        Fit copula params (cop_params) and weights internally. Returns max log likelihood sum.
        
        The fit is by using max likelihood of each copula's parameter and the weights. In total 5 params (3 copula
        params and 2 weights) are fitted by 'TNC' (Truncated Newton) method by default from scipy.minimize package. 
        Note that the result is relatively sensitive to tol value. By making tol larger, the fit will be faster, 
        however the result may stay relatively far from true optimal due to its truncating nature. By making tol too
        small, the result may not converge. 0.01 seems to be a good balancing point but the user needs to prepare to
        tune it when necessary.
        
        Possible choices for numerical methods are ('TNC', 'L-BFGS-B', 'SLSQP', 'Powell', 'trust-constr'). Please DO
        NOT use other methods as they are tested as invalid for this framework.

        :param data: (pd.DataFrame) Data in (n, 2) pd.DataFrame used to fit the mixed copula.
        :param cdf1: (func) Cumulative density function trained, for the 0th column in data.
        :param cdf2: (func) Cumulative density function trained, for the 1st column in data.
        :param tol: (float) Optional. Tolerance for termination for the numerical optimizer. The user is expected to
            tune this parameter for potentially better fit on different data. Defaults to 0.01.
        :param method: (str) Optional. Numerical algorithm used for optimization. By default it uses TNC (Truncated
            Newton) for better performance. Also the user can use 'L-BFGS-B' for faster calculation, but the fit is
            generally not as good as 'TNC'. Please DO NOT use other methods as they are tested as invalid for this
            framework.
        :return: (float) Sum of log likelihood for the fit.
        """
        # Make a quantile_data DataFrame by mapping original data with marginal cdfs.
        quantile_data = data * 0
        quantile_data.iloc[:, 0] = cdf1(data.iloc[:, 0])
        quantile_data.iloc[:, 1] = cdf1(data.iloc[:, 1])
        # Fit the quantile data.
        fit_res = self._fit_quantile(quantile_data, tol, method)
        # Assign sum of log likelihood.
        log_likelihood = fit_res.fun
        # Internally construct the parameters and weights from the fit result.
        self.cop_params = tuple(fit_res.x[0: 3])
        self.weights = (fit_res.x[3], fit_res.x[4], 1 - fit_res.x[3] - fit_res.x[4])
        self.clayton_cop = cg.Clayton(theta=self.cop_params[0])
        self.frank_cop = cg.Frank(theta=self.cop_params[1])
        self.gumbel_cop = cg.Gumbel(theta=self.cop_params[2])
        # List used for the MixedCopula superclass
        self.copulas = [self.clayton_cop, self.frank_cop, self.gumbel_cop]

        return log_likelihood
    
    # This is probably wrong
    def generate_pairs(self, num: int) -> np.array:
        r"""
        Generate pairs according to P.D.F., stored in a 2D np.array.

        :param num: (int) Number of points to generate.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """
        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))
        # Generate samples from each copula and multiply by weights. Stored in a list.
        samples_list = [self.weights[i] * copula.generate_pairs(unif_vec=unif_vec)
                        for (i, copula) in enumerate(self.copulas)]
        # Add up each component for the mixed copula sample.
        sample_pairs = samples_list[0] + samples_list[1] + samples_list[2]

        return sample_pairs


    def _fit_quantile(self, quantile_data: pd.DataFrame, tol: float, method: str) -> (OptimizeResult):
        """
        Fitting cop_params and weights by max likelihood from data.

        System default param: tol = 0.01, method = 'TNC'. Quantile data needs to be in [0, 1].
        :param quantile_data: (pd.DataFrame) The quantile data to be used for fitting.
        :param tol: (float) Tolerance for termination for the numerical optimizer. The user is expected to
            tune this parameter for potentially better fit on different data. Defaults to 0.01 in the class.
        :param method: (str) Numerical algorithm used for optimization. By default it uses TNC (Truncated
            Newton) for better performance in the class. Also the user can use 'L-BFGS-B' for faster calculation, but
            the fit is generally not as good as 'TNC'. Please DO NOT use other methods as they are tested as invalid
            for this framework.
        :return (OptimizeResult): The optimized result from the numerical algorithm.
        """
        u1 = quantile_data.iloc[:, 0].to_numpy()
        u2 = quantile_data.iloc[:, 1].to_numpy()
        # Define the objective function.
        def neg_log_likelihood_mixcop(params):
            theta_c, theta_f, theta_g, weight_c, weight_f = params
            # Edge case for frank and clayton. Turn them away from 0 but retain the sign.
            theta_c = self._away_from_0(theta_c)
            theta_f = self._away_from_0(theta_f)
            # Initiate 3 copulas with their own copula parameter respectively.
            clayton_cop = cg.Clayton(theta=theta_c)
            frank_cop = cg.Frank(theta=theta_f)
            gumbel_cop = cg.Gumbel(theta=theta_g)
            # Calculate log-likelihood respectively for each copula.
            likelihood_list_clayton = np.array([clayton_cop.c(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
            likelihood_list_frank = np.array([frank_cop.c(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
            likelihood_list_gumbel = np.array([gumbel_cop.c(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
            # Mix according to weights.
            likelihood_list_mix = weight_c * likelihood_list_clayton + weight_f * likelihood_list_frank \
                + (1 - weight_c - weight_f) * likelihood_list_gumbel
            # Calculate sum of log likelihood respectively.
            log_likelihood_sum = np.sum(np.log(likelihood_list_mix))
            
            return -log_likelihood_sum  # Minimizing the negative of log likelihood.
        
        # ML fit for obj func neg_log_likelihood_mixcop(theta_c, theta_f, theta_g, weight_c, weight_f)
        params = np.array([3, 3, 3, 0.33, 0.33])  # Initial guess
        # Constraint: theta_c in [-1, 100]. theta_f in [-50, 50]. theta_g in [1, 100]
        # weight_c in [0, 1]. weight_f in [0, 1]
        bnds = ((-1, 100), (-50, 50), (1, 100), (0, 1), (0, 1))
    
        res = minimize(neg_log_likelihood_mixcop, params, method=method, bounds=bnds,
                       options={'disp': False}, tol=tol)

        return res
    
    def _get_param(self):
        """
        Get the name and parameter(s) for this mixed copula instance.

        :return: (dict) Name and parameters for this copula.
        """

        descriptive_name = 'Bivariate Clayton-Frank-Gumbel Copula'
        class_name = 'CDFMixCop'
        cop_params = self.cop_params
        weights = self.weights
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'Clayton theta': cop_params[0], 'Frank theta': cop_params[1], 'Gumbel theta': cop_params[2],
                     'Clayton weight': weights[0], 'Frank weight': weights[1],'Gumbel weight': weights[2]}

        return info_dict
