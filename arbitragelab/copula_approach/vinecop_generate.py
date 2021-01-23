# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module that generates vine copulas.

Built on top of the :code:`pyvinecoplib` package. See https://github.com/vinecopulib/pyvinecopulib for more details.
"""

from abc import ABC, abstractmethod
from typing import List, Union
from scipy.optimize import minimize
from itertools import permutations
import numpy as np
import pandas as pd
import pyvinecopulib as pv
import scipy.integrate as integrate

class CVineCop():
    """
    Class for C-vine copulas.
    
    This is a wrapper class of pv.Vinecop that provides useful methods for statistical arbitrage. One key note is
    that, to keep the notaion consistent with literature consensus, all variables started with pv are indexed from
    1, not 0. In this way it is easier to see the nodes of the tree.
    """

    def __init__(self, cvine_cop: pv.Vinecop = None):
        """
        Initiate a C-vine copula class.

        :param cvine_cop: (pv.Vinecop) Optional. A fitted C-vine copula. Defaults to None.
        """

        # All the bivariate classes used to construct the vine copula.
        self._bicop_family = [pv.BicopFamily.bb1, pv.BicopFamily.bb6, pv.BicopFamily.bb7, pv.BicopFamily.bb8,
                              pv.BicopFamily.clayton, pv.BicopFamily.student, pv.BicopFamily.frank,
                              pv.BicopFamily.gaussian, pv.BicopFamily.gumbel, pv.BicopFamily.indep]
        # The pv.Vinecop being wrapped
        self.cvine_cop = cvine_cop

    def fit(self, data: pd.DataFrame, pv_target_idx: int = 1, if_renew: bool = True):
        """
        Fit data to the C-vine copula by selecting the center stock.

        The method will loop through all possible C-vine structures and choose the best fit by AIC. The targeted
        stock will never show up in the center of the tree, thus some C-vine structures need not be included. This
        method is relatively slow, and the complexity is O(n!) with n being the number of stocks. Hence please keep
        n <= 7.

        The original pv.Vinecop will be returned just in case there is some usage not covered by this class.

        :param data: (pd.DataFrame) The quantile data to be used to fit the C-vine copula.
        :param pv_target_idx: (int) Optional. The stock to be targeted for trading. This is indexed from 1, hence
            1 corresponds to the 0th column data in the data frame. Defalts to 1.
        :param if_renew: (bool) Optional. Whether to update the class attribute cvine_cop. Defaults to True.
        :return: (pv.Vinecop) The fitted pv.Vinecop object.
        """

        # Initializing.
        data_np = data.to_numpy()  # pyvinecopulib uses numpy arrays for fitting.
        data_dim = len(data.columns)  # Number of stocks.
        # List of all possible C-vine structures, a list of tuples.
        possible_cvine_structures = self._get_possible_cvine_structs(data_dim, pv_target_idx)

        # Fit among all possible structures.
        controls = pv.FitControlsVinecop(family_set = self._bicop_family)  # Bivar copula constituents for the C-vine.
        aics = dict()  # Dictionary for AIC values for all candidate C-vine copulas.
        cvine_cops = dict()  # Dictionary for storing all candiate C-vine copulas.
        for cvine_structure in possible_cvine_structures:
            temp_cvine_struct = pv.CVineStructure(order=cvine_structure)  # Specific C-vine structure
            temp_cvine_cop = pv.Vinecop(structure=temp_cvine_struct)  # Construct the C-vine copula
            temp_cvine_cop.select(data=data_np, controls=controls)  # Fit to data
            aics[cvine_structure] = temp_cvine_cop.aic(data_np)  # Calculate AIC
            cvine_cops[cvine_structure] = temp_cvine_cop  # Store the C-vine copula candidate

        # Select the structure that has the lowest aics
        aics = pd.Series(aics)
        fitted_cvine_strcture = aics.idxmin()  # aics indexed by the structure (tuple)

        # Generate a C-vine copula for the system
        fitted_cvine_cop = cvine_cops[fitted_cvine_strcture]

        if if_renew:  # Whether to renew the class C-vine copula.
            self.cvine_cop = fitted_cvine_cop

        return fitted_cvine_cop

    @staticmethod
    def _get_possible_cvine_structs(data_dim: int, pv_target_idx: int) -> List[tuple]:
        """
        Get all possible C-vine structures specified by the dimension and the targeted node.
        
        A C-vine copula is uniquelly determined by an ordered tuple, listing the center of each tree at every level
        read backwards. For example, for a 4-dim C-vine characterized by (4, 2, 1, 3), node 3 is the center for the
        0th tree, node 1 is the center for the 1st tree and so on. The targeted node representing the stock of
        interest, cannot be the at the center of every tree (except the last) and thus has to be the 0th element in the
        tuple, as claimed in [Stubinger et al. 2016]. For a tuple (4, 2, 1, 3), it says stock 4 is the targeted stock.

        :param data_dim: (int) The number of stocks.
        :param pv_target_idx: (int). The stock to be targeted for trading. This is indexed from 1, hence 1 corresponds
            to the 0th column data in a pandas data frame.
        :return: (List[tuple]) The list of all possible C-vine structures stored as tuples.
        """

        # Initiating
        all_items = list(range(1, data_dim+1))  # All the nodes, indexed from 1
        all_structures = permutations(all_items)  # All the possible stuctures, as permutations of the nodes

        # Loop thourgh all the structures 
        valid_structures = []
        for structure in all_structures:
            if structure[0] == pv_target_idx: # Only keep the ones where the targeted index is the 0-th element
                valid_structures.append(structure)
        
        return valid_structures

    def get_condi_prob(self, u: pd.Series, pv_target_idx: int = 1, eps: float = 1e-4) -> float:
        r"""
        Get the conditional probability of the C-vine copula.

        For example, if we have 5 stocks and pv_target_idx = 2, then this method calculates:
            P(U2 <= u2 | U1=u1, U3=u3, U4=u2, U5=u5)

        The calculation is numerical by integrating along the margin. By default it uses the 0th element of u as the
        target. Also this function's value is wrapped within [eps, 1-eps] to avoid potential edge values by default.

        :param u: (pd.Series) The vector value of quantiles.
        :param pv_target_idx: (int) Optional. The stock to be targeted for trading. This is indexed from 1, hence 1
            corresponds to the 0th column data in a pandas data frame. In this case it is the only variable not
            conditioned on. Defaults to 1.
        :param eps: (float) Optional. The small value that keeps results within [eps, 1-eps]. Defaults to 1e-4.
        :return: (float) The calculated conditional probability.
        """

        # Initiating
        u = np.array(u)  # Casting to 1D numpy array
        target_idx = pv_target_idx - 1  # Indexing from 0
        target = u[target_idx]  # Upper lim of the integration
        others = np.delete(u, target_idx)  # Those will be served as arguments in quad
        
        # The integrand function is the pdf
        def pdf_func(target, others):
            # the pv.Vinecop.pdf only takes 2D arrays as input. Here it is assembled back
            u_vec = np.insert(arr=others, obj=target_idx, values=target).reshape((1, 4))
            pdf = self.cvine_cop.pdf(u_vec)[0]

            return pdf

        # Integrating along the margin and normalizing the conditional probability
        sum_prob = integrate.quad(pdf_func, 0, target, args=others)[0]
        total_prob = integrate.quad(pdf_func, 0, 1, args=others)[0]
        condi_prob =  sum_prob / total_prob
        
        # Map the condi prob into [eps, 1-eps]
        wrapped_condi_prob = max(min(condi_prob, 1-eps), eps)

        return wrapped_condi_prob
    
    def get_cop_densities(self, u: Union[pd.DataFrame, np.array], num_threads: int = 1) -> Union[pd.Series, float]:
        """
        Calculate probability density of the vine copula.

        Result is analytical. You may also take advantage of multi-thread calculation. The result will be either a
        pandas series of numbers or a single number depends on the dimension of the input.

        :param u: (Union[pd.DataFrame, np.array]) The quantiles data to be used. The input can be a pandas dataframe,
            or a numpy array vector. The formal case yields the result with in pandas series in matching indices, and
            the latter yields a single float number.
        :param num_threads: (int) Optional. The number of threads to use for calculation. Defaults to 1.
        :return: (Union[pd.Series, float]) The calculated pdf. If the input is a dataframe then the result is a series
            with matching indices. If the input is a 1D np.array then the result is a float.
        """

        # When the input is a 1D np.array
        if isinstance(u, np.ndarray) and u.ndim == 1:
            u_np = u.reshape((1, -1))
            pdf = self.cvine_cop.pdf(u_np, num_threads)[0]
            # The output is a float
            return pdf

        # When tne input is a pd.Dataframe
        u_np = np.array(u)
        pdfs = self.cvine_cop.pdf(u_np, num_threads)
        pdfs_series = pd.Series(pdfs, index=u.index)
        # The result is a pd.Series with matching indices
        return pdfs_series

    def get_cop_evals(self, u: pd.DataFrame, mcn: int = 1e4, num_threads: int = 1) -> pd.Series:
        """
        Calculate cumulative density of the vine copula.

        Result is numerical through Monte-Carlo integration. You may also take advantage of multi-thread calculation.
        The result will be either a pandas series of numbers or a single number depends on the dimension of the input.

        :param u: (Union[pd.DataFrame, np.array]) The quantiles data to be used. The input can be a pandas dataframe,
            or a numpy array vector. The formal case yields the result with in pandas series in matching indices, and
            the latter yields a single float number.
        :param mcn: (int) Integer for the number of quasi-random numbers to draw to evaluate the distribution for the
            Monte-Carlo integration. Defaults to 1e4.
        :param num_threads: (int) Optional. The number of threads to use for calculation. Defaults to 1.
        :return: (Union[pd.Series, float]) The calculated cdf. If the input is a dataframe then the result is a series
            with matching indices. If the input is a 1D np.array then the result is a float.
        """
        
        # When the input is a 1D np.array
        if isinstance(u, np.ndarray) and u.ndim == 1:
            u_np = u.reshape((1, -1))
            cdf = self.cvine_cop.cdf(u_np, mcn, num_threads)[0]
            # The output is a float
            return cdf

        # When tne input is a pd.Dataframe
        u_np = np.array(u)
        cdfs = self.cvine_cop.cdf(u_np, mcn, num_threads)
        cdfs_series = pd.Series(cdfs, index=u.index)
        # The result is a pd.Series with matching indices
        return cdfs_series
    
    def simulate(self, n: int, qrn: bool = False, num_threads: int = 1, seeds: List[int] = []) -> np.ndarray:
        """
        Simulate from a vine copula model.

        :param n: (int) Number of observations.
        :param qrn: (bool) Optional. Set to True for quasi-random numbers. Defaults to False.
        :param num_threads: (int) Optional. The number of threads to use for calculation. Defaults to 1.
        :param seeds: (List[int]) Optional. Seeds of the random number generator. If empty then the random generator
            will be seeded randomly. Defaults to False.
        :return: (pd.ndarray) The generated random samples from the vine copula.
        """

        simulated_samples = self.cvine_cop.simulate(n, qrn, num_threads, seeds)

        return simulated_samples
    
    def aic(self, u: pd.Dataframe, num_threads: int = 1) -> float:
        """
        Evaluates the Akaike information criterion (AIC).
        
        :param u: (pd.DataFrame) The quantile data used for evaluation.
        :param num_threads: (int) Optional. The number of threads to use for calculation. Defaults to 1.
        :return: (float) calculated AIC value.
        """

        u_np = u.to_numpy()  # Transit to numpy ndarrays
        aic_value = self.cvine_cop.aic(u_np, num_threads)

        return aic_value

    def bic(self, u: pd.Dataframe, num_threads: int = 1) -> float:
        """
        Evaluates the Bayesian information criterion (BIC).
        
        :param u: (pd.DataFrame) The quantile data used for evaluation.
        :param num_threads: (int) Optional. The number of threads to use for calculation. Defaults to 1.
        :return: (float) calculated BIC value.
        """

        u_np = u.to_numpy()  # Transit to numpy ndarrays
        bic_value = self.cvine_cop.bic(u_np, num_threads)

        return bic_value

    def loglik(self, u: pd.Dataframe, num_threads: int = 1) -> float:
        """
        Evaluates the Sum of log-likelihood.
        
        :param u: (pd.DataFrame) The quantile data used for evaluation.
        :param num_threads: (int) Optional. The number of threads to use for calculation. Defaults to 1.
        :return: (float) calculated sum of log-likelihood.
        """

        u_np = u.to_numpy()  # Transit to numpy ndarrays
        loglik_value = self.cvine_cop.loglik(u_np, num_threads)

        return loglik_value

