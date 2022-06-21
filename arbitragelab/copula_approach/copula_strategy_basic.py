# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Master module that implements the basic copula trading strategy.

This module is almost identical in terms of functionality as copula_strategy. But is designed with better efficiency,
better structure, native pandas support, and supports mixed copulas. The trading logic is more clearly defined and all
wrapped in one method for easier adjustment when needed, due to the ambiguities from the paper.
"""

# pylint: disable = invalid-name, too-many-instance-attributes, abstract-class-instantiated
from typing import Callable, Tuple, Union
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import scipy.stats as ss

from arbitragelab.copula_approach.elliptical import (GaussianCopula, StudentCopula)
from arbitragelab.copula_approach.switcher import Switcher
import arbitragelab.copula_approach.base as cop
import arbitragelab.copula_approach.copula_calculation as copcalc
import arbitragelab.copula_approach.mixed_copulas as copmix
from arbitragelab.util import segment


class BasicCopulaStrategy:
    """
    Analyze a pair of stock prices using copulas.

    This module is a realization of the methodology in the following paper:
    `Liew, R.Q. and Wu, Y., 2013. Pairs trading: A copula approach. Journal of Derivatives & Hedge Funds, 19(1), pp.12-30.
    <https://dr.ntu.edu.sg/bitstream/10220/17826/1/jdhf20131a.pdf>`__

    We use the convention that the spread is defined as stock 1 in relation to stock 2.
    This class provides the following functionalities:

        1. Maximum likelihood fitting to training data. User provides the name and necessary parameters.
        2. Use a given copula to generate trading positions based on test data. By default it uses
           the fitted copula generated in fit_copula() method.
    """

    def __init__(self, copula: Union[cop.Copula, copmix.MixedCopula] = None, open_thresholds: tuple = (0.05, 0.95),
                 exit_thresholds: tuple = (0.5, 0.5)):
        """
        Initiate a BasicCopulaStrategy class.

        One can choose to initiate with no arguments, or to initiate with a given copula as the system's
        Copula.

        :param copula: (Copula, MixedCopula) Optional. A copula object that the class will use for all analysis. If
            there is no input then fit_copula method will create one when called.
        :param open_thresholds: (tuple) Optional. The default lower and upper threshold for opening a position for
            trading signal generation. Defaults to (0.05, 0.95).
        :param exit_thresholds: (tuple) Optional. The default lower and upper threshold for exiting a position for
            trading signal generation. Defaults to (0.5, 0.5).
        """

        # Copulas that uses theta as parameter
        self.archimedean_names = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14']
        self.elliptical_names = ['Gaussian', 'Student']
        self.mixed_cop_names = ['CFGMixCop', 'CTGMixCop']
        self.all_copula_names = self.archimedean_names + self.elliptical_names + self.mixed_cop_names

        # To be used for the test data set
        self.copula = copula

        # Default thresholds for opening trading positions
        self.l_open_threshold = open_thresholds[0]
        self.u_open_threshold = open_thresholds[1]
        # Default thresholds for exiting trading positions
        self.l_exit_threshold = exit_thresholds[0]
        self.u_exit_threshold = exit_thresholds[1]
        # Internal counters for different signal triggers
        self._long_count = 0
        self._short_count = 0
        self._exit_count = 0

        segment.track('BasicCopulaStrategy')

    @staticmethod
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

        column_count = len(data.columns)  # Number of columns
        cdf_lst = [None] * column_count  # List to store all marginal cdf functions
        quantile_data_lst = [None] * column_count  # List to store all quantile data in pd.Series

        # Loop through all columns
        for i in range(column_count):
            cdf_lst[i] = copcalc.construct_ecdf_lin(data.iloc[:, i])
            quantile_data_lst[i] = data.iloc[:, i].map(cdf_lst[i])

        quantile_data = pd.concat(quantile_data_lst, axis=1)  # Form the quantile DataFrame

        return quantile_data, cdf_lst

    def fit_copula(self, data: pd.DataFrame, copula_name: str, if_renew: bool = True,
                   **fit_params_kwargs) -> tuple:
        """
        Fit a copula to data.

        This function will internally calculate the quantile from data and use it to fit a given type of copula. For
        Archimedean and Gaussian copula, it uses a pseudo-maximum likelihood algorithm by matching the Kendall's tau
        from data with the copula parameter that describes the dependency structure (i.e., theta for Archimedean, rho
        for Gaussian). For Student copula, it uses a two-step algorithm by finding rho at first from Kendall's tau,
        then uses L-BFGS-B algorithm to numerically find the best nu to maximize likelihood. For mix copulas, it uses
        a two-step expectation maximization (EM) algorithm. Please refer to the mixed copula classes for related
        details.

        Archimedean and Gaussian copula's fit is instant, Student-t copula will take a few seconds for a thousand
        data points, and mixed copula's fit can range from 10 seconds to a 5 minutes depending on how fast it converges.
        Also, CFGMixCop converges much faster than CTGMixCop. The two available mixed copulas are Clayton-Frank-Gumbel
        and Clayton-Student-Gumbel, specifically chosen so that they can handle more complex tail dependencies.

        fit_params_kwargs deal with key word arguments to fine tune the parameters in the fitting algorithm.

            - 'nu_tol': (float) Optional. The final accuracy for finding nu for Student-t copula. Defaults to 0.05.
            - 'max_iter': (int) Optional. Maximum iteration for the EM method. Defaults to 25.
            - 'gamma_scad': (float) Optional. Tuning parameter for the SCAD penalty term. Defaults to 0.7.
            - 'a_scad': (float) Optional. Tuning parameter for the SCAD penalty term. Defaults to 6.

        The method returns:

            - result_dict: (dict) The name of the copula and its SIC, AIC, HQIC, Log-Likelihood values;
            - copula: (Copula) The fitted copula;
            - cdf1: (func) The cumulative density function for stock 1, using training data;
            - cdf2: (func) The cumulative density function for stock 2, using training data.

        :param data: (pd.DataFrame) Data used to fit the copula to.
        :param copula_name: (str) Name of the copula to fit to. Choices including 'Gumbel', 'Clayton', 'Frank',
            'Joe', 'N13', 'N14', 'Gaussian', 'Student', 'CFGMixCop', 'CTGMixCop'.
        :param if_renew: (bool) Optional. Whether use the fitted copula to replace the class copula in
            BasicCopulaStrategy.
        :param fit_params_kwargs: (dict) Optional. Key word arguments for fine tuning the fitting parameters. See
            the description part.
        :return: (tuple) The fit result (SIC, AIC, HQIC, Log-likelihood) in a dictionary, the fitted copula, the
            marginal cdf for the 0th column of data and the marginal cdf for the 1st column of data.
        """

        # Converting to quantile data for some of the methods
        num_of_instances = len(data)
        quantile_data, cdfs = self.to_quantile(data)
        cdf1 = cdfs[0]
        cdf2 = cdfs[1]

        # Initiate variables to be returned
        fitted_copula = None
        sic_value, aic_value, hqic_value, log_likelihood = None, None, None, None

        # Fitting an Archimedean copula
        if copula_name in self.archimedean_names:
            log_likelihood, fitted_copula = self._pseudo_ml_archimedean_fit(quantile_data, copula_name)
            # Only 1 param (theta) is estimated.
            sic_value, aic_value, hqic_value = self.get_info_criterion(log_likelihood, n=num_of_instances, k=1)

        # Fitting a Gaussian copula
        if copula_name == 'Gaussian':
            log_likelihood, fitted_copula = self._gaussian_cop_fit(quantile_data)
            # Only 1 param (rho) is estimated
            sic_value, aic_value, hqic_value = self.get_info_criterion(log_likelihood, n=num_of_instances, k=1)

        # Fitting a Student-t copula
        if copula_name == 'Student':
            log_likelihood, fitted_copula = self._t_cop_fit_two_step(quantile_data, **fit_params_kwargs)
            # 2 params (rho, nu) are estimated
            sic_value, aic_value, hqic_value = self.get_info_criterion(log_likelihood, n=num_of_instances, k=2)

        # Fitting mixed copulas. This is a wrapper around their own fit method so we use the original data
        if copula_name in self.mixed_cop_names:
            log_likelihood, fitted_copula = self._fit_mixed_cop(data, copula_name, **fit_params_kwargs)
            # Calculate scores in terms of various information criteria
            if copula_name == 'CFGMixCop':
                # 3 copula params, 2 weights are estimated
                sic_value, aic_value, hqic_value = self.get_info_criterion(log_likelihood, n=num_of_instances, k=5)

            if copula_name == 'CTGMixCop':
                # 4 copula params, 2 weights are estimated
                sic_value, aic_value, hqic_value = self.get_info_criterion(log_likelihood, n=num_of_instances, k=6)

        # Put the name of the copula and score in the dictionary
        result_dict = {'Copula Name': copula_name, 'SIC': sic_value, 'AIC': aic_value, 'HQIC': hqic_value,
                       'Log-likelihood': log_likelihood}

        if if_renew:  # Whether to renew the class's copula using the fitted copula
            self.copula = fitted_copula

        return result_dict, fitted_copula, cdf1, cdf2

    @staticmethod
    def get_info_criterion(log_likelihood: float, n: int, k: int) -> Tuple[float, float, float]:
        """
        Run SIC, AIC, and HQIC from the sum of log likelihood.

        This method only works if CopulaStrategy has fitted a copula given training data.

        :param log_likelihood: (float) Sum of log likelihood.
        :param n: (int) The number of data points.
        :param k: (int) The number of estimated variables.
        :return: (tuple) Result of SIC, AIC and HQIC.
        """

        sic_value = copcalc.sic(log_likelihood, n, k)
        aic_value = copcalc.aic(log_likelihood, n, k)
        hqic_value = copcalc.hqic(log_likelihood, n, k)

        return sic_value, aic_value, hqic_value

    @staticmethod
    def _pseudo_ml_archimedean_fit(data: pd.DataFrame, copula_name: str) -> tuple:
        """
        Fit a bivariate Archimedean copula by pseudo-max likelihood.

        Using Kendall's tau to calculate theta for each Archimedean copula.

        :param data: (pd.DataFrame) Data used to fit the copula to.
        :param copula_name: (str) Name of the copula to fit to.
        :return: (tuple) The calculated log-likelihood for the fitted copula, the fitted copula.
        """

        x = data.iloc[:, 0].to_numpy()
        y = data.iloc[:, 1].to_numpy()

        switch = Switcher()  # Initiate a switcher class to initiate copula by its name in string

        # Calculate Kendall's tau from data
        tau = ss.kendalltau(x, y)[0]
        # Calculate theta hat from the specific copula using Kendall's tau
        temp_arch_copula = switch.choose_copula(copula_name=copula_name)
        theta_hat = temp_arch_copula.theta_hat(tau)
        # Use the result to instantiate a copula as the fitted copula
        fitted_copula = switch.choose_copula(copula_name=copula_name, theta=theta_hat)
        # Calculate the sum of log likelihood
        log_likelihood = np.sum(np.log([fitted_copula.c(xi, yi) for (xi, yi) in zip(x, y)]))

        return log_likelihood, fitted_copula

    @staticmethod
    def _gaussian_cop_fit(data: pd.DataFrame) -> tuple:
        """
        Fit a bivariate Gaussian copula by pseudo-max likelihood.

        Using Kendall's tau to calculate rho for the Gaussian copula.

        :param data: (pd.DataFrame) Data used to fit the copula to.
        :return: (tuple) The calculated log-likelihood for the fitted copula, the fitted copula.
        """

        x = data.iloc[:, 0].to_numpy()
        y = data.iloc[:, 1].to_numpy()

        # Calculate Kendall's tau from data
        tau = ss.kendalltau(x, y)[0]
        # Calculate rho hat from the specific copula using Kendall's tau
        dud_cov = [[1, 0.5], [0.5, 1]]
        temp_gaussian_copula = GaussianCopula(cov=dud_cov)
        rho_hat = temp_gaussian_copula.theta_hat(tau)
        cov_hat = [[1, rho_hat], [rho_hat, 1]]
        # Use the result to instantiate a copula as the fitted copula
        fitted_copula = GaussianCopula(cov=cov_hat)
        # Calculate the sum of log likelihood
        log_likelihood = np.sum(np.log([fitted_copula.get_cop_density(xi, yi) for (xi, yi) in zip(x, y)]))

        return log_likelihood, fitted_copula

    @staticmethod
    def _t_cop_fit_two_step(data: pd.DataFrame, **fit_params_kwargs) -> tuple:
        """
        Fit a bivariate Student-t copula.

        Step 1: Using Kendall's tau to calculate rho for the Student-t copula.
        Step 2: Maximizing likelihood to find the nu.

        :param data: (pd.DataFrame) Data used to fit the copula to.
        :param fit_params_kwargs: (dict) The grid length for searching nu in maximum likelihood. The key is 'nu_tol'.
        :return: (tuple) The calculated log-likelihood for the fitted copula, the fitted copula.
        """

        # Default the nu_tol to 0.05
        nu_tol = fit_params_kwargs.get('nu_tol', 0.05)

        x = data.iloc[:, 0].to_numpy()
        y = data.iloc[:, 1].to_numpy()

        # 1. Fit data to find the correlation matrix
        # Calculate Kendall's tau from data
        tau = ss.kendalltau(x, y)[0]
        dud_cov = [[1, 0.5], [0.5, 1]]  # Dud param to initiate a t copula for calculation
        dud_nu = 4  # Dud param to initiate a t copula for calculation
        temp_t_copula = StudentCopula(cov=dud_cov, nu=dud_nu)
        # Calculate rho hat from the specific copula using Kendall's tau
        rho_hat = temp_t_copula.theta_hat(tau)
        cov_hat = [[1, rho_hat], [rho_hat, 1]]

        # 2. Max likelihood to find the degree of freedom nu
        # Define the objective function
        def neg_log_likelihood_for_t_copula(nu):

            temp_t_cop = StudentCopula(cov=cov_hat, nu=nu)
            log_likelihood_local = np.sum(np.log([temp_t_cop.get_cop_density(xi, yi) for (xi, yi) in zip(x, y)]))

            return -1 * log_likelihood_local  # Minimizing the negative of likelihood

        # Optimizing to find best nu
        nu0 = np.array([3])
        # Constraint: nu between [1, 15]. Too large nu value will lead to calculation issues for gamma function
        bnds = ((2, 15),)

        res = minimize(neg_log_likelihood_for_t_copula, nu0, method='L-BFGS-B', bounds=bnds,
                       options={'disp': False}, tol=nu_tol)

        nu_hat = res['x'][0]

        # 3. Return result
        # Use the result to instantiate a copula as the fitted copula
        fitted_copula = StudentCopula(cov=cov_hat, nu=nu_hat)
        # Calculate the sum of log likelihood
        log_likelihood = np.sum(np.log([fitted_copula.get_cop_density(xi, yi) for (xi, yi) in zip(x, y)]))

        return log_likelihood, fitted_copula

    @staticmethod
    def _fit_mixed_cop(data: pd.DataFrame, copula_name: str, **fit_params_kwargs) -> tuple:
        """
        Fit the mixed copula by expectation maximization (EM).

        A wrapper around each mixed copula's own fit function.

        :param data: (pd.DataFrame) Data used to fit the copula to.
        :param copula_name: (str) Name of the copula to fit to.
        :param fit_params_kwargs: (dict) The grid length for searching nu in maximum likelihood. The available keys are
            'max_iter', 'gamma_scad', 'a_scad', 'weight_margin'.
        :return: (tuple) The calculated log-likelihood for the fitted copula, the fitted copula.
        """

        # Set the default fitting params
        max_iter = fit_params_kwargs.get('max_iter', 25)
        gamma_scad = fit_params_kwargs.get('gamma_scad', 0.6)
        a_scad = fit_params_kwargs.get('a_scad', 6)
        weight_margin = fit_params_kwargs.get('weight_margin', 1e-2)

        copula = None
        log_likelihood = None

        # Wrapping around each mixed copula's own fit function
        if copula_name == 'CFGMixCop':
            copula = copmix.CFGMixCop()
            log_likelihood = copula.fit(data, max_iter, gamma_scad, a_scad, weight_margin)

        if copula_name == 'CTGMixCop':
            copula = copmix.CTGMixCop()
            log_likelihood = copula.fit(data, max_iter, gamma_scad, a_scad, weight_margin)

        return log_likelihood, copula

    def get_positions(self, data: pd.DataFrame, cdf1: Callable[[float], float], cdf2: Callable[[float], float],
                      init_pos: int = 0, open_thresholds: Tuple[float, float] = None,
                      exit_thresholds: Tuple[float, float] = None, exit_rule: str = 'and') -> pd.Series:
        r"""
        Get positions from the basic copula strategy.

        This is the threshold basic copula trading strategy implemented by [Liew et al. 2013]. First, one uses
        formation period prices to train a copula, then trade based on conditional probabilities calculated from the
        quantiles of the current price u1 and u2. If we define the spread as stock 1 in relation to stock 2, then the
        logic is as follows (All the thresholds can be customized via open_thresholds, exit_thresholds parameters):

            - If P(U1 <= u1 | U2 = u2) <= 0.05 AND P(U2 <= u2 | U1 = u1) >= 0.95, then stock 1 is under-valued and
              stock 2 is over-valued. Thus we short the spread.
            - If P(U1 <= u1 | U2 = u2) >= 0.95 AND P(U2 <= u2 | U1 = u1) <= 0.05, then stock 2 is under-valued and
              stock 1 is over-valued. Thus we long the spread.
            - We close the position if the conditional probabilities cross with 0.5.

        For the exiting condition, the author proposed a closure when stock 1 AND 2's conditional probabilities cross
        0.5. However, we found it sometimes too strict and fails to exit a position when it should occasionally. Hence
        we also provide the OR logic implementation. You can use it by setting exit_rule='or'. Also note that the
        signal generation is independent from the current position.

        The positions will be given in pd.Series of integers as 0: no position, 1: long the spread, -1: short the
        spread.

        :param data: (pd.DataFrame) The trading period data for two stocks.
        :param cdf1: (func) Marginal C.D.F. for stock 1 in training period.
        :param cdf2: (func) Marginal C.D.F. for stock 2 in training period.
        :param init_pos: (int) Optional. Starting position. Defaults to no position.
        :param open_thresholds: (tuple) Optional. The default lower and upper threshold for opening a position for
            trading signal generation. Defaults to (0.05, 0.95).
        :param exit_thresholds: (tuple) Optional. The default lower and upper threshold for exiting a position for
            trading signal generation. Defaults to (0.5, 0.5).
        :param exit_rule: (str) Optional. The logic for triggering an exit signal. Available choices are 'and', 'or'.
            They indicate whether both conditional probabilities need to cross 0.5. Defaults to 'and'.
        :return: (pd.Series) The suggested positions for the given price data.
        """

        # Map to quantile data using the trained marginal cdfs
        quantile_data_1 = data.iloc[:, 0].map(cdf1)
        quantile_data_2 = data.iloc[:, 1].map(cdf2)
        quantile_data = pd.concat((quantile_data_1, quantile_data_2), axis=1)

        # Reset the signal counters
        self._long_count, self._short_count, self._exit_count = 0, 0, 0

        num_of_instances = len(data)

        # Initiate positions Series
        positions = pd.Series(np.nan, index=quantile_data_1.index)
        positions[0] = init_pos

        # Updating open thresholds. Otherwise keep the class default value
        if open_thresholds is not None:
            self.l_open_threshold = open_thresholds[0]
            self.u_open_threshold = open_thresholds[1]
        # Updating exit thresholds. Otherwise keep the class default value
        if exit_thresholds is not None:
            self.l_exit_threshold = exit_thresholds[0]
            self.u_exit_threshold = exit_thresholds[1]

        condi_probs = self.get_condi_probs(quantile_data)  # All conditional probabilities
        who_exits = np.zeros(2)  # Initially there is no crossing
        for i in range(1, num_of_instances):  # Loop through the conditional probs to form positions
            positions[i], who_exits = self.get_cur_position(condi_probs=condi_probs.iloc[i, :],
                                                            pre_condi_probs=condi_probs.iloc[i-1, :],
                                                            pre_pos=positions[i-1],
                                                            exit_rule=exit_rule,
                                                            who_exits=who_exits)

        return positions

    def get_condi_probs(self, quantile_data: pd.DataFrame) -> pd.DataFrame:
        """
        Get conditional probabilities given the data.

        The input data needs to be quantile. The system should have a copula fitted to use. Make sure the quantile data
        does not have any NaN values.

        :param quantile_data: (pd.DataFrame) Data frame in quantiles with two columns.
        :return: (pd.DataFrame) The conditional probabilities calculated.
        """

        # Check if there is any NaN value
        has_nan = quantile_data.isnull().values.any()
        if has_nan:
            raise ValueError('Must not have NaN values in quantile_data.')

        # Initiate a data frame with zeros and the same index
        condi_probs = pd.DataFrame(np.nan, index=quantile_data.index, columns=quantile_data.columns)

        for row_count, row in enumerate(quantile_data.iterrows()):
            condi_probs.iloc[row_count] = [self.copula.get_condi_prob(row[1][0], row[1][1]),
                                           self.copula.get_condi_prob(row[1][1], row[1][0])]

        return condi_probs

    def get_cur_position(self, condi_probs: pd.Series, pre_condi_probs: pd.Series, pre_pos: int, exit_rule: str,
                         who_exits: np.array) -> Tuple[int, np.array]:
        """
        Determine the current position using history one step back.

        This is the threshold basic copula trading strategy implemented by [Liew et al. 2013]. One uses at first use
        formation period prices to train a copula, then trade based on conditional probabilities calculated from the
        quantiles of the current price u1 and u2. If we define the spread as stock 1 in relation to stock 2, then the
        logic is as follows (All the thresholds can be customized via open_thresholds, exit_thresholds parameters):

            - If P(U1 <= u1 | U2 = u2) <= 0.05 AND P(U2 <= u2 | U1 = u1) >= 0.95, then stock 1 is under-valued and stock
              2 is over-valued. Thus we short the spread.
            - If P(U1 <= u1 | U2 = u2) >= 0.95 AND P(U2 <= u2 | U1 = u1) <= 0.05, then stock 2 is under-valued and stock
              1 is over-valued. Thus we long the spread.
            - We close the position if the conditional probabilities cross with 0.5.

        For the exiting condition, the author proposed a closure when stock 1 AND 2's conditional probabilities cross
        0.5. However, we found it sometimes too strict and fails to exit a position when it should occasionally. Hence
        we also provide the OR logic implementation. You can use it by setting exit_rule='or'. Also note that the
        signal generation is independent from the current position.

        The position will be given in an integer as 0: no position, 1: long the spread, -1: short the spread.

        Note: numpy series will also work for condi_probs and pre_condi_probs. Also this is the only function to modify
        if you wish to change the precedence of different signals.

        :param condi_probs: (pd.Series) The current conditional probabilities for the stocks pair.
        :param pre_condi_probs: (pd.Series) The previous conditional probabilities for the stocks pair.
        :param pre_pos: (int) The previous trading position. Valid options are 0, 1, -1.
        :param exit_rule: (str) The logic for triggering an exit signal. Available choices are 'and', 'or'.
            They indicate whether both conditional probabilities need to cross 0.5.
        :param who_exits: (np.array) A binary (2, ) array indicating which stock has its conditional probability
            crossed with 0.5. For example, [1, 0] means only stock 1 has crossed. This variable will be passed
            internally for the 'and' trading logic. If you use 'or' logic, this variable does not affect the result.
        :return: (tuple) The current position, and the updated who_exit variable.
        """

        # Check open signals
        # Stock 2 over-valued, stock 1 under-valued, long the spread
        long_signal = (condi_probs[1] >= self.u_open_threshold and condi_probs[0] <= self.l_open_threshold)
        # Stock 1 over-valued, stock 2 under-valued, short the spread
        short_signal = (condi_probs[0] >= self.u_open_threshold and condi_probs[1] <= self.l_open_threshold)
        # Net result for the open signal. Change here for precedence within open signals
        open_signal = int(long_signal) - int(short_signal)

        # Check the exit signal and also update who_exit
        exit_signal, who_exits = self._exit_trigger(condi_probs, pre_condi_probs, exit_rule, who_exits,
                                                    exit_thresholds=(self.l_exit_threshold, self.u_exit_threshold))

        # Update the counters
        self._long_count += int(long_signal)
        self._short_count += int(short_signal)
        self._exit_count += exit_signal

        # Check if there is any valid open or close signal
        any_signal = bool(abs(open_signal) + abs(exit_signal))

        cur_pos = pre_pos  # Defaults to the previous position

        if any_signal:  # Update the signal when there is any valid signal
            # The current position is the net open signal when there is no exit signal, 0 When there is an exit signal
            # Change here for precedence of open and exit signals.
            cur_pos = open_signal * int((not bool(exit_signal)))

        if open_signal != 0:  # Reset who_exits when there is an open signal
            who_exits = pd.Series(np.zeros(2))

        return cur_pos, who_exits

    @staticmethod
    def _exit_trigger(condi_probs: pd.Series, pre_condi_probs: pd.Series, exit_rule: str,
                      who_exits: np.array, exit_thresholds: tuple) -> Tuple[int, np.array]:
        """
        Check if the exiting signal is triggered.

        'and' means both conditional probabilities need to cross 0.5.
        'or' means only one conditional probabilities crosses 0.5. The original approach in the paper is 'and'.
        You may change the thresholds to make the threshold a crossing band. Here, signal 1 means exit to no position,
        and 0 means not exit.

        :param condi_probs: (pd.Series) The current conditional probabilities for the stocks pair.
        :param pre_condi_probs: (pd.Series) The previous conditional probabilities for the stocks pair.
        :param exit_rule: (str) The logic for triggering an exit signal. Available choices are 'and', 'or'.
            They indicate whether both conditional probabilities need to cross 0.5.
        :param who_exits: (np.array) A binary (2, ) array indicating which stock has its conditional probability
            crossed with 0.5. For example, [1, 0] means only stock 1 has crossed. This variable will be passed
            internally for the 'and' trading logic. If you use 'or' logic, this variable does not affect the result.
        :param exit_thresholds: (np.array) Upper and lower bound of the threshold band.
        :return: (tuple) The exit signal in integer with 1 meaning exit, the updated who_exit variable.
        """

        exit_signal = 0  # Default exit signal is not exiting

        lower_exit_threshold = exit_thresholds[0]
        upper_exit_threshold = exit_thresholds[1]
        # Check if there are any crossings
        prob_u1_x_up = (pre_condi_probs[0] <= lower_exit_threshold
                        and condi_probs[0] >= upper_exit_threshold)  # Prob u1 crosses upward
        prob_u1_x_down = (pre_condi_probs[0] >= upper_exit_threshold
                          and condi_probs[0] <= lower_exit_threshold)  # Prob u1 crosses downward
        prob_u2_x_up = (pre_condi_probs[1] <= lower_exit_threshold
                        and condi_probs[1] >= upper_exit_threshold)  # Prob u2 crosses upward
        prob_u2_x_down = (pre_condi_probs[1] >= upper_exit_threshold
                          and condi_probs[1] <= lower_exit_threshold)  # Prob u2 crosses downward
        cross_events = [prob_u1_x_up, prob_u1_x_down, prob_u2_x_up, prob_u2_x_down]

        # Check at this step which variable crossed the band
        u1_cross = (prob_u1_x_up or prob_u1_x_down)
        u2_cross = (prob_u2_x_up or prob_u2_x_down)
        who_exits_now = np.array([int(u1_cross), int(u2_cross)])

        # Update for the who_exits variable.
        who_exits_bool = (who_exits_now + who_exits).astype(bool)  # Add the current info to the previous info
        who_exits = who_exits_bool.astype(int)  # Keep who_exits binary
        if np.all(who_exits_bool):  # Reset who_exits to [0, 0] when it is [1, 1]
            who_exits = pd.Series(np.zeros(2))

        # Under 'or' logic, if any crossing behavior happens, it exits
        if exit_rule == 'or':
            exit_signal = int(any(cross_events))
            return exit_signal, who_exits

        # Under 'and' logic, when both series are registered crossing, it exits
        if exit_rule == 'and' and np.all(who_exits_bool):
            exit_signal = 1
            return exit_signal, who_exits

        return exit_signal, who_exits
