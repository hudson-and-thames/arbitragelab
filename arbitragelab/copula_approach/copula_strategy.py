# Copyright 2020, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Master module that graphs and fits copula
Created on Sun Nov  8 19:12:38 2020

@author: Hansen
"""
import copula_generate
import copula_calculation as ccalc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Copula_Strategy:
    """
    """

    def __init__(self, copula=None):
        # Copulas that uses theta as parameter
        self.theta_copula_names = ['Gumbel', 'Clayton', 'Frank',
                                   'Joe', 'N13', 'N14']
        # Copulas that uses cov (covariance matrix) as parameter
        self.cov_copula_names = ['Gaussian', 'Student']
        self.all_copula_names = self.theta_copula_names \
                                + self.cov_copula_names
        # Trading positions. 1: Long the spread. -1: Short the spread. 0: No position.
        self.position_kind = [1, -1, 0]
        # To be used for the test data set.
        self.copula = copula
        # Thresholds for opening trading positions.
        self.lower_threshold = 0.05
        self.upper_threshold = 0.95

    def fit_copula(self, data, copula_name, if_empirical_cdf=True, **kwargs):
        """
        Conduct a max likelihood estimation and information criterion.

        :param data: (numpy.ndarray) Scaled stock pair price time series data, shape = (n x 2).
        :param copula_name: (str) Type of copula to fit.
        :param if_empirical_cdf: (bool) Whether use empirical cumulative density function to fit data.
        :param kwargs: Input degree of freedom if using Student-t copula. e.g. nu=10.
        :return: (tuple) result_dict, copula, s1_cdf, s2_cdf
            WHERE
            result_dict: (dict) The name of copula and SIC, AIC, HQIC values.
            copula: (Copula) The fitted copula with parameters satisfying maximum likelihood.
            s1_cdf: (function) The cumulative density function for stock 1, using training data.
            s2_cdf: (function) The cumulative density function for stock 2, using training data.
        """
        nu = kwargs.get('nu', None)
        # Stocks scaled price time series.
        s1_series = data[:, 0]
        s2_series = data[:, 1]
        # Finding a inverse cumulative density distribution (quantile) for each stock price series.
        s1_cdf = ccalc.find_marginal_cdf(s1_series, empirical=if_empirical_cdf)
        s2_cdf = ccalc.find_marginal_cdf(s2_series, empirical=if_empirical_cdf)
        # Quantile data for each stock.
        u1_series = s1_cdf(s1_series)
        u2_series = s2_cdf(s2_series)
        # Get log-likelihood value and the copula with parameters fitted to training data.
        log_likelihood, copula = ccalc.log_ml(u1_series, u2_series,
                                              copula_name, nu)
        # Information criterion for evaluating model fitting performance.
        sic_value = ccalc.sic(log_likelihood, n=len(data[:, 0]))
        aic_value = ccalc.aic(log_likelihood, n=len(data[:, 0]))
        hqic_value = ccalc.hqic(log_likelihood, n=len(data[:, 0]))

        result_dict = {'Copula Name': copula_name,
                       'SIC': sic_value,
                       'AIC': aic_value,
                       'HQIC': hqic_value}

        # If when the strategy is initialized it does not come with a copula, then the strategy will use this fitted
        # copula for further analysis.
        if self.copula is None:
            self.copula = copula

        return result_dict, copula, s1_cdf, s2_cdf

    def graph_copula(self, copula_name, **kwargs):
        """
        Graph the sample from a given copula.

        Randomly sample
        :param copula_name: (str) Name of the copula to graph.
        :param kwargs: Parameters for the copula grapher.
            num: (int) Number of sample points to plot.
            theta: (float) The copula parameter indicating correlation.
            cov: (array_like) 2x2 array for covariance matrix, needed for Gaussian and Student-t copula.
            nu: (float) Degree of freedom if using Student-t copula.
        :return: ax: Plot axis
        """
        num = kwargs.get('num', 2000)  # Num of data points to plot
        theta = kwargs.get('theta', None)
        cov = kwargs.get('cov', None)
        nu = kwargs.get('nu', None)

        # Those copulas uses theta as parameter.
        if copula_name in self.theta_copula_names:
            # Generate data for plotting.
            my_copula = self._create_copula_by_name(copula_name=copula_name,
                                                    theta=theta)
            result = my_copula.generate_pairs(num=num)

            fig, ax = self._graph_copula(my_copula, kwargs,
                                         result=result,
                                         copula_name=copula_name)
            #fig.show()
        # Gaussian copula uses cov (covariance matrix) as parameter.
        elif copula_name == 'Gaussian':
            # Generate data for plotting
            my_copula = self._create_copula_by_name(copula_name=copula_name,
                                                    cov=cov)
            result = my_copula.generate_pairs(num=num)

            fig, ax = self._graph_copula(my_copula, kwargs,
                                         result=result,
                                         copula_name=copula_name)
            #fig.show()
        # Student-t copula uses cov (covariance matrix) and nu (degree of freedom) as parameters.
        elif copula_name == 'Student':
            # Generate data for plotting
            my_copula = self._create_copula_by_name(copula_name=copula_name,
                                                    cov=cov,
                                                    nu=nu)
            result = my_copula.generate_pairs(num=num)

            fig, ax = self._graph_copula(my_copula, kwargs,
                                         result=result,
                                         copula_name=copula_name)
            #fig.show()

    def _create_copula_by_name(self, **kwargs):
        """
        Create a copula given name and parameters.
        """
        # Use the Switch class to generate copula given name and parameter(s).
        Switch = copula_generate.Switcher()
        result = Switch.choose_copula(**kwargs)
        return result

    def _graph_copula(self, *args, **kwargs):
        """
        Helper function that creates figure and axis.

        :param args: Copula object.
        :param kwargs: Parameters for graphing.
        :return: (tuple) fig, axis
            WHERE
            fig: Plotting figure.
            ax: Plotting axis.
        """
        dpi = kwargs.get('dpi', 150)
        s = kwargs.get('s', 1)
        copula_name = kwargs.get('copula_name', None)
        my_copula = args[0]  # Copula object.
        result = kwargs.get('result', None)  # Data to be plotted.

        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111)
        ax.scatter(result[:, 0], result[:, 1], s=s)
        ax.set_aspect('equal', adjustable='box')  # Equal scale in x and y.
        if copula_name in self.theta_copula_names:
            ax.set_title(r'{} Copula, $\theta={}$'
                         .format(copula_name,
                                 my_copula.theta))
        elif copula_name == "Gaussian":
            ax.set_title(r'{} Copula, $\rho={}$'
                         .format(copula_name,
                                 my_copula.rho))
        elif copula_name == "Student":
            ax.set_title(r'Student-t Copula, $\rho={}$, $\nu={}$'
                         .format(my_copula.rho,
                                 my_copula.nu))

        return fig, ax

    def analyse_price_series(self, s1_series, s2_series, cdf1, cdf2, **kwargs):  # Change to analyze later
        """
        Generates positions given price series of two stocks.

        :param s1_series:
        :param s2_series:
        :param cdf1:
        :param cdf2:
        :param kwargs:
        :return:
        """
        # Update the trading thresholds.
        self.upper_threshold = kwargs.get('upper_threshold', 0.95)
        self.lower_threshold = kwargs.get('lower_threshold', 0.05)

        # Cumulative conditional probability series for 2 stocks.
        prob_series = self.series_condi_prob(s1_series, s2_series,
                                             cdf1, cdf2)
        probs_1 = prob_series[:, 0]
        probs_2 = prob_series[:, 1]

        num_of_data = len(s1_series)

        # Calculate the positions given probabilities.
        positions = [self.position_kind[2]] * num_of_data
        for idx in range(1, num_of_data):
            positions[idx] = \
                self.get_next_position(prob_u1=probs_1[idx],
                                       prob_u2=probs_2[idx],
                                       prev_prob_u1=probs_1[idx - 1],
                                       prev_prob_u2=probs_2[idx - 1],
                                       current_pos=positions[idx - 1])
        return positions

    def get_next_position(self, prob_u1, prob_u2,
                          prev_prob_u1, prev_prob_u2, current_pos):
        """
        Get the next trading position.
        """
        signal, to_exit = self._generate_trading_signal(prob_u1,
                                                        prob_u2,
                                                        prev_prob_u1,
                                                        prev_prob_u2,
                                                        current_pos)

        # If currently have no position.
        if current_pos == self.position_kind[2]:
            # If the signal is long, then go long.
            # If the signal is short, then go short.
            if signal is not None:
                return signal
        # If the signal is to exit, and currently we have a position,
        # then hold no position.
        elif to_exit is True:
            return self.position_kind[2]
        # By default, hold the current position.
        return current_pos

    def _generate_trading_signal(self, prob_u1, prob_u2,
                                 prev_prob_u1, prev_prob_u2,
                                 current_pos):
        """
        Trading signal given pairs price data.
        """
        # Default values.
        open_type = None
        to_exit = False

        # If currently hold no position, then check if can open position
        if current_pos == self.position_kind[2]:
            open_type = self._check_open_position(prob_u1,
                                                  prob_u2,
                                                  self.upper_threshold,
                                                  self.lower_threshold)
            return open_type, to_exit
        # If have a position already, check if any probabilities cross.
        # If any probabilities cross, exit position.
        else:
            to_exit = self._check_if_cross(prob_u1, prob_u2,
                                           prev_prob_u1, prev_prob_u2)
            return open_type, to_exit

        return open_type, to_exit

    def _condi_prob(self, s1, s2, cdf1, cdf2):
        """
        Conditional accumulative probability for pair's price data and copula.
        """
        u1 = cdf1(s1)
        u2 = cdf2(s2)
        # P(U1<=u1 | U2=u2)
        prob_u1_given_u2 = self.copula.condi_cdf(u1, u2)
        # P(U2<=u2 | U1=u1)
        prob_u2_given_u1 = self.copula.condi_cdf(u2, u1)

        return prob_u1_given_u2, prob_u2_given_u1

    def series_condi_prob(self, s1_series, s2_series, cdf1, cdf2):
        """
        Calculate the condition probabilities for two price series
        """
        prob_series = np.zeros((len(s1_series), 2))
        for row_idx, each_row in enumerate(prob_series):
            each_row[0], each_row[1] = \
                self._condi_prob(s1_series[row_idx],
                                 s2_series[row_idx],
                                 cdf1, cdf2)

        return prob_series

    def _check_open_position(self, prob_u1, prob_u2,
                             upper_threshold=0.95,
                             lower_threshold=0.05):
        """
        Check if open position from price quantile data given threshold
        """
        position = self.position_kind  # i.e., [long, short, no position]
        pairs_action = None  # Default to None

        if (prob_u1 <= lower_threshold  # u1 undervalued
                and prob_u2 >= upper_threshold):  # u2 overvalued
            pairs_action = position[0]  # Go long the spread
        elif (prob_u1 >= upper_threshold  # u1 overvalued
              and prob_u2 <= lower_threshold):  # u2 undervalued
            pairs_action = position[1]  # Go short the spread

        return pairs_action

    def _check_if_cross(self, prob_u1, prob_u2, prev_prob_u1, prev_prob_u2,
                        upper_exit_threshold=0.5,
                        lower_exit_threshold=0.5):
        """
        Check if to exit position from conditional probability.
        
        When any one of the conditional probability crosses the boundary, we
        attemp to close the position.
        """
        prob_u1_x_up = (prev_prob_u1 < lower_exit_threshold
                        and prob_u1 > upper_exit_threshold)  # Prob u1 crosses upward
        prob_u1_x_down = (prev_prob_u1 > upper_exit_threshold
                          and prob_u1 < lower_exit_threshold)  # Prob u1 crosses downward
        prob_u2_x_up = (prev_prob_u2 < lower_exit_threshold
                        and prob_u2 > upper_exit_threshold)  # Prob u2 crosses upward
        prob_u2_x_down = (prev_prob_u2 > upper_exit_threshold
                          and prob_u2 < lower_exit_threshold)  # Prob u2 crosses downward
        cross_events = [prob_u1_x_up, prob_u1_x_down,
                        prob_u2_x_up, prob_u2_x_down]
        if_cross = any(cross_events)

        return if_cross
