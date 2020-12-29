# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Master module that uses copula for trading strategy.

This is a legacy module.
"""

# pylint: disable = invalid-name
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

import arbitragelab.copula_approach.copula_generate as cg
import arbitragelab.copula_approach.copula_calculation as ccalc


class CopulaStrategy:
    """
    Analyze a pair of stock prices using copulas.

    We use the convention that the spread is defined as stock 1 in relation to stock 2.
    This class provides the following functionalities:

        1. Maximum likelihood fitting to training data. User provides the name and necessary parameters.
        2. Use a given copula to generate trading positions based on test data. By default it uses
           the fitted copula generated in functionality 1.
        3. Scatter plot of a given copula about its probability density.

    """

    def __init__(self, copula: cg.Copula = None, position_kind: list = None,
                 default_lower_threshold: float = 0.05, default_upper_threshold: float = 0.95):
        """
        Initiate a CopulaStrategy class.

        One can choose to initiate with no arguments, or to initiate with a given copula as the system's
        Copula.

        :param copula: (Copula) Optional. A copula object that the class will use for all analysis. If there
            is no input then fit_copula method will create one when called.
        :param position_kind: (list) Optional. The integers to represent long/short/no positions for the pair's trading
            framework. By default it uses [1, -1, 0].
        :param default_lower_threshold: (float) Optional. The default lower threshold for opening a position for
            trading signal generation. Defaults to 0.05.
        :param default_upper_threshold: (float) Optional. The default upper threshold for opening a position for
            trading signal generation. Defaults to 0.95.
        """

        # Copulas that uses theta as parameter
        self.theta_copula_names = ['Gumbel', 'Clayton', 'Frank',
                                   'Joe', 'N13', 'N14']
        # Copulas that uses cov (covariance matrix) as parameter
        self.cov_copula_names = ['Gaussian', 'Student']
        self.all_copula_names = self.theta_copula_names + self.cov_copula_names

        # Default trading positions.
        # 1: Long the spread.
        # -1: Short the spread.
        # 0: No position.
        if position_kind is None:
            self.position_kind = [1, -1, 0]
        else:
            self.position_kind = position_kind
        # To be used for the test data set.
        self.copula = copula

        # Default thresholds for opening trading positions.
        self.lower_threshold = default_lower_threshold
        self.upper_threshold = default_upper_threshold

    def fit_copula(self, s1_series: np.array, s2_series: np.ndarray, copula_name: str,
                   if_empirical_cdf: bool = True, if_renew: bool = True, nu_tol: float = 0.05) -> tuple:
        """
        Conduct a pseudo max likelihood estimation and information criterion.

        Note: s1_series and s2_series need to be pre-processed. In general, raw price data is depreciated.
            One may use log return or cumulative log return. CopulaStrategy class provides a method to
            calculate cumulative log return.

        If fitting a Student-t copula, it also includes a max likelihood fit for nu using COBYLA method from
        scipy.optimize.minimize. nu's fit range is [1, 15]. When the user wishes to use nu > 15, please delegate to
        Gaussian copula instead. This step is relatively slow.

        The output returns:
            - result_dict: (dict) The name of the copula and its SIC, AIC, HQIC values;
            - copula: (Copula) The fitted copula with parameters satisfying maximum likelihood;
            - s1_cdf: (func) The cumulative density function for stock 1, using training data;
            - s2_cdf: (func) The cumulative density function for stock 2, using training data.

        :param s1_series: (np.array) 1D stock time series data in desired form.
        :param s2_series: (np.array) 1D stock time series data in desired form.
        :param copula_name: (str) Type of copula to fit.
        :param if_empirical_cdf: (bool) Whether use empirical cumulative density function to fit data.
        :param if_renew: (bool) Whether use the fitted copula to replace the copula in CopulaStrategy.
        :param nu_tol: (float) Optional. The final accuracy for finding nu for Student-t copula. Defaults to 0.05.
        :return: (dict, Copula, func, func)
            The name of the copula and its SIC, AIC, HQIC values;
            The fitted copula with parameters satisfying maximum likelihood;
            The cumulative density function for stock 1, using training data;
            The cumulative density function for stock 2, using training data.
        """

        num_of_instances = len(s1_series)  # Number of instances.

        # Finding an inverse cumulative density distribution (quantile) for each stock price series.
        s1_cdf = ccalc.find_marginal_cdf(s1_series, empirical=if_empirical_cdf)
        s2_cdf = ccalc.find_marginal_cdf(s2_series, empirical=if_empirical_cdf)

        # Quantile data for each stock w.r.t. their cumulative log return.
        u1_series = s1_cdf(s1_series)
        u2_series = s2_cdf(s2_series)

        # Get log-likelihood value and the copula with parameters fitted to training data.
        if copula_name == 'Student':
            fitted_nu = ccalc.fit_nu_for_t_copula(u1_series, u2_series, nu_tol)
            log_likelihood, copula = ccalc.log_ml(u1_series, u2_series,
                                                  copula_name, fitted_nu)
        else:
            log_likelihood, copula = ccalc.log_ml(u1_series, u2_series, copula_name)

        # Information criterion for evaluating model fitting performance.
        sic_value = ccalc.sic(log_likelihood, n=num_of_instances)
        aic_value = ccalc.aic(log_likelihood, n=num_of_instances)
        hqic_value = ccalc.hqic(log_likelihood, n=num_of_instances)

        result_dict = {'Copula Name': copula_name,
                       'SIC': sic_value,
                       'AIC': aic_value,
                       'HQIC': hqic_value}

        # If when the strategy is initialized it does not come with a copula, or the user intends to renew the
        # system's copula, then the strategy will use this fitted copula for further analysis.
        if if_renew or self.copula is None:
            self.copula = copula

        return result_dict, copula, s1_cdf, s2_cdf

    def ic_test(self, s1_test: np.array, s2_test: np.array,
                cdf1: Callable[[float], float], cdf2: Callable[[float], float],
                copula: cg.Copula = None) -> dict:
        """
        Run SIC, AIC, and HQIC of the fitted copula given data.

        This method only works if CopulaStrategy has fitted a copula given training data.

        Note: s1_series and s2_series need to be pre-processed. In general, raw price data is depreciated.
            One may use log return or cumulative log return. CopulaStrategy class provides a method to
            calculate cumulative log return.

        :param s1_test: (np.array) 1D stock time series data in desired form.
        :param s2_test: (np.array) 1D stock time series data in desired form.
        :param cdf1: (func) Cumulative density function trained, for the security in s1_test.
        :param cdf2: (func) Cumulative density function trained, for the security in s2_test.
        :param copula: (Copula) The copula to be evaluated. By default it uses the system's copula.
        :return: (dict) Result of SIC, AIC and HQIC.
        """

        num_of_instances = len(s1_test)
        if copula is None:
            copula = self.copula
        # Get the copula's name.
        copula_name = copula.__class__.__name__
        # if copula_name not in self.all_copula_names:
        #     raise ValueError('CopulaStrategy does not have a currently defined copula.')
        # Get nu (degree of freedom) if it is a Student-t copula.
        if copula_name == 'Student':
            nu = copula.nu
        else:
            nu = None

        # Quantile data for each stock w.r.t. their cumulative log return.
        u1_series = cdf1(s1_test)
        u2_series = cdf2(s2_test)

        # Get log-likelihood value and the copula with parameters fitted to training data.
        log_likelihood, _ = ccalc.log_ml(u1_series, u2_series,
                                         copula_name, nu)

        # Information criterion for evaluating model fitting performance.
        sic_value = ccalc.sic(log_likelihood, n=num_of_instances)
        aic_value = ccalc.aic(log_likelihood, n=num_of_instances)
        hqic_value = ccalc.hqic(log_likelihood, n=num_of_instances)

        result_dict = {'Copula Name': copula_name,
                       'SIC': sic_value,
                       'AIC': aic_value,
                       'HQIC': hqic_value}

        return result_dict

    def graph_copula(self, copula_name: str, ax: plt.axes = None, **kwargs: dict) -> plt.axes:
        """
        Graph the sample from a given copula by its parameters. Returns axis.

        Randomly sample using copula density. User may further specify axis parameters for plotting in kwargs.

        kwargs include
            - num: (int) Number of sample points to plot.
            - theta: (float) The copula parameter indicating correlation.
            - cov: (np.array) 2x2 array for covariance matrix, needed for Gaussian and Student-t copula.
            - nu: (float) Degree of freedom if using Student-t copula.
            - other plt.axes specific kwargs.

        :param copula_name: (str) Name of the copula to graph.
        :param ax: (plt.axes) Plotting axes.
        :param kwargs: Parameters for the copula and the plot axes.
        :return: (plt.axes) Plotting axes.
        """

        num = kwargs.get('num', 2000)  # Num of data points to plot.
        # Copula specific parameters.
        theta = kwargs.get('theta', None)
        cov = kwargs.get('cov', None)
        nu = kwargs.get('nu', None)
        # Separate plotting kwargs from copula parameters.
        copula_params = ['theta', 'cov', 'nu', 'num']
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in copula_params}

        # Those copulas use theta as parameter.
        if copula_name in self.theta_copula_names:
            # Generate data for plotting.
            my_copula = self._create_copula_by_name(copula_name=copula_name,
                                                    theta=theta)
            result = my_copula.generate_pairs(num=num)
            # Modify axes.
            self._graph_copula(my_copula=my_copula, ax=ax,
                               result=result,
                               copula_name=copula_name,
                               **plot_kwargs)

        # Gaussian copula uses cov (covariance matrix) as parameter.
        if copula_name == 'Gaussian':
            # Generate data for plotting.
            my_copula = self._create_copula_by_name(copula_name=copula_name,
                                                    cov=cov)
            result = my_copula.generate_pairs(num=num)
            # Modify axes.
            self._graph_copula(my_copula=my_copula, ax=ax,
                               result=result,
                               copula_name=copula_name,
                               **plot_kwargs)

        # Student-t copula uses cov (covariance matrix) and nu (degree of freedom) as parameters.
        if copula_name == 'Student':
            # Generate data for plotting.
            my_copula = self._create_copula_by_name(copula_name=copula_name,
                                                    cov=cov,
                                                    nu=nu)
            result = my_copula.generate_pairs(num=num)
            # Modify axes
            self._graph_copula(my_copula=my_copula,
                               ax=ax,
                               result=result,
                               copula_name=copula_name,
                               **plot_kwargs)

        return ax

    def analyze_time_series(self, s1_series: np.array, s2_series: np.array,
                            cdf1: Callable[[float], float], cdf2: Callable[[float], float],
                            upper_threshold: float = None, lower_threshold: float = None,
                            start_position: int = None) -> np.array:
        """
        Generate positions given time series of two stocks.

        Note: s1_series and s2_series need to be pre-processed. In general, raw price data is depreciated.
            One may use log return or cumulative log return. CopulaStrategy class provides a method to
            calculate cumulative log return.

        :param s1_series: (np.array) 1D time series from stock 1.
        :param s2_series: (np.array) 1D time series from stock 2.
        :param cdf1: (func) Marginal C.D.F. for stock 1.
        :param cdf2: (func) Marginal C.D.F. for stock 2.
        :param start_position: (int) Optional. Starting position. Defaults to no position.
        :param upper_threshold: (float) Upper threshold. Class defaults to 0.95.
        :param lower_threshold: (float) Lower threshold. Class defaults to 0.05.
        :return: (np.array) The suggested positions for the given price data.
        """

        # Update the trading thresholds if there are inputs. Otherwise use the default.
        if upper_threshold is not None:
            self.upper_threshold = upper_threshold
        if lower_threshold is not None:
            self.lower_threshold = lower_threshold
        # Default starting position is no position.
        if start_position is None:
            start_position = self.position_kind[2]

        # Conditional probability series for 2 stocks.
        prob_series = self.series_condi_prob(s1_series, s2_series,
                                             cdf1, cdf2)
        probs_1 = prob_series[:, 0]
        probs_2 = prob_series[:, 1]

        num_of_data = len(s1_series)

        # Calculate the positions given probabilities.
        positions = np.zeros(num_of_data)
        positions[0] = start_position  # Set up the starting position.

        # Generate position data given its immediate previous probabilities, current
        # probabilities and position.
        for idx in range(1, num_of_data):
            positions[idx] = \
                self.get_next_position(prob_u1=probs_1[idx],
                                       prob_u2=probs_2[idx],
                                       prev_prob_u1=probs_1[idx - 1],
                                       prev_prob_u2=probs_2[idx - 1],
                                       current_pos=positions[idx - 1])

        return positions

    def get_next_position(self, prob_u1: float, prob_u2: float,
                          prev_prob_u1: float, prev_prob_u2: float, current_pos: int) -> int:
        """
        Get the next trading position.

        Use the probability pair associated with the current price, the probability pair associated
        with the most recent price, and the most recent trading position, to get the current trading
        position.

        Logic:
            If currently have no position, and there is an open signal, then change position into what the
            open signal indicates (long or short).
            If currently have a position, and exit signal says to exit, then exit the current position.

        :param prob_u1: (float) Current marginal C.D.F. for stock 1, given stock 2.
        :param prob_u2: (float) Current marginal C.D.F. for stock 2, given stock 1.
        :param prev_prob_u1: (float) Previous marginal C.D.F. for stock 1, given stock 2.
        :param prev_prob_u2: (float) Previous marginal C.D.F. for stock 2, given stock 1.
        :param current_pos: (int) Most recent trading position.
        :return: (int) Suggested trading position after assessment.
        """

        open_signal, exit_signal = self._generate_trading_signal(prob_u1,
                                                                 prob_u2,
                                                                 prev_prob_u1,
                                                                 prev_prob_u2,
                                                                 current_pos)

        # If currently position is not open.
        if current_pos == self.position_kind[2]:
            # If the signal is long, then go long.
            # If the signal is short, then go short.
            if open_signal is not None:
                return open_signal
        # If the signal is to exit, and currently we have a position,
        # then hold no position.
        elif exit_signal is True:
            return self.position_kind[2]

        # By default, hold the current position.
        return current_pos

    @staticmethod
    def cum_log_return(price_series: np.array, start: float = None) -> np.array:
        """
        Convert a price time series to cumulative log return.

        clr[i] = log(S[i]/S[0]) = log(S[i]) - log(S[0]).

        :param price_series: (np.array) 1D price time series.
        :param start: (float) Initial price. Default to the starting element of price_series.
        :return: (np.array) 1D cumulative log return series.
        """

        if start is None:
            start = price_series[0]
        log_start = np.log(start)

        # Natural log of price series.
        log_prices = np.log(price_series)
        # Calculate cumulative log return.
        clr = np.array([log_price_now - log_start for log_price_now in log_prices])

        return clr

    def series_condi_prob(self, s1_series: np.array, s2_series: np.array,
                          cdf1: Callable[[float], float], cdf2: Callable[[float], float]) -> np.array:
        """
        Calculate the conditional probabilities for two time series.

        i.e., P(U1 <= u1 | U2 = u2) and P(U2 <= u2 | U1 = u1)

        Note: s1_series and s2_series need to be pre-processed. In general, raw price data is depreciated.
            One may use log return or cumulative log return. CopulaStrategy class provides a method to
            calculate cumulative log return.

        :param s1_series: (np.array) 1D time series from stock 1.
        :param s2_series: (np.array) 1D time series from stock 2.
        :param cdf1: (func) Marginal C.D.F. for stock 1.
        :param cdf2: (func) Marginal C.D.F. for stock 2.
        :return: (np.array) (N, 2) shaped array storing the conditional C.D.F. pair for the stock pair.
        """

        num_of_instances = len(s1_series)

        prob_series = np.zeros((num_of_instances, 2))
        # For pair, calculate and store its conditional C.D.F. pair.
        for row_idx, each_row in enumerate(prob_series):
            each_row[0], each_row[1] = \
                self._condi_prob(s1_series[row_idx],
                                 s2_series[row_idx],
                                 cdf1, cdf2)

        return prob_series

    @staticmethod
    def _create_copula_by_name(**kwargs: dict) -> object:
        """
        Construct a copula given name (str) and parameters.

        Wrapper function around a switch emulator.

        :param kwargs: (dict) Input arguments.
        :return: (Copula) Constructed copula.
        """

        # Use the Switch class to generate copula given name and parameter(s).
        Switch = cg.Switcher()
        result = Switch.choose_copula(**kwargs)

        return result

    def _graph_copula(self, my_copula: cg.Copula, copula_name: str, result: np.array,
                      ax: plt.axes = None, **kwargs: dict) -> tuple:
        """
        Create figure and axis.

        :param my_copula: (Copula) Copula object to be graphed.
        :param copula_name: (str) Name of the copula.
        :param result: (np.array) Copula sample data to be plotted. Shape=(num of instances, 2).
        :param ax: (plt.ax) Plotting axex to be modified.
        :param kwargs: (dict) Kwargs specifying plotting features.
        :return: (plt.fig, plt.axis) Plotting fig and axex.
        """

        # Unpacking plotting kwargs.
        plot_kwargs = kwargs

        ax = ax or plt.gca()
        ax.scatter(result[:, 0], result[:, 1], **plot_kwargs)
        ax.set_aspect('equal', adjustable='box')  # Equal scale in x and y.
        if copula_name in self.theta_copula_names:
            ax.set_title(r'{} Copula, $\theta={:.3f}$'.format(copula_name, my_copula.theta))
        if copula_name == "Gaussian":
            ax.set_title(r'{} Copula, $\rho={:.3f}$'.format(copula_name, my_copula.rho))
        if copula_name == "Student":
            ax.set_title(r'Student-t Copula, $\rho={:.3f}$, $\nu={}$'.format(my_copula.rho, my_copula.nu))

    def _generate_trading_signal(self, prob_u1: float, prob_u2: float,
                                 prev_prob_u1: float, prev_prob_u2: float,
                                 current_pos: int) -> tuple:
        """
        Generate trading signal given pairs probabilities and positions.

        There are two signals:
            open_type: -1, 1, or None, indicating the type of position one should open. None means do not open.
            to_exit: bool, indicating if one should exit the current position.

        :param prob_u1: (float) Current marginal C.D.F. for stock 1, given stock 2.
        :param prob_u2: (float) Current marginal C.D.F. for stock 2, given stock 1.
        :param prev_prob_u1: (float) Previous marginal C.D.F. for stock 1, given stock 2.
        :param prev_prob_u2: (float) Previous marginal C.D.F. for stock 2, given stock 1.
        :param current_pos: (int) Most recent trading position.
        :return: (tuple) Suggested trading signal after assessment.
        """

        # Default values.
        open_type = None
        to_exit = False

        # If currently hold no position, then check if can open position
        if current_pos == self.position_kind[2]:  # pylint: disable = no-else-return
            open_type = self._check_open_position(prob_u1,
                                                  prob_u2)
            return open_type, to_exit
        # If have a position already, check if any probabilities cross.
        # If any probabilities cross, exit position.
        else:
            to_exit = self._check_if_cross(prob_u1, prob_u2,
                                           prev_prob_u1, prev_prob_u2)

            return open_type, to_exit

    def _condi_prob(self, s1: np.array, s2: np.array,
                    cdf1: Callable[[float], float], cdf2: Callable[[float], float]):
        """
        Conditional accumulative probability for pair's price data and copula.

        i.e. tuple(P(U1<=u1 | U2=u2), P(U2<=u2 | U1=u1)) for probability u1, u2 in association with s1, s2.

        Note: The calculation assumes copula being Archimedean, and thus the conditional C.D.F. is symmetric.
            s1 and s2 need to be pre-processed. In general, raw price data is depreciated.
            One may use log return or cumulative log return. CopulaStrategy class provides a method to
            calculate cumulative log return.

        :param s1: (np.array) Single data from stock 1.
        :param s2: (np.array) Single data from stock 2.
        :param cdf1: (func) Marginal C.D.F. for stock 1.
        :param cdf2: (func) Marginal C.D.F. for stock 2.
        :return: (tuple) P(U1<=u1 | U2=u2), P(U2<=u2 | U1=u1).
        """

        u1 = cdf1(s1)
        u2 = cdf2(s2)
        # P(U1<=u1 | U2=u2)
        prob_u1_given_u2 = self.copula.condi_cdf(u1, u2)
        # P(U2<=u2 | U1=u1)
        prob_u2_given_u1 = self.copula.condi_cdf(u2, u1)

        return prob_u1_given_u2, prob_u2_given_u1

    def _check_open_position(self, prob_u1: float, prob_u2: float) -> int:
        """
        Check if open position from price quantile data given threshold.

        Looks at probabilities, then decide if one should open a trade position.

        Note: This function has no information about the current trading position. Hence its decision is solely
        based on the current probabilities. Higher level class methods will assemble this information later for
        suggesting a full trading position.

        :param prob_u1: (float) Current marginal C.D.F. for stock 1, given stock 2.
        :param prob_u2: (float) Current marginal C.D.F. for stock 2, given stock 1.
        :return: (int) Suggested opening position after assessment.
        """

        # Use the class attribute for thresholds.
        lower_threshold = self.lower_threshold
        upper_threshold = self.upper_threshold
        position = self.position_kind  # i.e., [long, short, no position]
        pairs_action = None  # Default to None

        if (prob_u1 <= lower_threshold  # u1 undervalued
                and prob_u2 >= upper_threshold):  # u2 overvalued
            pairs_action = position[0]  # Go long the spread
        elif (prob_u1 >= upper_threshold  # u1 overvalued
              and prob_u2 <= lower_threshold):  # u2 undervalued
            pairs_action = position[1]  # Go short the spread

        return pairs_action

    @staticmethod
    def _check_if_cross(prob_u1: float, prob_u2: float, prev_prob_u1: float, prev_prob_u2: float,
                        upper_exit_threshold: float = 0.5,
                        lower_exit_threshold: float = 0.5):
        """
        Check if to exit position from conditional probability.

        When any one of the conditional probability crosses the threshold band, we
        attempt to close the position.

        :param prob_u1: (float) Current marginal C.D.F. for stock 1, given stock 2.
        :param prob_u2: (float) Current marginal C.D.F. for stock 2, given stock 1.
        :param prev_prob_u1: (float) Previous marginal C.D.F. for stock 1, given stock 2.
        :param prev_prob_u2: (float) Previous marginal C.D.F. for stock 2, given stock 1.
        :param upper_exit_threshold: (float) Upper bound of the threshold band. Defaults to 0.5.
        :param lower_exit_threshold: (float) Lower bound of the threshold band. Defaults to 0.5.
        :return: (int) Suggested trading position after assessment.
        """

        prob_u1_x_up = (prev_prob_u1 <= lower_exit_threshold
                        and prob_u1 >= upper_exit_threshold)  # Prob u1 crosses upward
        prob_u1_x_down = (prev_prob_u1 >= upper_exit_threshold
                          and prob_u1 <= lower_exit_threshold)  # Prob u1 crosses downward
        prob_u2_x_up = (prev_prob_u2 <= lower_exit_threshold
                        and prob_u2 >= upper_exit_threshold)  # Prob u2 crosses upward
        prob_u2_x_down = (prev_prob_u2 >= upper_exit_threshold
                          and prob_u2 <= lower_exit_threshold)  # Prob u2 crosses downward
        cross_events = [prob_u1_x_up, prob_u1_x_down,
                        prob_u2_x_up, prob_u2_x_down]
        if_cross = any(cross_events)

        return if_cross
