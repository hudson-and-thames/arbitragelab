# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

# pylint: disable=missing-module-docstring, invalid-name
import warnings
from scipy.optimize import root_scalar, fsolve, minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from arbitragelab.optimal_mean_reversion.ou_model import OrnsteinUhlenbeck
from arbitragelab.util import devadarsh


class OUModelOptimalThreshold:
    def __init__(self):
        pass


    # Users can choose two different ways to initialize the OU model.
    @classmethod
    def construct_ou_model_from_given_parameters(cls):
        """
        Construct the OU model from given parameters.
        """
        pass


    @classmethod
    def fit_ou_model_to_data(cls):
        """
        Fits the OU model to given data.
        """
        pass


    def _transform_to_dimensionless_system(self):
        """
        Transform Xt to Yt based on the OU model parameters.
        """
        pass


    def _back_transform_from_dimensionless_system(self):
        """
        Transform Yt back to Xt based on the OU model parameters.
        """
        pass


    def _solve_gamma_equation(self):
        """
        Sovle equations (20) or (23) in the second paper using SymPy
        """
        pass


    # Users will need to choose which assumption to use,  one-side boundary or two-sided boundary,
    # to calculate the following metrics.
    def get_expected_trading_cycle(self):
        pass


    def get_expected_return(self):
        pass


    def get_variance(self):
        pass


    def get_threshold_by_maximize_expected_return(self):
        """
        Calculate equation (14) in the first paper to get the result.
        But the equation may need to be rewritten under the assumption that
        the mean of the OU process may be non-zero.
        """
        pass


    def get_threshold_by_maximize_sharpe_ratio(self):
        """
        Optimize equation (16) in the first paper using Scipy to get the result.
        But the equation may need to be rewritten under the assumption that
        the mean of the OU process may be non-zero.
        """
        pass


    def get_threshold_by_conventional_optimal_rule(self):
        """
        Sovle equations (20) in the second to get the result.
        """
        pass


    def get_threshold_by_new_optimal_rule(self):
        """
        Sovle equations (23) in the second to get the result.
        """
        pass


    def plot_threshold_vs_cost(self):
        pass


    def plot_expected_retrun_vs_cost(self):
        pass


    def plot_variance_vs_cost(self):
        pass


    def plot_sharpe_ratio_vs_cost(self):
        pass






