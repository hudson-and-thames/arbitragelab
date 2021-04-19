# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Test functions for the Optimal Convergence models in the Stochastic Control Approach module.
"""
# pylint: disable=protected-access

import warnings
import unittest
import os

import numpy as np
import pandas as pd

from arbitragelab.stochastic_control_approach.optimal_convergence import OptimalConvergence

class TestOptimalConvergence(unittest.TestCase):
    """
    Test Optimal Convergence model in Stochastic Control Approach module
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up data and parameters
        """

        np.random.seed(0)

        project_path = os.path.dirname(__file__)

        # Setting up the first dataset
        path = project_path + '/test_data/gld_gdx_data.csv'
        data = pd.read_csv(path)
        data = data.set_index('Date')
        cls.dataframe = data[['GLD', 'GDX']]

        # Setting up the second dataset
        path = project_path + '/test_data/shell-rdp-close_USD.csv'
        data = pd.read_csv(path, index_col='Date').ffill()
        data.index = pd.to_datetime(data.index, format="%d/%m/%Y")
        cls.shell_rdp_data = data

    def test_unconstrained_continuous(self):

        oc = OptimalConvergence()
        oc.fit(self.dataframe)

        oc.b_squared = 0.3 ** 2
        oc.sigma_squared = 0.15 ** 2
        oc.beta = 0.5

        phi_1, phi_2, phi_m = oc.unconstrained_portfolio_weights_continuous(self.dataframe, mu_m=0.05, sigma_m=0.35, gamma=4, r=0.02)

        print(phi_m)