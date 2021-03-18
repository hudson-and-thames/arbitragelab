# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Test functions for the Mudchanatongsuk OU model in the Stochastic Control Approach module.
"""

import unittest
import os
import numpy as np
import pandas as pd

from arbitragelab.stochastic_control_approach.ou_model_mudchanatongsuk import StochasticControlMudchanatongsuk


class TestOUModelMudchanatongsuk(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        np.random.seed(0)

        project_path = os.path.dirname(__file__)
        cls.path = project_path + '/test_data/gld_gdx_data.csv'
        data = pd.read_csv(cls.path)
        data = data.set_index('Date')

        cls.dataframe = data[['GLD', 'GDX']]

        cls.sc = StochasticControlMudchanatongsuk()

    def test_fit(self):
        sc = StochasticControlMudchanatongsuk()

        sc.fit(self.dataframe)

        np.testing.assert_array_equal(sc.spread, np.zeros(len(sc.time_array)))
        self.assertAlmostEqual(sc.sigma, 0, delta=1e-4)
        self.assertAlmostEqual(sc.mu, 0, delta=1e-4)
        self.assertAlmostEqual(sc.k, 0, delta=1e-4)
        self.assertAlmostEqual(sc.theta, 0, delta=1e-4)
        self.assertAlmostEqual(sc.eta, 0, delta=1e-4)
        self.assertAlmostEqual(sc.rho, 0, delta=1e-4)


    def test_optimal_weights(self):
        sc = StochasticControlMudchanatongsuk()

        sc.fit(self.dataframe)

        weights = sc.optimal_portfolio_weights(self.dataframe, gamma = 10)

        np.testing.assert_array_equal(weights, np.zeros(len(sc.time_array)))
