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
    """
    Tests the Mudchanatongsuk OU model in the Stochastic Control Approach module.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Setup data and params.
        """

        np.random.seed(0)

        project_path = os.path.dirname(__file__)
        cls.path = project_path + '/test_data/gld_gdx_data.csv'
        data = pd.read_csv(cls.path)
        data = data.set_index('Date')

        cls.dataframe = data[['GLD', 'GDX']]

        cls.sc_mudchana = StochasticControlMudchanatongsuk()


    def test_fit(self):
        """
        Tests the fit method in the class.
        """

        sc_mudchana = StochasticControlMudchanatongsuk()

        sc_mudchana.fit(self.dataframe)

        spread_value = [2.0887923929288523, 2.1094166241052568, 2.142854916105869, 2.0874899841437258, 2.063640345328846,
         2.0746168560671223, 2.1097829292924457, 2.1073878043606307, 2.1186930608205428, 2.114695107155425,
         2.103095768137663, 2.1215705039862782, 2.124685352890708, 2.1157789172191848, 2.1269984320130346,
         2.120925503747776, 2.0763581413867684, 2.0595664254555057, 2.0523098321895366, 2.0797748079356957,
         2.1197252327440586, 2.131219793094243, 2.082081096325099, 2.0929293564196105, 2.1285514677643356,
         2.117750899307466, 2.0830138676604886, 2.106107155811972, 2.0496029865939436, 2.0050226960285182,
         1.9786263643353657, 1.9816431811190234, 1.9894204445041153, 1.9514558194020695, 1.9853812497671273,
         1.9846230785716679, 1.939022254160427, 1.9350347152497411, 1.9579985647080247, 1.996822168058443,
         1.963797288878891, 1.9847212944236645, 1.9674718306977712, 1.941258381912097, 1.971988526032174,
         1.9723574683305594, 1.973661841888858, 2.013718843187712, 2.0205219195979014, 2.0100021771087766,
         2.0004053530573787, 2.012571119972438, 2.0470873187820344, 2.0774352146534167, 2.053195021245263,
         2.0774606535295734, 2.06082098723021, 2.0742880142111084, 2.060417339214223, 2.0530928294396382,
         2.0890217923893957, 2.059036586082702, 2.035044438470959, 2.072153226841589, 2.069307015169184,
         2.0383888802559467, 2.043152450272762, 2.0504190679184715, 2.034227852538867, 2.0090985697450647,
         2.019908369440301, 2.009667357261374, 1.9797365648180438, 2.01008349909157, 2.0202245834522397]

        np.testing.assert_array_equal(sc_mudchana.spread, spread_value)
        self.assertAlmostEqual(sc_mudchana.sigma, 0.503695, delta=1e-4)
        self.assertAlmostEqual(sc_mudchana.mu, 0.114877, delta=1e-4)
        self.assertAlmostEqual(sc_mudchana.k, 3.99205, delta=1e-4)
        self.assertAlmostEqual(sc_mudchana.theta, 1.98816, delta=1e-4)
        self.assertAlmostEqual(sc_mudchana.eta, 0.404292, delta=1e-4)
        self.assertAlmostEqual(sc_mudchana.rho, 0.96202, delta=1e-4)


    def test_describe(self):
        """
        Tests the describe method in the class.
        """

        sc_mudchana = StochasticControlMudchanatongsuk()

        with self.assertRaises(Exception):
            sc_mudchana.describe()

        sc_mudchana.fit(self.dataframe)

        index = ['Ticker of first stock', 'Ticker of second stock',
                 'long-term mean of spread', 'rate of mean reversion of spread', 'standard deviation of spread', 'half-life of spread',
                 'Drift of stock B', 'standard deviation of stock B']

        data = ['GLD', 'GDX', 1.98816, 3.99205, 0.404292, 0.173632, 0.114877, 0.503695]

        pd.testing.assert_series_equal(pd.Series(index=index,data=data), sc_mudchana.describe(), check_exact=False, atol=1e-4)


    def test_optimal_weights(self):
        """
        Tests the optimal portfolio weights method in the class.
        """

        sc_mudchana = StochasticControlMudchanatongsuk()

        sc_mudchana.fit(self.dataframe)

        with self.assertRaises(Exception):
            sc_mudchana.optimal_portfolio_weights(self.dataframe, gamma = 10)

        weights = sc_mudchana.optimal_portfolio_weights(self.dataframe, gamma = -10)

        weights_value = [0.6807970425341383, 0.5788270189292528, 0.4237456386361687, 0.6450834372737371,
                         0.7318800196910422, 0.6725665016257416, 0.5137455982771261, 0.5117099817524071,
                         0.4536787173239446, 0.4587916338411348, 0.4947409799799004, 0.40935585749175935,
                         0.38656582779882165, 0.4119577215551326, 0.3576746243419241, 0.37205756995215905,
                         0.5364494177553357, 0.5912391763672612, 0.6087201255096889, 0.4933662450056616,
                         0.3322708500364613, 0.2807566340156271, 0.45706866084645936, 0.40803843763022246,
                         0.2683653938434968, 0.3009241934120373, 0.4202148927130246, 0.32903803274992643,
                         0.5246204647066117, 0.6749545941905897, 0.7588892323365495, 0.7385587466201069,
                         0.7020478426104525, 0.8238791404743722, 0.6979215228988004, 0.6917047438277323,
                         0.8365209290118862, 0.8401240368143309, 0.754516641656195, 0.6183636050618108,
                         0.7182449688122026, 0.6425285645177642, 0.6901494231876113, 0.7653424279618145,
                         0.6605525316290899, 0.652089070778512, 0.6409423355789974, 0.5118482097864904,
                         0.4857268994701272, 0.5119999415779317, 0.5350930783653811, 0.49397895578467543,
                         0.3888903487219631, 0.29800762140526377, 0.3642822152122685, 0.29285532211506354,
                         0.33686879455357427, 0.2971957603672995, 0.3329060663478341, 0.35032173033381425,
                         0.2523906371601354, 0.3301480022615564, 0.390645486517426, 0.29293440407814975,
                         0.29903173795128984, 0.3754395278348421, 0.36179056247411795, 0.34247802333072985,
                         0.3803355129092354, 0.43858981861753515, 0.4110154830924428, 0.43302883383533586,
                         0.4995100506045553, 0.42845097673506394, 0.40433684605306186]

        np.testing.assert_array_equal(weights, weights_value)
