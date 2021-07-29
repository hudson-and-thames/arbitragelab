# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Tests functions from Regime Switching Arbitrage Rule module.
"""

# pylint: disable=missing-module-docstring, invalid-name
import unittest
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

from arbitragelab.time_series_approach.regime_switching_arbitrage_rule import RegimeSwitchingArbitrageRule


class TestRegimeSwitchingArbitrageRule(unittest.TestCase):
    """r
    Tests the class of Regime Switching Arbitrage Rule module
    """

    def setUp(self):
        """
        Sets the file path for the data and testing variables.
        """

        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/CL=F_NG=F_data.csv'
        data = pd.read_csv(self.path)
        data = data.set_index('Date')
        Ratt = data["CL=F"]/data["NG=F"]

        self.Ratts = Ratt.values, Ratt, pd.DataFrame(Ratt)
        self.Ratt_inv = data["NG=F"]/data["CL=F"]

    def test_signal(self):
        """
        Tests the functions for getting signal.
        """

        # Creating an object of class
        test = RegimeSwitchingArbitrageRule(delta = 1.5, rho = 0.6)

        # Setting window size
        window_size = 60

        # Getting current signal and testing the results
        for i in [0, 1, 2]:
            signal = test.get_signal(self.Ratts[i][-window_size:], switching_variance = False, silence_warnings = True)
            self.assertEqual(signal, 0)

        signal = test.get_signal(self.Ratts[0][-window_size:], switching_variance = False, silence_warnings = False)
        self.assertEqual(signal, 0)

        signal = test.get_signal(self.Ratts[0][-window_size:], switching_variance = True, silence_warnings = True)
        self.assertTrue(math.isnan(signal))

        signal = test.get_signal(self.Ratts[0][-window_size * 8:], switching_variance = True, silence_warnings = True)
        self.assertEqual(signal, 0)

        # Getting signals on a rolling basis
        signals_1 = test.get_signals(self.Ratts[0], window_size, switching_variance = False, silence_warnings = True)
        signals_2 = test.get_signals(pd.DataFrame(self.Ratts[0]), window_size//2, switching_variance = False, silence_warnings = True)
        signals_3 = test.get_signals(pd.DataFrame(self.Ratts[0]), window_size//4, switching_variance = False, silence_warnings = True)
        signals_4 = test.get_signals(self.Ratt_inv, window_size, switching_variance = False, silence_warnings = True)

        # Testing the result
        self.assertEqual(signals_1[1], 0)
        self.assertEqual(signals_1[125], 0)
        self.assertEqual(signals_1[126], 1)
        self.assertEqual(signals_1[127], 1)
        self.assertEqual(signals_1[128], 1)
        self.assertEqual(signals_1[190], 0)
        self.assertEqual(signals_1[191], -1)
        self.assertEqual(signals_1[192], -1)
        self.assertEqual(signals_1[193], -1)

        self.assertEqual(signals_2[1], 0)
        self.assertEqual(signals_2[48], 0)
        self.assertEqual(signals_2[49], 1)
        self.assertEqual(signals_2[50], 1)
        self.assertEqual(signals_2[51], 1)
        self.assertEqual(signals_2[61], 0)
        self.assertEqual(signals_2[62], -1)
        self.assertEqual(signals_2[63], 0)
        self.assertEqual(signals_2[64], -1)

        self.assertEqual(signals_3[1], 0)
        self.assertEqual(signals_3[18], 0)
        self.assertEqual(signals_3[19], -1)
        self.assertEqual(signals_3[20], -1)
        self.assertEqual(signals_3[21], -1)
        self.assertEqual(signals_3[33], 0)
        self.assertEqual(signals_3[34], 1)
        self.assertEqual(signals_3[35], 1)
        self.assertEqual(signals_3[36], 0)

        self.assertEqual(signals_4[1], 0)
        self.assertEqual(signals_4[58], 0)
        self.assertEqual(signals_4[59], 1)
        self.assertEqual(signals_4[60], 1)
        self.assertEqual(signals_4[61], 1)

    def test_plot(self):
        """
        Tests the functions for plotting.
        """

        # Creating an object of class
        test = RegimeSwitchingArbitrageRule(delta = 1.5, rho = 0.6)

        # Setting window size
        window_size = 60

        # Getting signals on a rolling basis
        signals = test.get_signals(self.Ratts[0], window_size, switching_variance = False, silence_warnings = True)

        # Testing
        for i in [0, 1, 2]:
            fig = test.plot_trades(self.Ratts[i], signals)
            self.assertEqual(type(fig), type(plt.figure()))
        plt.close("all")
