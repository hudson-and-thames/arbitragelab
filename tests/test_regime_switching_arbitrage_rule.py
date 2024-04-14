"""
Tests functions from Regime Switching Arbitrage Rule module.
"""
# pylint: disable=invalid-name

import unittest
import os

import pandas as pd
import matplotlib.pyplot as plt

from arbitragelab.time_series_approach.regime_switching_arbitrage_rule import RegimeSwitchingArbitrageRule


class TestRegimeSwitchingArbitrageRule(unittest.TestCase):
    """
    Tests the class of Regime Switching Arbitrage Rule module.
    """

    def setUp(self):
        """
        Sets the file path for the data and testing variables.
        """

        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/CL=F_NG=F_data.csv'
        data = pd.read_csv(self.path)
        data = data.set_index('Date')
        Ratt = data["NG=F"]/data["CL=F"]


        self.Ratts = Ratt.values, Ratt, pd.DataFrame(Ratt)

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
            signal = test.get_signal(self.Ratts[i][-window_size:], switching_variance=False, silence_warnings=False)
            self.assertEqual(signal.tolist(), [False, True, True, False])

        signal = test.get_signal(self.Ratts[0][-window_size:], switching_variance=True, silence_warnings=True)
        self.assertEqual(signal.tolist(), [False, True, True, False])

        # Getting signals on a rolling basis
        signals = test.get_signals(self.Ratts[0], window_size, switching_variance=True, silence_warnings=True)

        # Testing the result
        self.assertEqual(signals[0].tolist(), [False, False, False, False])
        self.assertEqual(signals[79].tolist(), [False, False, False, False])
        self.assertEqual(signals[113].tolist(), [False, False, False, False])

    def test_trade(self):
        """
        Tests the functions for getting and plotting trades.
        """

        # Creating an object of class
        test = RegimeSwitchingArbitrageRule(delta=1.5, rho=0.6)

        # Setting window size
        window_size = 100

        # Getting signals on a rolling basis
        signals = test.get_signals(self.Ratts[0], window_size, switching_variance=True, silence_warnings=True)

        # Deciding the trades based on the signals
        trades = test.get_trades(signals)

        # Testing the result
        self.assertEqual(trades[0].tolist(), [True, False, False, True])
        self.assertEqual(trades[79].tolist(), [False, False, False, False])
        self.assertEqual(trades[113].tolist(), [False, False, False, False])

        # Plotting trades
        for i in [0, 1, 2]:
            fig = test.plot_trades(self.Ratts[i], trades)
            self.assertEqual(type(fig), type(plt.figure()))

    def test_change(self):
        """
        Tests the functions for changing strategy.
        """

        # Creating an object of class
        test = RegimeSwitchingArbitrageRule(delta=1.5, rho=0.6)

        # Setting window size
        window_size = 60

        # Changing rules in the high regime
        ol_rule = lambda Xt, mu, delta, sigma: Xt <= mu - delta*sigma
        cl_rule = lambda Xt, mu, delta, sigma: Xt >= mu
        os_rule = lambda Xt, mu, delta, sigma: Xt >= mu + delta*sigma
        cs_rule = lambda Xt, mu, delta, sigma: Xt <= mu

        test.change_strategy("High", "Long", "Open", ol_rule)
        test.change_strategy("High", "Long", "Close", cl_rule)
        test.change_strategy("High", "Short", "Open", os_rule)
        test.change_strategy("High", "Short", "Close", cs_rule)

        # Testing the result
        self.assertEqual(test.strategy["High"]["Long"]["Open"], ol_rule)
        self.assertEqual(test.strategy["High"]["Long"]["Close"], cl_rule)
        self.assertEqual(test.strategy["High"]["Short"]["Open"], os_rule)
        self.assertEqual(test.strategy["High"]["Short"]["Close"], cs_rule)

        # Changing rules in the low regime
        ol_rule = lambda Xt, mu, delta, sigma, prob: Xt <= mu - delta*sigma and prob >= 0.7
        cl_rule = lambda Xt, mu, delta, sigma: Xt >= mu
        os_rule = lambda Xt, mu, delta, sigma, prob, rho: Xt >= mu + delta*sigma and prob >= rho
        cs_rule = lambda Xt, mu, delta, sigma: Xt <= mu

        test.change_strategy("Low", "Long", "Open", ol_rule)
        test.change_strategy("Low", "Long", "Close", cl_rule)
        test.change_strategy("Low", "Short", "Open", os_rule)
        test.change_strategy("Low", "Short", "Close", cs_rule)

        # Testing the result
        self.assertEqual(test.strategy["Low"]["Long"]["Open"], ol_rule)
        self.assertEqual(test.strategy["Low"]["Long"]["Close"], cl_rule)
        self.assertEqual(test.strategy["Low"]["Short"]["Open"], os_rule)
        self.assertEqual(test.strategy["Low"]["Short"]["Close"], cs_rule)

        # Getting signals on a rolling basis
        signals = test.get_signals(self.Ratts[0], window_size, switching_variance=True, silence_warnings=True)

        # Deciding the trades based on the signals
        trades = test.get_trades(signals)

        # Testing the result
        self.assertEqual(signals[0].tolist(), [False, False, False, False])
        self.assertEqual(signals[79].tolist(), [False, False, False, False])
        self.assertEqual(signals[143].tolist(), [True, False, False, True])
        self.assertEqual(trades[0].tolist(), [False, False, False, False])

        # Testing the exception
        with self.assertRaises(Exception):
            ol_rule = lambda Xt, mu, delta, sigma, error: Xt <= mu - delta*sigma*error
            test.change_strategy("High", "Long", "Open", ol_rule)
