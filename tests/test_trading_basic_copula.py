# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Unit tests for new basic_copula strategy.
"""
# pylint: disable = invalid-name, protected-access, too-many-locals, unsubscriptable-object, too-many-statements, undefined-variable

import os
import unittest

import pandas as pd

from arbitragelab.trading.copula_approach import BasicCopulaTradingRule
from arbitragelab.copula_approach import construct_ecdf_lin
import arbitragelab.copula_approach.archimedean as coparc


class TestBasicCopulaTradingRule(unittest.TestCase):
    """
    Test the BasicCopulaTradingRule class.
    """

    def setUp(self):
        """
        Get the correct directory and data.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + "/test_data/BKD_ESC_2009_2011.csv"
        self.stocks = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_standard_use_or(self):
        """
        Testing the standard use of the strategy with 'or' logic.
        """

        train_data = self.stocks[900:]
        test_data = self.stocks[:900]

        # Create a trading strategy
        cop_trading = BasicCopulaTradingRule(exit_rule='or', open_probabilities=(0.5, 0.95),
                                             exit_probabilities=(0.5, 0.5))


        # Setting probabilities
        cop_trading.current_probabilities = (0.4, 0.6)
        cop_trading.prev_probabilities = (0.6, 0.6)

        # Adding a copula
        cop = coparc.Gumbel(theta=2)
        cop_trading.set_copula(cop)

        # Constructing cdf for x and y
        cdf_x = construct_ecdf_lin(train_data['BKD'])
        cdf_y = construct_ecdf_lin(train_data['ESC'])
        cop_trading.set_cdf(cdf_x, cdf_y)

        # Trading simulation
        for time, values in test_data.iterrows():
            x_price = values['BKD']
            y_price = values['ESC']

            # Adding values
            cop_trading.update_probabilities(x_price, y_price)

            # Check if it's time to enter a trade
            trade, side = cop_trading.check_entry_signal()

            # Close previous trades if needed
            cop_trading.update_trades(update_timestamp=time)

            if trade:  # Open a new trade if needed
                cop_trading.add_trade(start_timestamp=time, side_prediction=side)

        # Test correct number of trades
        self.assertEqual(len(cop_trading.open_trades), 0)
        self.assertEqual(len(cop_trading.closed_trades), 45)

        # Test dates of trades
        self.assertEqual(list(cop_trading.closed_trades.keys())[0].to_datetime64(),
                         pd.Timestamp('2009-09-29 00:00+00:00').to_datetime64())
        self.assertEqual(list(cop_trading.closed_trades.keys())[10].to_datetime64(),
                         pd.Timestamp('2009-10-16 00:00+00:00').to_datetime64())
        self.assertEqual(list(cop_trading.closed_trades.keys())[20].to_datetime64(),
                         pd.Timestamp('2011-03-02 00:00+00:00').to_datetime64())

        # Test if all keys are present
        keys = ['t1', 'exit_proba', 'uuid', 'side', 'initial_proba']
        for key in keys:
            self.assertIn(key, list(cop_trading.closed_trades.values())[0].keys())

    def test_standard_use_and(self):
        """
        Testing the standard use of the strategy with 'and' logic.
        """

        train_data = self.stocks[900:]
        test_data = self.stocks[:900]

        # Create a trading strategy
        cop_trading = BasicCopulaTradingRule(exit_rule='and', open_probabilities=(0.5, 0.95),
                                             exit_probabilities=(0.9, 0.5))


        # Setting probabilities
        cop_trading.current_probabilities = (0.4, 0.6)
        cop_trading.prev_probabilities = (0.6, 0.6)

        # Adding a copula
        cop = coparc.Gumbel(theta=2)
        cop_trading.set_copula(cop)

        # Constructing cdf for x and y
        cdf_x = construct_ecdf_lin(train_data['BKD'])
        cdf_y = construct_ecdf_lin(train_data['ESC'])
        cop_trading.set_cdf(cdf_x, cdf_y)

        # Trading simulation
        for time, values in test_data.iterrows():
            x_price = values['BKD']
            y_price = values['ESC']

            # Adding values
            cop_trading.update_probabilities(x_price, y_price)

            # Check if it's time to enter a trade
            trade, side = cop_trading.check_entry_signal()

            # Close previous trades if needed
            cop_trading.update_trades(update_timestamp=time)

            if trade:  # Open a new trade if needed
                cop_trading.add_trade(start_timestamp=time, side_prediction=side)

        self.assertEqual(len(cop_trading.open_trades), 21)
        self.assertEqual(len(cop_trading.closed_trades), 24)

        # Test dates of trades
        self.assertEqual(list(cop_trading.closed_trades.keys())[0].to_datetime64(),
                         pd.Timestamp('2009-09-29 00:00+00:00').to_datetime64())
        self.assertEqual(list(cop_trading.closed_trades.keys())[10].to_datetime64(),
                         pd.Timestamp('2009-10-16 00:00+00:00').to_datetime64())
        self.assertEqual(list(cop_trading.closed_trades.keys())[20].to_datetime64(),
                         pd.Timestamp('2011-03-02 00:00+00:00').to_datetime64())

        # Test if all keys are present
        keys = ['t1', 'exit_proba', 'uuid', 'side', 'initial_proba']
        for key in keys:
            self.assertIn(key, list(cop_trading.closed_trades.values())[0].keys())


    def test_errors_wrong_use(self):
        """
        Testing the errors if the strategy is not used correctly.
        """

        # Create a trading strategy
        cop_trading = BasicCopulaTradingRule(exit_rule='or')

        # Updating probability without copula
        with self.assertRaises(ValueError):
            cop_trading.update_probabilities(0.1, 0.2)

        # Updating probability with copula but without cdf
        cop = coparc.Gumbel(theta=2)
        cop_trading.set_copula(cop)

        with self.assertRaises(ValueError):
            cop_trading.update_probabilities(0.1, 0.2)
