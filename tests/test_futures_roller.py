# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests functionality of Futures Rolling module.
"""
import os
import unittest

import numpy as np
import pandas as pd

from arbitragelab.util.rollers import (CrudeOilFutureRoller, NBPFutureRoller, RBFutureRoller,
                                       GrainFutureRoller, EthanolFutureRoller, plot_historical_future_slope_state)


class TestFuturesRoller(unittest.TestCase):
    """
    Test Futures Rolling implementation.
    """

    def setUp(self):
        """
        Loads futures price data.
        """

        project_path = os.path.dirname(__file__)

        # Load individual contracts that have individual rolling implementations.
        cl_data = pd.read_csv(project_path + '/test_data/cl.csv',
                              parse_dates=True, index_col="Dates").dropna()
        cl_data = cl_data['2006-01': '2019-12']
        self.cl_data = cl_data

        nbp_data = pd.read_csv(project_path + '/test_data/nbp.csv',
                               parse_dates=True, index_col="Dates").dropna()
        nbp_data = nbp_data['2006-01': '2019-12']
        self.nbp_data = nbp_data

        rb_data = pd.read_csv(project_path + '/test_data/rb.csv',
                              parse_dates=True, index_col="Dates").dropna()
        rb_data = rb_data['2006-01': '2019-12']
        self.rb_data = rb_data

        s_data = pd.read_csv(project_path + '/test_data/s.csv',
                             parse_dates=True, index_col="Date").dropna()
        s_data = s_data['2006-01': '2019-12']
        self.s_data = s_data

        eh1_data = pd.read_csv(project_path + '/test_data/eh1.csv',
                               parse_dates=True, index_col="Date").dropna()
        eh1_data = eh1_data['2006-01': '2019-12']
        self.eh1_data = eh1_data

        eh2_data = pd.read_csv(project_path + '/test_data/eh2.csv',
                               parse_dates=True, index_col="Date").dropna()
        eh2_data = eh2_data['2006-01': '2019-12']
        self.eh2_data = eh2_data

    def test_crude_roller(self):
        """
        Tests futures roller implementation that expects expiration to
        be 3-4 days prior to the 25th of each month. In this case the Crude
        Oil contract is used as an example.
        """

        # Fit the roller object with the Crude Oil future contract
        wti_roller = CrudeOilFutureRoller().fit(self.cl_data)
        wti_gaps = wti_roller.transform()

        # Roll the prices series, should be mostly positive, but at the end there should be
        # a negative dip that needs to show up in the rolled up series.
        self.assertEqual(np.sign(wti_gaps.cumsum().mean()), np.sign(-1))

        # Roll the future price series and handle negative roll, (the event of march 2020).
        # The end result from the 'handle_negative_roll' should always be positive.
        non_negative_cl = wti_roller.transform(handle_negative_roll=True)
        self.assertEqual(np.sign(non_negative_cl.cumsum().mean()), np.sign(1))

        summary = wti_roller.diagnostic_summary()
        # Check sign of top 10 gaps from the whole series. Should be negative.
        top_gaps = summary.sort_values(by='gap').head(10)['gap']
        mean_top_gaps = top_gaps.values.cumsum().mean()
        self.assertEqual(np.sign(mean_top_gaps), np.sign(-1))

    def test_uk_gas_roller(self):
        """
        Tests futures roller implementation that expects expiration to
        be 2 days prior to the end of the month. In this case the NBP contract
        is used as an example.
        """

        # Fit the roller object with the UK Natural Gas future contract
        nbp_roller = NBPFutureRoller().fit(self.nbp_data)
        nbp_gaps = nbp_roller.transform()

        # Check that the forward rolled series is positive.
        self.assertEqual(
            np.sign((self.nbp_data['PX_LAST'] - nbp_gaps).mean()), np.sign(1))

        # Check that the backward rolled series is also positive.
        non_negative_nbp = nbp_roller.transform(
            roll_forward=False, handle_negative_roll=True)
        self.assertEqual(np.sign(non_negative_nbp.cumsum().mean()), np.sign(1))

    def test_gasoline_roller(self):
        """
        Test futures roller implementation that expects expiration to be
        the last day of the month. In this case the RBOB contract is used
        as an example.
        """

        # Fit the roller object with the RBOB Gasoline future contract
        rbob_roller = RBFutureRoller().fit(self.rb_data)
        rbob_gaps = rbob_roller.transform()

        # Check that the forward rolled series is positive.
        self.assertEqual(np.sign(rbob_gaps.cumsum().mean()), np.sign(1))

    def test_grain_roller(self):
        """
        Tests futures roller implementation that expects expiration to
        be on 15th of each month. In this case the Soybean Contract as an
        example.
        """

        # Fit the roller object with the Soybean future contract
        soyb_roller = GrainFutureRoller().fit(self.s_data)
        soyb_gaps = soyb_roller.transform()

        # Roll the prices series, should be mostly positive.
        self.assertEqual(np.sign(soyb_gaps.cumsum().mean()), np.sign(1))

    def test_ethanol_roller(self):
        """
        Tests futures roller implementation that expects expiration to
        be on the 3rd of each month.
        """

        # Fit the roller object with the Ethanol future contract
        ethanol_roller = EthanolFutureRoller().fit(self.eh1_data)
        ethanol_gaps = ethanol_roller.transform()

        # Roll the prices series, should be mostly positive.
        self.assertEqual(np.sign(ethanol_gaps.cumsum().mean()), np.sign(1))

    def test_contango_backwardation_plotter(self):
        """
        Verifies execution of plot for contango/backwardation plotting
        method.
        """

        plot_historical_future_slope_state(self.eh1_data['PX_LAST'], self.eh2_data['PX_OPEN'])
