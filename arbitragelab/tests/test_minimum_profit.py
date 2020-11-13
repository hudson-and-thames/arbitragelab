"""
Tests function of Minimum Profit Condition Optimization module:
minimum_profit/MinimumProfitSimulation.py, MinimumProfit.py
"""
import unittest
import warnings
import os
import pandas as pd
import numpy as np


class TestMinimumProfit(unittest.TestCase):
    """
    Test Minimum Profit Condition Optimization module.
    """

    def setUp(self):
        """
        Read empirical data mentioned in the paper.
        ANZ-ADB pair (Jan 2nd 2001 to Aug 30th 2002)
        BHP-RIO pair (Jan 2nd 2004 to Jun 30th 2005)
        TNS-TVL pair (Jan 2nd 2004 to Jun 30th 2005)

        Simulated data will be tested on the fly
        :return:
        """