# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Unit tests for mixed copulas - CFG and CTG.
"""
# pylint: disable =

import os
import unittest

import pandas as pd

from arbitragelab.copula_approach.mixed_copulas import CFGMixCop, CTGMixCop


class TestBasicCopulaStrategy(unittest.TestCase):
    """
    Test the BasicCopulaStrategy class.
    """

    def setUp(self):
        """
        Get the correct directory and data.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + "/test_data/BKD_ESC_2009_2011.csv"
        self.stocks = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_cfgmixcop_fit(self):
        """
        Test CFGMixCop copula class.
        """

        # Init without parameters
        _ = CFGMixCop()

        # Init with parameters
        cop = CFGMixCop([2, 2, 2])

        # Fit to data
        cop.fit(self.stocks)

        # Check describe
        descr = cop.describe()
        self.assertEqual(descr['Descriptive Name'], 'Bivariate Clayton-Frank-Gumbel Mixed Copula')
        self.assertEqual(descr['Class Name'], 'CFGMixCop')
        self.assertAlmostEqual(descr['Clayton theta'], 6.6258756, 2)
        self.assertAlmostEqual(descr['Frank theta'], 4.0003041, 1)
        self.assertAlmostEqual(descr['Gumbel theta'], 4.7874674, 1)
        self.assertAlmostEqual(descr['Clayton weight'], 0.503564, 1)
        self.assertAlmostEqual(descr['Frank weight'], 0.0, 1)
        self.assertAlmostEqual(descr['Gumbel weight'], 0.4964350, 1)

        # Check side-loading pairs generation
        sample_pairs = cop.sample(num=100)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

    def test_ctgmixcop_fit(self):
        """
        Test CTGMixCop copula class.
        """

        # Init without parameters
        _ = CTGMixCop()

        # Init with parameters
        cop = CTGMixCop([4, 0.9, 4, 4])

        # Fit to data, lower sample size to improve unit test speed
        cop.fit(self.stocks.iloc[:20])

        # Check describe
        descr = cop.describe()
        self.assertEqual(descr['Descriptive Name'], 'Bivariate Clayton-Student-Gumbel Mixed Copula')
        self.assertEqual(descr['Class Name'], 'CTGMixCop')
        self.assertAlmostEqual(descr['Clayton theta'], 2.1268764, 1)
        self.assertAlmostEqual(descr['Student rho'], 0.001, 1)
        self.assertAlmostEqual(descr['Student nu'], 4.00676, 1)
        self.assertAlmostEqual(descr['Gumbel theta'], 5, 1)
        self.assertAlmostEqual(descr['Clayton weight'], 1, 1)
        self.assertAlmostEqual(descr['Student weight'], 0, 1)
        self.assertAlmostEqual(descr['Gumbel weight'], 0, 1)

        # Check side-loading pairs generation
        sample_pairs = cop.sample(num=100)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))
