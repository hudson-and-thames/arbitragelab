"""
Tests Spread modeling Threshold AutoRegression model implementation.
"""
import os
import unittest

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.iolib.summary import Summary

from arbitragelab.ml_approach.tar import TAR

class TestTAR(unittest.TestCase):
    """
    Test Threshold AutoRegressive Implementation.
    """

    def setUp(self):
        """
        Loads data needed for model fitting.
        """

        # Set working seed.
        np.random.seed(0)

        project_path = os.path.dirname(__file__)

        # Load non negative versions of CL and RB contracts.
        wti_contract_df = pd.read_csv(
            project_path + '/test_data/NonNegative_CL_forward_roll.csv').set_index('Dates')
        rbob_contract_df = pd.read_csv(
            project_path + '/test_data/NonNegative_nRB_forward_roll.csv').set_index('Dates')

        # Concatenate both contracts into one dataframe.
        working_df = pd.concat([wti_contract_df, rbob_contract_df], axis=1)
        working_df.index = pd.to_datetime(working_df.index)
        working_df.columns = ['wti', 'gasoline']
        working_df.dropna(inplace=True)

        self.working_df = working_df

    def test_tar(self):
        """
        Test TAR model using standard unprocessed spread as input value.
        """
        #pylint: disable=too-many-function-args

        # Initialize TAR model with the standard [leg1 - leg2] spread as input value.
        model = TAR((self.working_df['gasoline'] - self.working_df['wti']))

        # Check if returned a valid object.
        self.assertTrue(type(model), TAR)

        tar_results = model.fit()

        # Check that it returned valid regression results.
        self.assertTrue(type(tar_results), RegressionResults)

        # Check fitted values characteristics.
        self.assertAlmostEqual(tar_results.fittedvalues.mean(), 0, 0)
        self.assertAlmostEqual(tar_results.fittedvalues.max(), 0.011, 3)
        self.assertTrue(np.sign(tar_results.fittedvalues.min()), np.sign(-1))

        self.assertTrue(type(tar_results.summary()), Summary)

        # Check that it returned valid custom model results.
        self.assertTrue(type(model.summary()), pd.DataFrame)
