"""
Tests Spread Modeling Helper Class.
"""
import os
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf

from arbitragelab.cointegration_approach.johansen import JohansenPortfolio
from arbitragelab.ml_approach.regressor_committee import RegressorCommittee
from arbitragelab.util.spread_modeling_helper import SpreadModelingHelper

class TestSpreadModelingHelper(unittest.TestCase):
    """
    Tests Spread Modeling Helper class.
    """

    def setUp(self):
        """
        Loads data needed for model fitting.
        """

        # Set seed values to numerical libraries.
        seed_value = 0
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

        # Collect all contract price data.
        project_path = os.path.dirname(__file__)

        wti_contract_df = pd.read_csv(project_path + '/test_data/NonNegative_CL_forward_roll.csv').set_index('Dates')
        rbob_contract_df = pd.read_csv(project_path + '/test_data/NonNegative_nRB_forward_roll.csv').set_index('Dates')

        working_df = pd.concat([wti_contract_df, rbob_contract_df], axis=1)
        working_df.index = pd.to_datetime(working_df.index)
        working_df.columns = ['wti', 'gasoline']

        working_df.dropna(inplace=True)

        # Transform contract price data to spread.
        johansen_portfolio = JohansenPortfolio()
        johansen_portfolio.fit(working_df)
        sprd = johansen_portfolio.construct_mean_reverting_portfolio(working_df).pct_change()

        self.working_df = working_df
        self.sprd = sprd

    def test_vanilla_helper(self):
        """
        Tests overall framework structure from dataset generation to model fitting.
        """

        helper = SpreadModelingHelper(self.sprd, insample_date_range=('2006', '2016'),
                                      oosample_date_range=('2017', None), feat_expansion=False,
                                      unique_sampling=True)

        _, frame_size = helper.input_train.shape

        mlp_params = {'frame_size': frame_size, 'hidden_size': 8, 'num_outputs': 1, 'loss_fn': "mean_squared_error",
                      'optmizer': "adam", 'metrics': [], 'hidden_layer_activation_function': "sigmoid",
                      'output_layer_act_func': "linear"}

        committee = RegressorCommittee(mlp_params, num_committee=2, epochs=100, verbose=False)
        fitted_com = committee.fit(helper.input_train, helper.target_train, helper.input_test, helper.target_test)

        # Check Result plotting functionality.
        self.assertTrue(issubclass(type(helper.plot_model_results(committee)), np.ndarray))

        # Check if fit return is a valid Committee object.
        self.assertTrue(type(fitted_com), RegressorCommittee)

        # Check Loss plotting functionality.
        self.assertTrue(issubclass(type(committee.plot_losses()), np.ndarray))

        # Check metrics return results.
        self.assertTrue(issubclass(type(helper.get_metrics(self.working_df)), pd.DataFrame))

        # Check Predicted values' means.
        self.assertAlmostEqual(helper.oos_pred.mean(), 0, 1)
        self.assertAlmostEqual(helper.test_pred.mean(), 0, 1)

        filter_events = helper.get_filtering_results(helper.target_oos, helper.oos_pred,
                                                     helper.test_pred, self.working_df)

        # Check Number of events returned.
        self.assertTrue(len(filter_events), 3)

    def test_honn_helper(self):
        """
        Tests overall framework structure from dataset generation to model fitting.
        """

        helper = SpreadModelingHelper(self.sprd, insample_date_range=('2006', '2016'),
                                      oosample_date_range=('2017', None), feat_expansion=True,
                                      unique_sampling=False)

        _, frame_size = helper.input_train.shape

        mlp_params = {'frame_size': frame_size, 'hidden_size': 8, 'num_outputs': 1, 'loss_fn': "mean_squared_error",
                      'optmizer': "adam", 'metrics': [], 'hidden_layer_activation_function': "sigmoid",
                      'output_layer_act_func': "linear"}

        committee = RegressorCommittee(mlp_params, num_committee=1, epochs=100, verbose=False)
        fitted_com = committee.fit(helper.input_train, helper.target_train, helper.input_test, helper.target_test)

        # Check Result plotting functionality.
        self.assertTrue(issubclass(type(helper.plot_model_results(committee)), np.ndarray))

        # Check Predicted values' means.
        self.assertAlmostEqual(helper.oos_pred.mean(), 0, 1)
        self.assertAlmostEqual(helper.test_pred.mean(), 0, 1)

        # Check if fit return is a valid Committee object.
        self.assertTrue(type(fitted_com), RegressorCommittee)
