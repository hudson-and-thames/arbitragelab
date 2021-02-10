# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests Spread Modeling Helper Class.
"""
import os
import unittest
import pandas as pd

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

        project_path = os.path.dirname(__file__)

        wti_contract_df = pd.read_csv(project_path + '/test_data/NonNegative_CL_forward_roll.csv').set_index('Dates')
        rbob_contract_df = pd.read_csv(project_path + '/test_data/NonNegative_nRB_forward_roll.csv').set_index('Dates')

        working_df = pd.concat([wti_contract_df, rbob_contract_df], axis=1)
        working_df.index = pd.to_datetime(working_df.index)
        working_df.columns = ['wti', 'gasoline']

        working_df.dropna(inplace=True)

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

        committee = RegressorCommittee(mlp_params, num_committee=1, epochs=100, verbose=False)
        fitted_com = committee.fit(helper.input_train, helper.target_train, helper.input_test, helper.target_test)

        # Check if fit return is a valid Committee object.
        self.assertTrue(type(fitted_com), RegressorCommittee)

        # Check if predictions of all sets are returned.
        self.assertTrue(len(helper.plot_model_results(committee)) == 3)

        _, test_pred, oos_pred = helper.plot_model_results(committee)

        committee.plot_losses()

        helper.get_metrics(self.working_df)

        helper.get_filtering_results(helper.target_oos, oos_pred, test_pred, self.working_df)

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

        # Check if fit return is a valid Committee object.
        self.assertTrue(type(fitted_com), RegressorCommittee)

        # Check if predictions of all sets are returned.
        self.assertTrue(len(helper.plot_model_results(committee)) == 3)
