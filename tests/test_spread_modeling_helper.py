# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests Spread Modeling Helper Class.
"""

import unittest

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from arbitragelab.cointegration_approach.johansen import JohansenPortfolio
from arbitragelab.ml_approach.regressor_committee import RegressorCommittee
#from arbitragelab.ml_approach.neural_networks import MultiLayerPerceptron, RecurrentNeuralNetwork, PiSigmaNeuralNetwork
from arbitragelab.util.spread_modeling_helper import SpreadModelingHelper

class TestSpreadModelingHelper(unittest.TestCase):
    """
    Tests Spread Modeling Helper class.
    """

    @staticmethod
    def test_helper():
        """
        Tests higher order term generation.
        """

        wti_contract_df = pd.read_csv('./test_data/NonNegative_CL_forward_roll.csv').set_index('Dates')
        rbob_contract_df = pd.read_csv('./test_data/NonNegative_nRB_forward_roll.csv').set_index('Dates')

        working_df = pd.concat([wti_contract_df, rbob_contract_df], axis=1)
        working_df.index = pd.to_datetime(working_df.index)
        working_df.columns = ['wti', 'gasoline']

        working_df.dropna(inplace=True)
        #working_df

        johansen_portfolio = JohansenPortfolio()
        johansen_portfolio.fit(working_df)
        sprd = johansen_portfolio.construct_mean_reverting_portfolio(working_df).pct_change()

        helper = SpreadModelingHelper(sprd, False, unique_sampling=True)

        _, frame_size = helper.input_train.shape

        mlp_params = {'frame_size': frame_size, 'hidden_size': 8, 'num_outputs': 1, 'loss_fn': "mean_squared_error",
                      'optmz': "adam", 'metrics': [], 'hidden_layer_activation_function': "sigmoid",
                      'output_layer_act_func': "linear"}

        committee = RegressorCommittee(mlp_params, num_committee=2, epochs=100, verbose=False)
        committee.fit(helper.input_train, helper.target_train)

        #train_set, train_pred, test_set, test_pred, oos_set, oos_pred = helper.plot_model_results(committee)
