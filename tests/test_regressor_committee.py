# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests Regressor Committee Class.
"""

import unittest
from keras.engine.training import Model
from keras.callbacks.callbacks import History
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from arbitragelab.ml_approach.neural_networks import MultiLayerPerceptron, RecurrentNeuralNetwork, PiSigmaNeuralNetwork
from arbitragelab.ml_approach.regressor_committee import RegressorCommittee

# pylint: disable=unbalanced-tuple-unpacking

class TestRegressorCommittee(unittest.TestCase):
    """
    Test Regressor Committee Implementation.
    """

    def test_mlp_committee(self):
        """
        Tests the Multi Layer Perceptron implementation.
        """

        # Generate regression data.
        features, target = make_regression(500)

        _, frame_size = features.shape

        mlp_params = {'frame_size': frame_size, 'hidden_size': 8, 'num_outputs': 1, 'loss_fn': "mean_squared_error", 
                            'optmz': "adam", 'metrics': [], 'hidden_layer_activation_function': "sigmoid",
                            'output_layer_act_func': "linear"}

        # Initialize mlp committee.
        committee = RegressorCommittee(mlp_params, num_committee=2, epochs=100, verbose=False)

        feat_train, feat_test, trgt_train, _ = train_test_split(
            features, target, test_size=0.3, shuffle=False)

        # Check if fit return is a valid RegressorCommittee model.
        self.assertTrue(type(committee.fit(feat_train, trgt_train)), RegressorCommittee)

        # Check if amount of predicted values match the input values.
        self.assertTrue(len(committee.predict(feat_test)) > 0)
