"""
Tests Regressor Committee Class.
"""

import unittest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
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
                      'optmizer': "adam", 'metrics': [], 'hidden_layer_activation_function': "sigmoid",
                      'output_layer_act_func': "linear"}

        # Initialize mlp committee.
        committee = RegressorCommittee(mlp_params, num_committee=2, epochs=100, verbose=False)

        feat_train, feat_test, trgt_train, trgt_test = train_test_split(
            features, target, test_size=0.3, shuffle=False)

        result = committee.fit(feat_train, trgt_train, feat_test, trgt_test)

        # Check if fit return is a valid RegressorCommittee model.
        self.assertTrue(type(result), RegressorCommittee)

        # Check if amount of predicted values match the input values.
        self.assertTrue(len(committee.predict(feat_test)) > 0)
