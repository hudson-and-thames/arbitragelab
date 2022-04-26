# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests Spread Modeling Neural Network Classes.
"""

import unittest
import numpy as np
import tensorflow as tf
from keras.engine.training import Model
from keras.callbacks import History
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from arbitragelab.ml_approach.neural_networks import MultiLayerPerceptron, RecurrentNeuralNetwork, PiSigmaNeuralNetwork

# pylint: disable=unbalanced-tuple-unpacking

class TestNeuralNetworks(unittest.TestCase):
    """
    Test Neural Network Implementations.
    """

    def setUp(self):
        """
        Loads data needed for model fitting.
        """

        # Set seed values to numerical libraries.
        seed_value = 0
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)

        self.seed_value = seed_value

    def test_mlp(self):
        """
        Tests the Multi Layer Perceptron implementation.
        """

        # Generate regression data.
        features, target = make_regression(500, random_state=self.seed_value)

        _, frame_size = features.shape

        # Initialize mlp.
        regressor = MultiLayerPerceptron(frame_size, num_outputs=1, loss_fn="mean_squared_error",
                                         optmizer="adam", metrics=[], hidden_layer_activation_function="relu",
                                         output_layer_act_func="linear")

        # Check if built model is a valid keras model.
        self.assertTrue(type(regressor.build()), Model)

        feat_train, feat_test, trgt_train, _ = train_test_split(
            features, target, test_size=0.3, shuffle=False)

        # Check if fit return is a valid History model.
        self.assertTrue(type(regressor.fit(feat_train, trgt_train,
                                           batch_size=20, epochs=10, verbose=False)), History)

        # Check if amount of predicted values match the input values.
        self.assertTrue(len(regressor.predict(feat_test)) > 0)

        # Check Predicted values' means.
        self.assertAlmostEqual(regressor.predict(feat_test).mean(), -1.182, 2)

        # Check if proper plotting object is returned.
        self.assertTrue(type(regressor.plot_loss()), list)

    def test_rnn(self):
        """
        Tests the Recurrent Neural Network implementation.
        """

        # Generate regression data.
        features, target = make_regression(500, random_state=self.seed_value)

        feat_train, feat_test, trgt_train, _ = train_test_split(
            features, target, test_size=0.3, shuffle=True)

        feat_train = feat_train.reshape((feat_train.shape[0], feat_train.shape[1], 1))

        # Initialize rnn.
        regressor = RecurrentNeuralNetwork((feat_train.shape[1], 1), num_outputs=1, loss_fn="mean_squared_error",
                                           optmizer="adam", metrics=["accuracy"], hidden_layer_activation_function="relu",
                                           output_layer_act_func="linear")

        # Check if built model is a valid keras model.
        self.assertTrue(type(regressor.build()), Model)

        # Check if fit return is a valid History model.
        self.assertTrue(type(regressor.fit(feat_train, trgt_train,
                                           epochs=10, verbose=False)), History)

        feat_test = feat_test.reshape((feat_test.shape[0], feat_test.shape[1], 1))

        # Check if amount of predicted values match the input values.
        self.assertTrue(len(regressor.predict(feat_test)) > 0)

        # Check Predicted values' means.
        self.assertAlmostEqual(regressor.predict(feat_test).mean(), -0.020863, 1)

        # Check if proper plotting object is returned.
        self.assertTrue(type(regressor.plot_loss()), list)

    def test_pisigma(self):
        """
        Tests the Pi Sigma Neural Network implementation.
        """

        # Generate regression data.
        features, target = make_regression(500, random_state=self.seed_value)

        _, frame_size = features.shape

        # Initialize honn.
        regressor = PiSigmaNeuralNetwork(frame_size, num_outputs=1, loss_fn="mean_squared_error",
                                         optmizer="adam", metrics=[], hidden_layer_activation_function="relu",
                                         output_layer_act_func="linear")

        # Check if built model is a valid keras model.
        self.assertTrue(type(regressor.build()), Model)

        feat_train, feat_test, trgt_train, _ = train_test_split(
            features, target, test_size=0.3, shuffle=False)

        # Check if fit return is a valid History model.
        self.assertTrue(type(regressor.fit(feat_train, trgt_train,
                                           batch_size=20, epochs=10, verbose=False)), History)

        # Check if amount of predicted values match the input values.
        self.assertTrue(len(regressor.predict(feat_test)) > 0)

        # Check Predicted values' means.
        self.assertAlmostEqual(regressor.predict(feat_test).mean(), 0.82836, 2)

        # Check if proper plotting object is returned.
        self.assertTrue(type(regressor.plot_loss()), list)
