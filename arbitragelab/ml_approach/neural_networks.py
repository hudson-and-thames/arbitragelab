# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This module implements the Multi Layer Perceptron, RNN model and the Pi Sigma Model.
"""

from keras.models import Model
# from keras.layers.core import Dense, Activation, Lambda
from keras.layers import  Input, LSTM, Dense, Activation, Lambda
from keras.callbacks.callbacks import History

import tensorflow as tf

import matplotlib.pyplot as plt

class BaseNeuralNetwork:
    """
    Skeleton Class to be inherited by child
    neural network implementations.
    """

    def __init__(self):
        """
        Initializing variables.
        """

        self.fitted_model = None

    def fit(self, *args, **kwargs) -> History:
        """
        Wrapper over the keras model fit function.
        """

        fitted_model = self.model.fit(*args, **kwargs)
        self.fitted_model = fitted_model

        return fitted_model

    def predict(self, *args, **kwargs):
        """
        Wrapper over the keras model predict function.
        """

        return self.model.predict(*args, **kwargs)

    def plot_loss(self) -> list:
        """
        Method that returns visual plot of the loss trajectory.
        """

        result = plt.plot(self.fitted_model.history['loss'])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        return result

class MultiLayerPerceptron(BaseNeuralNetwork):
    """
    Multi Layer Perceptron implementation.

    Regression: loss_fn="mean_squared_error", optmz="adam", metrics=["r2_score"]
    num_outputs=1

    Classification: loss_fn="categorical_crossentropy", optmz="adam", metrics=["accuracy"]
    num_outputs=num_classes?
    """

    def __init__(self, frame_size, hidden_size=2, num_outputs=1, loss_fn="mean_squared_error",
                 optmz="adam", metrics="accuracy",
                 hidden_layer_activation_function="relu", output_layer_act_func="linear"):
        """
        Initialization of variables.
        """

        super().__init__()

        self.model = None
        self.frame_size = frame_size
        self.hidden_size = hidden_size
        self.output_size = num_outputs
        self.loss_fn = loss_fn
        self.optimizer = optmz
        self.metrics = metrics
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_act_func

    def build(self):
        """

        :return: (Model)
        """

        input_layer = Input((self.frame_size,))

        hidden_layer = Dense(self.hidden_size,
                             activation=self.hidden_layer_activation_function)(input_layer)

        output_layer = Dense(self.output_size,
                             activation=self.output_layer_activation_function)(hidden_layer)

        model = Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(loss=self.loss_fn, optimizer=self.optimizer,
                      metrics=self.metrics)

        self.model = model

        return model

class RecurrentNeuralNetwork(BaseNeuralNetwork):
    """
    Recurrent Neural Network implementation.
    """

    def __init__(self, input_shape, hidden_size=50, num_outputs=1, loss_fn="mean_squared_error",
                 optmz="adam", metrics="accuracy",
                 hidden_layer_activation_function="relu", output_layer_act_func="linear"):
        """
        Initialization of Variables.
        """

        super().__init__()

        self.model = None
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_size = num_outputs
        self.loss_fn = loss_fn
        self.optimizer = optmz
        self.metrics = metrics
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_act_func

    def build(self):
        """
        Builds and compiles model architecture.
        """

        input_layer = Input(self.input_shape)

        hidden_layer = LSTM(self.hidden_size, activation=self.hidden_layer_activation_function,
                            input_shape=self.input_shape)(input_layer)

        output_layer = Dense(self.output_size,
                             activation=self.output_layer_activation_function)(hidden_layer)

        model = Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(loss=self.loss_fn, optimizer=self.optimizer,
                      metrics=self.metrics)

        self.model = model

        return model

class PiSigmaNeuralNetwork(BaseNeuralNetwork):
    """
    Pi Sigma Neural Network implementation.
    """

    def __init__(self, frame_size, hidden_size=2, num_outputs=1, loss_fn="mean_squared_error",
                 optmz="sgd", metrics="accuracy",
                 hidden_layer_activation_function="linear", output_layer_act_func="sigmoid"):
        """
        Inialization of variables.
        """

        super().__init__()

        self.model = None
        self.frame_size = frame_size
        self.hidden_size = hidden_size
        self.output_size = num_outputs
        self.loss_fn = loss_fn
        self.optimizer = optmz
        self.metrics = metrics
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_act_func

    def build(self):
        """

        :return: (Model)
        """

        input_layer = Input((self.frame_size,))

        second_sigma_layer = Dense(self.hidden_size,
                                   activation=self.hidden_layer_activation_function)(input_layer)

        pi_layer = Lambda(self._pi_this)(second_sigma_layer)

        act_layer = Activation(self.output_layer_activation_function)(pi_layer)

        model = Model(inputs=[input_layer], outputs=[act_layer])

        model.compile(loss=self.loss_fn, optimizer=self.optimizer,
                      metrics=self.metrics)

        self.model = model

        return model

    @staticmethod
    def _pi_this(tensor):
        """
        :return: (tf.Tensor)
        """

        prod = tf.math.reduce_prod(tensor, keepdims=True, axis=1)

        return prod
