# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This module implements the Multi Layer Perceptron, RNN model and the Pi Sigma Model.
"""

#pylint: disable=wrong-import-position
import os
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras.models import Model
from keras.callbacks.callbacks import History
from keras.layers import  Input, LSTM, Dense, Activation, Lambda
import matplotlib.pyplot as plt

class BaseNeuralNetwork:
    """
    Skeleton Class to be inherited by child neural network implementations.
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

    def plot_loss(self):
        """
        Method that returns visual plot of the loss trajectory in
        terms of epochs spent training.
        """

        plt.plot(self.fitted_model.history['loss'])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Plot")

class MultiLayerPerceptron(BaseNeuralNetwork):
    """
    Vanilla Multi Layer Perceptron implementation.
    """

    def __init__(self, frame_size: int, hidden_size: int = 2, num_outputs: int = 1,
                 loss_fn: str = "mean_squared_error", optmizer: str = "adam", metrics: str = "accuracy",
                 hidden_layer_activation_function: str = "relu", output_layer_act_func: str = "linear"):
        """
        Initialization of variables.

        :param frame_size: (int) The size of the input dataset.
        :param hidden_size: (int) Number of hidden units.
        :param num_outputs: (int) Number of output units.
        :param loss_fn: (str) String name of loss function to be used during training and testing.
        :param optmizer: (str) String (name of optimizer) or optimizer instance.
        :param metrics: (str) Metric to be use when evaluating the model during training and testing.
        :param hidden_layer_activation_function: (str) String name of the activation function used by
                                                        the hidden layer.
        :param output_layer_act_func: (str) String name of the activation function used by the output
                                            layer.
        """

        super().__init__()

        self.model = None
        self.frame_size = frame_size
        self.hidden_size = hidden_size
        self.output_size = num_outputs
        self.loss_fn = loss_fn
        self.optimizer = optmizer
        self.metrics = metrics
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_act_func

    def build(self) -> Model:
        """
        Builds and compiles model architecture.

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

    def __init__(self, input_shape: tuple, hidden_size: int = 10, num_outputs: int = 1,
                 loss_fn: str = "mean_squared_error", optmizer: str = "adam", metrics: str = "accuracy",
                 hidden_layer_activation_function: str = "relu", output_layer_act_func: str = "linear"):
        """
        Initialization of Variables.

        :param input_shape: (tuple) Three dimensional tuple explaining the structure of the windowed
                                    data. Ex; (No_of_samples, Time_steps, No_of_features).
        :param hidden_size: (int) Number of hidden units.
        :param num_outputs: (int) Number of output units.
        :param loss_fn: (str) String name of loss function to be used during training and testing.
        :param optmizer: (str) String (name of optimizer) or optimizer instance.
        :param metrics: (str) Metric to be use when evaluating the model during training and testing.
        :param hidden_layer_activation_function: (str) String name of the activation function used by
                                                        the hidden layer.
        :param output_layer_act_func: (str) String name of the activation function used by the output
                                            layer.
        """

        super().__init__()

        self.model = None
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_size = num_outputs
        self.loss_fn = loss_fn
        self.optimizer = optmizer
        self.metrics = metrics
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_act_func

    def build(self) -> Model:
        """
        Builds and compiles model architecture.

        :return: (Model)
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

    def __init__(self, frame_size: int, hidden_size: int = 2, num_outputs: int = 1,
                 loss_fn: str = "mean_squared_error", optmizer: str = "sgd", metrics: str = "accuracy",
                 hidden_layer_activation_function: str = "linear", output_layer_act_func: str = "sigmoid"):
        """
        Initialization of variables.

        :param frame_size: (int) The size of the input dataset.
        :param hidden_size: (int) Number of hidden units.
        :param num_outputs: (int) Number of output units.
        :param loss_fn: (str) String name of loss function to be used during training and testing.
        :param optmizer: (str) String (name of optimizer) or optimizer instance.
        :param metrics: (str) Metric to be use when evaluating the model during training and testing.
        :param hidden_layer_activation_function: (str) String name of the activation function used by
                                                        the hidden layer.
        :param output_layer_act_func: (str) String name of the activation function used by the output
                                            layer.
        """

        super().__init__()

        self.model = None
        self.frame_size = frame_size
        self.hidden_size = hidden_size
        self.output_size = num_outputs
        self.loss_fn = loss_fn
        self.optimizer = optmizer
        self.metrics = metrics
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_act_func

    def build(self) -> Model:
        """
        Builds and compiles model architecture.

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
    def _pi_this(tensor: tf.Tensor) -> tf.Tensor:
        """
        Computes the product of elements across 'axis=1' of the input tensor and
        will return the reduced version of the tensor.

        :param tensor: (tf.Tensor) Weights from the hidden layer.
        :return: (tf.Tensor) Product of input tensor.
        """

        prod = tf.math.reduce_prod(tensor, keepdims=True, axis=1)

        return prod
