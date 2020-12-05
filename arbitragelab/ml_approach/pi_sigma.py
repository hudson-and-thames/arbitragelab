# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This module implements the Pi Sigma Model.
"""

from keras.models import Model
from keras.layers.core import Dense, Activation, Lambda
from keras.layers import Input
import tensorflow as tf

from arbitragelab.ml_approach.base import BaseNeuralNetwork


class PiSigmaNeuralNetwork(BaseNeuralNetwork):
    """
    Pi Sigma Neural Network implementation.
    """

    def __init__(self, frame_size, num_outputs=1, loss_fn="mean_squared_error",
                 optmz="sgd", metrics="accuracy",
                 hidden_layer_activation_function="linear", output_layer_act_func="sigmoid"):
        """
        Inialization of variables.
        """

        super().__init__()

        self.model = None
        self.frame_size = frame_size
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

        second_sigma_layer = Dense(
            2, activation=self.hidden_layer_activation_function)(input_layer)

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
