# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This module implements the RNN model.
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense

from arbitragelab.ml_approach.base import BaseNeuralNetwork


class RecurrentNeuralNetwork(BaseNeuralNetwork):
    """
    Recurrent Neural Network implementation.
    """

    def __init__(self, input_shape, num_outputs=1, loss_fn="mean_squared_error",
                 optmz="adam", metrics="accuracy",
                 hidden_layer_activation_function="relu", output_layer_act_func="linear"):
        """
        Initialization of Variables.
        """

        super().__init__()

        self.model = None
        self.input_shape = input_shape
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

        hidden_layer = LSTM(50, activation=self.hidden_layer_activation_function,
                            input_shape=self.input_shape)(input_layer)

        output_layer = Dense(
            self.output_size, activation=self.output_layer_activation_function)(hidden_layer)

        model = Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(loss=self.loss_fn, optimizer=self.optimizer,
                      metrics=self.metrics)

        self.model = model

        return model
