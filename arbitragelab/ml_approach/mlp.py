# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module implements the Multi-Layer Perceptron.
"""

from keras.models import Model
from keras.layers.core import Dense
from keras.layers import Input

from arbitragelab.ml_approach.base import BaseNeuralNetwork


class MultiLayerPerceptron(BaseNeuralNetwork):
    """
    Multi-Layer Perceptron implementation.

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

        hidden_layer = Dense(
            self.hidden_size, activation=self.hidden_layer_activation_function)(input_layer)

        output_layer = Dense(
            self.output_size, activation=self.output_layer_activation_function)(hidden_layer)

        model = Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(loss=self.loss_fn, optimizer=self.optimizer,
                      metrics=self.metrics)

        self.model = model

        return model
