# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This is the base class for all the neural network implementations in this module.
"""

import matplotlib.pyplot as plt
from keras.callbacks.callbacks import History


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

        return plt.plot(self.fitted_model.history['loss'])
