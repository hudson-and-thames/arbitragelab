# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Regressor Committee.
"""

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from arbitragelab.ml_approach import neural_networks

# This silencer is related to the protected access
# pylint: disable=W0212

class RegressorCommittee:
    """
    Regressor Committee implementation which basically fits N number of models
    and takes the mean value of their predictions.
    """

    def __init__(self, regressor_params: dict, regressor_class: str = 'MultiLayerPerceptron',
                 num_committee: int = 10, epochs: int = 20, patience: int = 100, verbose: bool = True):
        """
        Initializes Variables.

        :param regressor_params: (dict) Any acceptable Keras model params.
        :param regressor_class: (str) Any class from the 'neural_networks' namespace in arbitragelab.
        :param num_committee: (int) Number of members in the voting committee.
        :param epochs: (int) Number of epochs per member.
        :param patience: (int) Number of epochs to be used by the EarlyStopping class.
        :param verbose: (bool) Print debug information.
        """

        self.regressor_params = regressor_params
        self.regressor_class = regressor_class
        self.num_committee = num_committee
        self.committee_members = []
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.rvoter = None

    def fit(self, xtrain: pd.DataFrame, ytrain: pd.DataFrame, xtest: pd.DataFrame, ytest: pd.DataFrame):
        """
        Fits the member models, then the voting object.

        :param xtrain: (pd.DataFrame) Input training data.
        :param ytrain: (pd.DataFrame) Target training data.
        :param xtest: (pd.DataFrame) Input test data.
        :param ytest: (pd.DataFrame) Target test data.
        """

        committee_members = []
        idx = 0

        while idx in range(self.num_committee):
            # Dynamically initialize the Neural Network class using the
            # 'getattr' method. This lets us find an object using
            # string class name.
            class_ = getattr(neural_networks, self.regressor_class)

            # Initialize class object and build keras model using
            # given parameters.
            regressor = class_(**self.regressor_params).build()

            # Initialize Early Stopping Object to be used in the
            # model fitting.
            early_stopper = EarlyStopping(monitor='val_loss', mode='min',
                                          verbose=self.verbose, patience=self.patience)

            # Fit keras model with early stopper as callback.
            regressor.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=self.epochs,
                          verbose=self.verbose, callbacks=[early_stopper])

            # Store all model instances for later use.
            committee_members.append(regressor)

            idx = idx + 1

        self.committee_members = committee_members

        return self

    def predict(self, xtest: pd.DataFrame) -> pd.DataFrame:
        """
        Collects results from all the committee members and returns
        average result.

        :param xtest: (pd.DataFrame) Input test data.
        :return: (pd.DataFrame) Model predictions.
        """

        predictions = []

        for member in self.committee_members:
            predictions.append(member.predict(xtest))

        # Take current shape (2, len(xtest), 1) and reshape to (2, len(xtest))
        # to make it easier to get the mean of all the results.
        reshaped_predictions = np.array(predictions).reshape(len(self.committee_members), len(xtest))

        # Return Axis 0 wise mean.
        return np.mean(reshaped_predictions, axis=0)

    def plot_losses(self, figsize: tuple = (15, 5)):
        """
        Plot all individual member loss metrics.

        :param figsize: (tuple)
        """

        for idx, member in enumerate(self.committee_members):
            plt.figure(figsize=figsize)
            plt.plot(member.history.history['loss'])
            plt.plot(member.history.history['val_loss'])
            plt.legend(['Training Loss', 'Validation Loss'])
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Loss Plot of Member " + str(idx))
            plt.show()
