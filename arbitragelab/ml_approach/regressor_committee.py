# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Regressor Committee.
"""

import pandas as pd
from sklearn.ensemble import VotingRegressor
from keras.wrappers.scikit_learn import KerasRegressor

from arbitragelab.ml_approach import neural_networks

# pylint: disable=W0212

class RegressorCommittee:
    """
    Regressor Committee implementation, based on the Sklearn VotingRegressor.
    """

    def __init__(self, regressor_params: dict, regressor_class: str = 'MultiLayerPerceptron',
                 num_committee: int = 10, epochs: int = 20, verbose: bool = True):
        """
        Initializes Variables.

        :param regressor_params: (dict) Any acceptable Keras model params.
        :param regressor_class: (str) Any class from the 'neural_networks' namespace in arbitragelab.
        :param num_committee: (int) Number of members in the voting committee.
        :param epochs: (int) Number of epochs per member.
        :param verbose: (bool) Print debug information.
        """

        self.regressor_params = regressor_params
        self.regressor_class = regressor_class
        self.num_committee = num_committee
        self.epochs = epochs
        self.verbose = verbose
        self.rvoter = None

    def fit(self, xtrain: pd.DataFrame, ytrain: pd.DataFrame):
        """
        Fits the member models, then the voting object.

        :param xtrain: (pd.DataFrame)
        :param ytrain: (pd.DataFrame)
        """

        committee_members = []

        for nth_member in range(self.num_committee):
            # Dynamically initialize the Neural Network class using the
            # 'getattr' method.
            class_ = getattr(neural_networks, self.regressor_class)
            regressor = class_(**self.regressor_params)

            # KerasRegressor is an sklearn wrapper that is initialized
            # with a build_fn which returns a compiled keras model.
            r_estimator = KerasRegressor(build_fn=getattr(regressor, 'build'),
                                         epochs=self.epochs, batch_size=10, verbose=self.verbose)

            # The KerasRegressor implementation fails to have this
            # constant set, which for newer versions of sklearn is
            # mandatory to have.
            r_estimator._estimator_type = 'regressor'

            committee_members.append(('reg' + str(nth_member), r_estimator))

        # Initialize VotingRegressor with all the generated member instances.
        rvoter = VotingRegressor(committee_members)

        rvoter.fit(xtrain, ytrain)

        self.rvoter = rvoter

        return self

    def predict(self, xtest: pd.DataFrame):
        """
        Collects results from all the committee members and returns
        average result.

        :param xtest: (pd.DataFrame)
        :return: (pd.DataFrame) Model predictions.
        """

        return self.rvoter.predict(xtest)
