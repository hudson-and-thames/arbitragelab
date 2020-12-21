# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Regressor Committee
"""

from sklearn.ensemble import VotingRegressor
from keras.wrappers.scikit_learn import KerasRegressor

from arbitragelab.ml_approach import neural_networks

# pylint: disable=W0212

class RegressorCommittee:
    """
    Regressor Committee
    """

    def __init__(self, regressor_params, regressor_class='MultiLayerPerceptron', num_committee=10, epochs=20, verbose=True):
        """
        Initializes Variables.
        """

        self.regressor_params = regressor_params
        self.regressor_class = regressor_class
        self.num_committee = num_committee
        self.epochs = epochs
        self.verbose = verbose
        self.rvoter = None

    def fit(self, xtrain, ytrain):
        """
        Fits the member models, then the voting object.
        """

        committee_members = []

        for nth_member in range(self.num_committee):
            class_ = getattr(neural_networks, self.regressor_class)
            regressor = class_(**self.regressor_params)

            r_estimator = KerasRegressor(build_fn=getattr(regressor, 'build'),
                                         epochs=self.epochs, batch_size=10, verbose=self.verbose)

            r_estimator._estimator_type = 'regressor'

            committee_members.append(('reg' + str(nth_member), r_estimator))


        rvoter = VotingRegressor(committee_members)

        rvoter.fit(xtrain, ytrain)

        self.rvoter = rvoter

        return self

    def predict(self, xtest):
        """
        Collects results from all the committee members and returns
        average result.
        """

        return self.rvoter.predict(xtest)
