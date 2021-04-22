"""
This module houses the ML Based Approaches.
"""

from arbitragelab.ml_approach.dbscan_pairs_clustering import DBSCANPairsClustering
from arbitragelab.ml_approach.tar import TAR
from arbitragelab.ml_approach.feature_expander import FeatureExpander
from arbitragelab.ml_approach.regressor_committee import RegressorCommittee
from arbitragelab.ml_approach.filters import ThresholdFilter, CorrelationFilter, VolatilityFilter
from arbitragelab.ml_approach.neural_networks import MultiLayerPerceptron, RecurrentNeuralNetwork, PiSigmaNeuralNetwork
