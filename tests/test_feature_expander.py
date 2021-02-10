# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Tests Spread Modeling Feature Expander Implementation.
"""

import unittest
import numpy as np

from arbitragelab.ml_approach.feature_expander import FeatureExpander

class TestFeatureExpander(unittest.TestCase):
    """
    Tests feature expansion class.
    """

    def test_feature_expander(self):
        """
        Tests higher order term generation.
        """

        # Set the input data, which in this case is the standard XOR.
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        expanded_data = FeatureExpander(methods=['laguerre', 'power', 'chebyshev', 'legendre'],
                                        n_orders=2).fit(data).transform()

        # Check that it returned the right amount of terms.
        self.assertEqual(len(expanded_data.columns), 26)

        expanded_data = FeatureExpander(methods=['product'],
                                        n_orders=2).fit(data).transform()

        # Check that it returned the right amount of terms.
        self.assertEqual(len(expanded_data.columns), 3)