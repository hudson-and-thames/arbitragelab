# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Tests function of Minimum Profit Condition Optimization module:
cointegration_approach/minimum_profit_simulation.py
"""

import unittest
from copy import deepcopy

import numpy as np

from arbitragelab.cointegration_approach.minimum_profit_simulation import MinimumProfitSimulation


class TestMinimumProfitSimulation(unittest.TestCase):
    """
    Test Minimum Profit Condition Optimization module.
    """

    def setUp(self):
        """
        Set up the parameters for simulations.
        """

        np.random.seed(42)
        self.normal_price_params = {
            "ar_coeff": 0.4,
            "white_noise_var": 1,
            "constant_trend": 2.
        }
        self.normal_coint_params = {
            "ar_coeff": 0.2,
            "white_noise_var": 1.,
            "constant_trend": 5.,
            "beta": -0.2
        }
        self.default_price_params = {
            "ar_coeff": 0.1,
            "white_noise_var": 0.5,
            "constant_trend": 13.
        }
        self.default_coint_params = {
            "ar_coeff": 0.2,
            "white_noise_var": 1.,
            "constant_trend": 13.,
            "beta": -0.2
        }
        self.faulty_price_params = deepcopy(self.normal_price_params)
        self.faulty_price_params.pop("ar_coeff")
        self.faulty_coint_params = deepcopy(self.normal_coint_params)
        self.faulty_coint_params.pop("beta")

    def test_get_params(self):
        """
        Unit tests for parameter getters.
        """

        # 5 time series, each of 100 length
        simulation = MinimumProfitSimulation(5, 100)

        # Test getters
        self.assertEqual(simulation.get_price_params(), self.default_price_params)
        self.assertEqual(simulation.get_coint_params(), self.default_coint_params)

    def test_set_params1(self):
        """
        Unit tests for parameter setters.
        """

        # 5 time series, each of 100 length
        simulation = MinimumProfitSimulation(5, 100)

        # Test correct setters for price
        for key, value in self.normal_price_params.items():
            simulation.set_price_params(key, value)

        self.assertEqual(simulation.get_price_params(), self.normal_price_params)

        # Test correct setters for coint
        for key, value in self.normal_coint_params.items():
            simulation.set_coint_params(key, value)

        self.assertEqual(simulation.get_coint_params(), self.normal_coint_params)

    def test_set_params2(self):
        """
        Unit tests for parameter setters when wrong values are provided.
        """

        # 5 time series, each of 100 length
        simulation = MinimumProfitSimulation(5, 100)

        abnormal_value = ('alpha', 1.)
        self.assertRaises(KeyError, simulation.set_price_params, *abnormal_value)
        self.assertRaises(KeyError, simulation.set_coint_params, *abnormal_value)

    def test_set_params3(self):
        """
        Unit tests for parameter setters when a full set of parameters are provided as a dictionary.
        """

        # 5 time series, each of 100 length
        simulation = MinimumProfitSimulation(5, 100)

        simulation.load_params(self.normal_price_params, target='price')
        simulation.load_params(self.normal_coint_params, target='coint')
        self.assertEqual(simulation.get_price_params(), self.normal_price_params)
        self.assertEqual(simulation.get_coint_params(), self.normal_coint_params)

        self.assertRaises(ValueError, simulation.load_params, self.normal_price_params, target='spread')
        self.assertRaises(KeyError, simulation.load_params, self.faulty_price_params, target='price')

    def test_simulate_ar(self):
        """
        Unit tests for AR(1) process simulation.

        AR(1) coefficients will get compared to designated value.
        """

        print("Test simulate_ar():")
        # 50 time series, each of 250 length
        simulation = MinimumProfitSimulation(20, 250)

        simulated_manual = simulation.simulate_ar(self.normal_price_params,
                                                  use_statsmodel=False)
        simulated_statsmodel = simulation.simulate_ar(self.normal_price_params,
                                                      use_statsmodel=True)

        # Check shape: we really do have 50 time series with 250 length
        self.assertEqual(simulated_manual.shape, (250, 20))
        self.assertEqual(simulated_statsmodel.shape, (250, 20))

        # Check AR(1) coefficient
        manual_mean, manual_std = simulation.verify_ar(simulated_manual)
        stats_mean, stats_std = simulation.verify_ar(simulated_statsmodel)
        target = self.normal_price_params['ar_coeff']
        self.assertTrue(manual_mean - manual_std <= target <= manual_mean + manual_std)
        self.assertTrue(stats_mean - stats_std <= target <= stats_mean + stats_std)

        # Now check for faulty parameters
        self.assertRaises(KeyError, simulation.simulate_ar, self.faulty_price_params,
                          use_statsmodel=False)
        self.assertRaises(KeyError, simulation.simulate_ar, self.faulty_price_params,
                          use_statsmodel=True)

        # Check when only 1 series is generated
        sim_special = MinimumProfitSimulation(1, 250)
        sim_spec_manual = sim_special.simulate_ar(self.normal_price_params,
                                                  use_statsmodel=False)
        sim_spec_stats = sim_special.simulate_ar(self.normal_price_params,
                                                 use_statsmodel=True)

        self.assertEqual(sim_spec_manual.shape, (250, 1))
        self.assertEqual(sim_spec_stats.shape, (250, 1))

    def test_simulate_coint(self):
        """
        Unit tests for cointegration simulation.

        Cointegration coefficient will be compared to designated value.
        """

        print("Test simulate_coint():")
        # 50 time series, each of 250 length
        simulation = MinimumProfitSimulation(20, 250)
        simulation.load_params(self.normal_price_params, target='price')
        simulation.load_params(self.normal_coint_params, target='coint')

        manual_s1, manual_s2, manual_coint = simulation.simulate_coint(initial_price=100.,
                                                                       use_statsmodel=False)
        stats_s1, stats_s2, stats_coint = simulation.simulate_coint(initial_price=100.,
                                                                    use_statsmodel=True)

        # Check shape:
        self.assertEqual(manual_s1.shape, (250, 20))
        self.assertEqual(manual_s2.shape, (250, 20))
        self.assertEqual(manual_coint.shape, (250, 20))
        self.assertEqual(stats_s1.shape, (250, 20))
        self.assertEqual(stats_s2.shape, (250, 20))
        self.assertEqual(stats_coint.shape, (250, 20))

        # Check beta:
        beta_mean, beta_std = simulation.verify_coint(manual_s1, manual_s2)
        self.assertAlmostEqual(beta_mean, self.normal_coint_params['beta'], places=3)

        beta_mean, beta_std = simulation.verify_coint(stats_s1, stats_s2)
        self.assertAlmostEqual(beta_mean, self.normal_coint_params['beta'], places=3)

        # Check when only 1 cointegrated series pair is generated
        sim_special = MinimumProfitSimulation(1, 250)
        manual_s1, manual_s2, manual_coint = sim_special.simulate_coint(initial_price=100.,
                                                                        use_statsmodel=False)
        stats_s1, stats_s2, stats_coint = sim_special.simulate_coint(initial_price=100.,
                                                                     use_statsmodel=True)

        self.assertEqual(manual_s1.shape, (250, 1))
        self.assertEqual(manual_s2.shape, (250, 1))
        self.assertEqual(manual_coint.shape, (250, 1))
        self.assertEqual(stats_s1.shape, (250, 1))
        self.assertEqual(stats_s2.shape, (250, 1))
        self.assertEqual(stats_coint.shape, (250, 1))

    def test_verify_ar(self):
        """
        Unit test for AR(1) process verification.
        """

        print("Test verify_ar():")
        # 50 time series, each of 250 length
        simulation = MinimumProfitSimulation(20, 250)
        manual_series = simulation.simulate_ar(self.normal_price_params,
                                               use_statsmodel=False)
        stats_series = simulation.simulate_ar(self.normal_price_params,
                                              use_statsmodel=True)

        sim_ar_coeff_mean, sim_ar_coeff_std = simulation.verify_ar(manual_series)
        self.assertAlmostEqual(sim_ar_coeff_mean, 0.3774837331120529, places=4)
        self.assertAlmostEqual(sim_ar_coeff_std, 0.04989072095902113, places=4)

        sim_ar_coeff_mean, sim_ar_coeff_std = simulation.verify_ar(stats_series)
        self.assertAlmostEqual(sim_ar_coeff_mean, 0.42126390829852073, places=4)
        self.assertAlmostEqual(sim_ar_coeff_std, 0.05385757005000903, places=4)

        # 1 time series, length of 250
        sim_spec = MinimumProfitSimulation(1, 250)
        manual_series = sim_spec.simulate_ar(self.normal_price_params,
                                             use_statsmodel=False)
        stats_series = sim_spec.simulate_ar(self.normal_price_params,
                                            use_statsmodel=False)

        sim_ar_coeff_mean, sim_ar_coeff_std = sim_spec.verify_ar(manual_series)
        self.assertAlmostEqual(sim_ar_coeff_mean, 0.41788812286870847, places=4)
        self.assertIsNone(sim_ar_coeff_std)

        sim_ar_coeff_mean, sim_ar_coeff_std = sim_spec.verify_ar(stats_series)
        self.assertAlmostEqual(sim_ar_coeff_mean, 0.3602970511752997, places=4)
        self.assertIsNone(sim_ar_coeff_std)

    def test_verify_coint(self):
        """
        Unit tests for cointegration coefficient verification.
        """

        print("Test verify_coint():")
        # 50 time series, each of 250 length
        simulation = MinimumProfitSimulation(20, 250)
        simulation.load_params(self.normal_price_params, target='price')
        simulation.load_params(self.normal_coint_params, target='coint')

        manual_s1, manual_s2, _ = simulation.simulate_coint(initial_price=100.,
                                                            use_statsmodel=False)

        stats_s1, stats_s2, _ = simulation.simulate_coint(initial_price=100.,
                                                          use_statsmodel=True)

        sim_beta_coeff_mean, sim_beta_coeff_std = simulation.verify_coint(manual_s1, manual_s2)
        self.assertAlmostEqual(sim_beta_coeff_mean, self.normal_coint_params['beta'], places=3)
        self.assertAlmostEqual(sim_beta_coeff_std, 0.00033109426961471286, places=5)

        sim_beta_coeff_mean, sim_beta_coeff_std = simulation.verify_coint(stats_s1, stats_s2)
        self.assertAlmostEqual(sim_beta_coeff_mean, self.normal_coint_params['beta'], places=3)
        self.assertAlmostEqual(sim_beta_coeff_std, 0.0005001308215110697, places=5)

        # 1 time series, length of 250
        sim_spec = MinimumProfitSimulation(1, 250)
        manual_s1, manual_s2, _ = sim_spec.simulate_coint(initial_price=100.,
                                                          use_statsmodel=False)

        stats_s1, stats_s2, _ = sim_spec.simulate_coint(initial_price=100.,
                                                        use_statsmodel=True)

        sim_beta_coeff_mean, sim_beta_coeff_std = sim_spec.verify_coint(manual_s1, manual_s2)
        self.assertAlmostEqual(sim_beta_coeff_mean, self.normal_coint_params['beta'], places=4)
        self.assertEqual(sim_beta_coeff_std, None)

        sim_beta_coeff_mean, sim_beta_coeff_std = sim_spec.verify_coint(stats_s1, stats_s2)
        self.assertAlmostEqual(sim_beta_coeff_mean, self.normal_coint_params['beta'], places=4)
        self.assertEqual(sim_beta_coeff_std, None)
