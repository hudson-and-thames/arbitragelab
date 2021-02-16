# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Unit tests for generating and fitting mixed copulas.
"""
# pylint: disable = invalid-name,  protected-access
import os
import unittest
import datetime as dt
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from arbitragelab.copula_approach import copula_generate_mixedcopula as cgmix
from arbitragelab.copula_approach import copula_generate as cg
from arbitragelab.copula_approach import copula_calculation as ccalc


class TestCopulaGenerateMixedCopula(unittest.TestCase):
    """
    Testing module copula_generate_mixedcopula.py and also methods in copula_calculation related to mixed copulas.

    Does not include fitting methods since those are tested under BasicCopulaStrategy.
    """

    def setUp(self):

        project_path = os.path.dirname(__file__)
        self.data_path = project_path + r'/test_data'

        self.pair_prices = pd.read_csv(self.data_path + r'/BKD_ESC_2009_2011.csv', index_col=0)
        formatted_dates = [dt.datetime.strptime(d, '%m/%d/%Y').date() for d in self.pair_prices.index]
        self.pair_prices.index = formatted_dates

    # At first, test new functions in copula_calculation related to mixed copulas
    def test_ccalc_scad_penalty(self):
        """
        Testing scad_penalty function in copula_calculation.
        """

        self.assertAlmostEqual(ccalc.scad_penalty(x=1, gamma=0.7, a=6), 0.6909999999999998, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_penalty(x=2, gamma=0.7, a=6), 1.2309999999999997, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_penalty(x=-1, gamma=0.7, a=6), 0.6909999999999998, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_penalty(x=0, gamma=0.7, a=6), 0, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_penalty(x=4.2, gamma=0.7, a=6), 1.7149999999999999, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_penalty(x=4.3, gamma=0.7, a=6), 1.7149999999999999, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_penalty(x=4.1, gamma=0.7, a=6), 1.7139999999999993, delta=1e-4)

    def test_ccalc_scad_derivative(self):
        """
        Testing scad_derivative function in copula_calculation.
        """

        self.assertAlmostEqual(ccalc.scad_derivative(x=1, gamma=0.7, a=6), 0.6399999999999998, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_derivative(x=2, gamma=0.7, a=6), 0.43999999999999984, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_derivative(x=-1, gamma=0.7, a=6), 0.7, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_derivative(x=0, gamma=0.7, a=6), 0.7, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_derivative(x=4.2, gamma=0.7, a=6), 0, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_derivative(x=4.3, gamma=0.7, a=6), 0, delta=1e-4)
        self.assertAlmostEqual(ccalc.scad_derivative(x=4.1, gamma=0.7, a=6), 0.019999999999999928, delta=1e-4)

    @staticmethod
    def test_ccalc_adjust_weights():
        """
        Testing adjust_weight function in copula_calculation.
        """

        result1 = ccalc.adjust_weights(weights=np.array([0, 0, 1]), threshold=0.01)
        expected_result1 = np.array([0, 0, 1])
        np.testing.assert_almost_equal(result1, expected_result1, decimal=4)

        result2 = ccalc.adjust_weights(weights=np.array([0.01, 0, 0.99]), threshold=0.01)
        expected_result2 = np.array([0, 0, 1])
        np.testing.assert_almost_equal(result2, expected_result2, decimal=4)

        result3 = ccalc.adjust_weights(weights=np.array([0.495, 0.01, 0.495]), threshold=0.02)
        expected_result3 = np.array([0.5, 0, 0.5])
        np.testing.assert_almost_equal(result3, expected_result3, decimal=4)

    # Then we test methods in copula_generate_mixedcopula mpod
    def test_ctg_init(self):
        """
        Testing CTGMixCop initiation and related methods in its abstract parent class.
        """
        # Initiate an empty CTG
        ctg_empty = cgmix.CTGMixCop()
        self.assertIsNone(ctg_empty.weights)
        self.assertIsNone(ctg_empty.cop_params)
        self.assertIsNone(ctg_empty.clayton_cop)
        self.assertIsNone(ctg_empty.t_cop)
        self.assertIsNone(ctg_empty.gumbel_cop)
        for i in range(3):
            self.assertIsNone(ctg_empty.copulas[i])

        # Initiate a CTG mixed copula with given parameters.
        cop_params = (3, 0.5, 4, 3)
        weights = (0.5, 0.25, 0.25)
        ctg = cgmix.CTGMixCop(cop_params, weights)
        # Check initiation and description.
        description = ctg.describe()
        expected_descrb = {'Descriptive Name': 'Bivariate Clayton-Student-Gumbel Mixed Copula',
                           'Class Name': 'CTGMixCop',
                           'Clayton theta': cop_params[0], 'Student rho': cop_params[1], 'Student nu': cop_params[2],
                           'Gumbel theta': cop_params[3],
                           'Clayton weight': weights[0], 'Student weight': weights[1], 'Gumbel weight': weights[2]}
        pd.testing.assert_series_equal(description, pd.Series(expected_descrb))
        self.assertIsInstance(ctg.copulas[0], cg.Clayton)
        self.assertIsInstance(ctg.copulas[1], cg.Student)
        self.assertIsInstance(ctg.copulas[2], cg.Gumbel)
        np.testing.assert_array_almost_equal(ctg.weights, weights, decimal=6)
        np.testing.assert_array_almost_equal(ctg.cop_params, cop_params, decimal=6)
        self.assertAlmostEqual(ctg.clayton_cop.theta, cop_params[0], delta=1e-4)
        self.assertAlmostEqual(ctg.t_cop.rho, cop_params[1], delta=1e-4)
        self.assertAlmostEqual(ctg.t_cop.nu, cop_params[2], delta=1e-4)
        self.assertAlmostEqual(ctg.gumbel_cop.theta, cop_params[3], delta=1e-4)

    @staticmethod
    def test_ctg_abs_class_methods():
        """
        Testing CTGMixCop mixed copula methods in its abstract parent class.
        """

        cop_params = (3, 0.5, 4, 3)
        weights = (0.5, 0.25, 0.25)
        ctg = cgmix.CTGMixCop(cop_params, weights)
        us = [0, 1, 1, 0, 0.3, 0.7, 0.5]
        vs = [0, 1, 0, 1, 0.7, 0.3, 0.5]

        # Check c(u, v), i.e., prob densities of the mixed cop.
        expected_densities = [42856.22880307941, 18356.037799997655, 97.54259654567115, 97.54259654567115,
                              0.48723388612506763, 0.48723388612506763, 1.7946192932939031]
        densities = [ctg.get_cop_density(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_densities, densities, decimal=6)

        # Check C(u, v), i.e., cumulative densities of the mixed cop.
        expected_cumdensities = [4.837929924689697e-05, 0.9998164208438649, 9.922442017680219e-05,
                                 9.927550024586962e-05, 0.2870850302069872, 0.28708499518336156, 0.3904651911354585]
        cumdensities = [ctg.get_cop_eval(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_cumdensities, cumdensities, decimal=6)

        # Check condi_cdf(u, v), i.e., conditional probs of the mixed cop.
        expected_cumdensities = [0.23816113067906816, 0.875634105685517, 0.9985132417816676,
                                 0.001486758218343033, 0.067166026946893, 0.9184718285315775, 0.47278137437820367]
        cumdensities = [ctg.get_condi_prob(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_cumdensities, cumdensities, decimal=6)

        # Test _away_from_0 method in MixedCopula parent class
        testing_x = [1, -1, 0, 1e-5, -1e-5, 1e-6, -1e-6]
        expected_result = [1, -1, 1e-5, 1e-5, -1e-5, 1e-5, -1e-5]
        testing_reslt = [ctg._away_from_0(x) for x in testing_x]
        np.testing.assert_array_almost_equal(expected_result, testing_reslt, decimal=7)

    def test_ctg_generate_pairs(self):
        """
        Testing pairs generation in CTGMixCop by matching with Kendall's tau.
        """

        np.random.seed(724)
        # 1. Choose params such that all copula components will generate pairs with Kendall's tau = 0.8.
        # Note: nu's value for Student-t copula does not matter.
        cop_params = (8, 0.9510565162951535, 3, 5)  # Calculated in association with tau = 0.8
        weights = (1/3, 1/3, 1 - 2/3)  # Equal weights.

        # 2. Generate pairs
        ctg = cgmix.CTGMixCop(cop_params, weights)
        sample_pairs = ctg.generate_pairs(num=5000)

        # 3. Calculate Kendall's tau from sample
        kendalls_taus_sample = ss.kendalltau(sample_pairs[:, 0], sample_pairs[:, 1])[0]
        self.assertAlmostEqual(kendalls_taus_sample, 0.8, delta=0.05)
        np.random.seed(None)  # Reset random seed.

    def test_cfg_init(self):
        """
        Testing CFGMixCop initiation and related methods in its abstract parent class.
        """

        # Initiate an empty CTG
        cfg_empty = cgmix.CFGMixCop()
        self.assertIsNone(cfg_empty.weights)
        self.assertIsNone(cfg_empty.cop_params)
        self.assertIsNone(cfg_empty.clayton_cop)
        self.assertIsNone(cfg_empty.frank_cop)
        self.assertIsNone(cfg_empty.gumbel_cop)
        for i in range(3):
            self.assertIsNone(cfg_empty.copulas[i])

        # Initiate a CTG mixed copula with given parameters.
        cop_params = [3, 4, 5]
        weights = [0.5, 0.25, 0.25]
        cfg = cgmix.CFGMixCop(cop_params, weights)
        # Check initiation and description.
        description = cfg.describe()
        expected_descrb = {'Descriptive Name': 'Bivariate Clayton-Frank-Gumbel Mixed Copula',
                           'Class Name': 'CFGMixCop',
                           'Clayton theta': cop_params[0], 'Frank theta': cop_params[1], 'Gumbel theta': cop_params[2],
                           'Clayton weight': weights[0], 'Frank weight': weights[1], 'Gumbel weight': weights[2]}
        pd.testing.assert_series_equal(description, pd.Series(expected_descrb))
        self.assertIsInstance(cfg.copulas[0], cg.Clayton)
        self.assertIsInstance(cfg.copulas[1], cg.Frank)
        self.assertIsInstance(cfg.copulas[2], cg.Gumbel)
        np.testing.assert_array_almost_equal(cfg.weights, weights, decimal=6)
        np.testing.assert_array_almost_equal(cfg.cop_params, cop_params, decimal=6)
        self.assertAlmostEqual(cfg.clayton_cop.theta, cop_params[0], delta=1e-4)
        self.assertAlmostEqual(cfg.frank_cop.theta, cop_params[1], delta=1e-4)
        self.assertAlmostEqual(cfg.gumbel_cop.theta, cop_params[2], delta=1e-4)

    @staticmethod
    def test_cfg_abs_class_methods():
        """
        Testing CFGMixCop mixed copula methods in its abstract parent class.
        """

        cop_params = [3, 4, 5]
        weights = [0.5, 0.25, 0.25]
        cfg = cgmix.CFGMixCop(cop_params, weights)
        us = [0, 1, 1, 0, 0.3, 0.7, 0.5]
        vs = [0, 1, 0, 1, 0.7, 0.3, 0.5]

        # Check get_cop_density(u, v), i.e., prob densities of the mixed cop.
        expected_densities = [41624.98642234784, 28720.660686591415, 0.01865885301119625, 0.018658853011196245,
                              0.3816145685458273, 0.3816145685458273, 2.1471963217627814]
        densities = [cfg.get_cop_density(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_densities, densities, decimal=6)

        # Check get_cop_eval(u, v), i.e., cumulative densities of the mixed cop.
        expected_cumdensities = [4.6050662931472644e-05, 0.9998213129311458, 9.999981335175073e-05,
                                 9.999981335175073e-05, 0.2914772991528678, 0.2914772991528678, 0.40510936419358706]
        cumdensities = [cfg.get_cop_eval(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_cumdensities, cumdensities, decimal=6)

        # Check get_condi_prob(u, v), i.e., conditional probs of the mixed cop.
        expected_cumdensities = [0.22435467527472625, 0.8935568957992956, 0.9999998134152023,
                                 1.8658479839651037e-07, 0.049245024408386615, 0.9342460479433407, 0.4707809952066607]
        cumdensities = [cfg.get_condi_prob(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_cumdensities, cumdensities, decimal=6)

        # Test _away_from_0 method in MixedCopula parent class
        testing_x = [1, -1, 0, 1e-5, -1e-5, 1e-6, -1e-6]
        expected_result = [1, -1, 1e-5, 1e-5, -1e-5, 1e-5, -1e-5]
        testing_result = [cfg._away_from_0(x) for x in testing_x]
        np.testing.assert_array_almost_equal(expected_result, testing_result, decimal=7)

    def test_cfg_generate_pairs(self):
        """
        Testing pairs generation in CFGMixCop by matching with Kendall's tau.
        """

        np.random.seed(724)
        # 1. Choose params such that all copula components will generate pairs with Kendall's tau = 0.8.
        cop_params = [8, 18.191539750851604, 5]  # Calculated in association with tau = 0.8
        weights = [1/3, 1/3, 1 - 2/3]  # Equal weights.

        # 2. Generate pairs
        cfg = cgmix.CFGMixCop(cop_params, weights)
        sample_pairs = cfg.generate_pairs(num=5000)

        # 3. Calculate Kendall's tau from sample
        kendalls_taus_sample = ss.kendalltau(sample_pairs[:, 0], sample_pairs[:, 1])[0]
        self.assertAlmostEqual(kendalls_taus_sample, 0.8, delta=0.05)
        np.random.seed(None)  # Reset random seed.

    def test_plot_abs_class_method(self):
        """
        Testing the plot method in the Copula abstract class.
        """

        rho = 0.5
        nu = 4
        theta = 5
        weights = (0.3, 0.4, 0.3)
        ctg = cgmix.CTGMixCop(cop_params=(theta, rho, nu, theta), weights=weights)
        cfg = cgmix.CFGMixCop(cop_params=(theta, theta, theta), weights=weights)

        # Initiate without an axes
        axs = dict()
        axs['CTG'] = ctg.plot(200)
        axs['CFG'] = cfg.plot(200)
        for key in axs:
            self.assertEqual(str(type(axs[key])), "<class 'matplotlib.axes._subplots.AxesSubplot'>")

        # Initiate with an axes
        _, ax = plt.subplots()
        axs = dict()
        axs['CTG'] = ctg.plot(200, ax)
        axs['CFG'] = cfg.plot(200, ax)
        plt.close()

        for key in axs:
            self.assertEqual(str(type(axs[key])), "<class 'matplotlib.axes._subplots.AxesSubplot'>")
