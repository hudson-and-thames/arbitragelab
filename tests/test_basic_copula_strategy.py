# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Unit tests for basic copula strategy.
"""
# pylint: disable = invalid-name,  protected-access, too-many-locals

import os
import unittest
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
from arbitragelab.copula_approach import copula_generate, copula_strategy, copula_calculation


class TestCopulaStrategy(unittest.TestCase):
    """
    Test each copula class, calculations and strategy.
    """

    def setUp(self):
        """
        Get the correct directory.
        """

        project_path = os.path.dirname(__file__)
        self.data_path = project_path + r'/test_data'

        pair_prices = pd.read_csv(self.data_path + r'/BKD_ESC_2009_2011.csv')
        self.BKD_series = pair_prices['BKD'].to_numpy()
        self.ESC_series = pair_prices['ESC'].to_numpy()
        warnings.simplefilter('ignore')

    def test_gumbel(self):
        """
        Test gumbel copula class.
        """

        cop = copula_generate.Gumbel(theta=2)
        # Check describe
        descr = cop.describe()
        self.assertEqual(descr['Descriptive Name'], 'Bivariate Gumbel Copula')
        self.assertEqual(descr['Class Name'], 'Gumbel')
        self.assertEqual(descr['theta'], 2)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7))
        self.assertEqual(cop.C(0.7, 1), cop.C(1, 0.7))
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertEqual(cop.C(0.7, 1), 0.7)
        self.assertAlmostEqual(cop.C(0.7, 0.5), 0.458621, delta=1e-4)

        # Check copula joint probability density c(U=u,V=v)
        self.assertEqual(cop.c(0.5, 0.7), cop.c(0.7, 0.5))
        self.assertAlmostEqual(cop.c(0.5, 0.7), 1.21699, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.299774, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(0.5), 2, delta=1e-4)

        # Check side-loading pairs generation
        unif_vec = np.random.uniform(low=0, high=1, size=(100, 2))
        sample_pairs = cop.generate_pairs(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.generate_pairs()

        def _Kc(w: float, theta: float):
            return w * (1 - np.log(w) / theta)

        result = cop._generate_one_pair(0, 0, 3, Kc=_Kc)
        expected = np.array([1, 1e10])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_frank(self):
        """
        Test Frank copula class.
        """

        cop = copula_generate.Frank(theta=10)
        # Check describe
        descr = cop.describe()
        self.assertEqual(descr['Descriptive Name'], 'Bivariate Frank Copula')
        self.assertEqual(descr['Class Name'], 'Frank')
        self.assertEqual(descr['theta'], 10)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), 0.7, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.5, 0.7), 0.487979, delta=1e-4)

        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7, 0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.5, 0.7), 1.06418, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.119203, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(0.21389456921960), 2, delta=1e-4)

        # Check side-loading pairs generation
        unif_vec = np.random.uniform(low=0, high=1, size=(100, 2))
        sample_pairs = cop.generate_pairs(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.generate_pairs()

    def test_clayton(self):
        """
        Test Clayton copula class.
        """

        cop = copula_generate.Clayton(theta=2)
        # Check describe
        descr = cop.describe()
        self.assertEqual(descr['Descriptive Name'], 'Bivariate Clayton Copula')
        self.assertEqual(descr['Class Name'], 'Clayton')
        self.assertEqual(descr['theta'], 2)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), 0.7, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 0.5), 0.445399, delta=1e-4)

        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7, 0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.5, 0.7), 1.22649, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.257605, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(0.5), 2, delta=1e-4)

        # Check side-loading pairs generation
        unif_vec = np.random.uniform(low=0, high=1, size=(100, 2))
        sample_pairs = cop.generate_pairs(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.generate_pairs()

    def test_joe(self):
        """
        Test Joe copula class.
        """

        cop = copula_generate.Joe(theta=6)
        # Check describe
        descr = cop.describe()
        self.assertEqual(descr['Descriptive Name'], 'Bivariate Joe Copula')
        self.assertEqual(descr['Class Name'], 'Joe')
        self.assertEqual(descr['theta'], 6)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), 0.7, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.5, 0.7), 0.496244, delta=1e-4)

        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7, 0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.5, 0.7), 0.71849, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.0737336, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(0.35506593315175), 2, delta=1e-4)

        # Check side-loading pairs generation
        unif_vec = np.random.uniform(low=0, high=1, size=(100, 2))
        sample_pairs = cop.generate_pairs(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.generate_pairs()

        def _Kc(w: float, theta: float):
            return w - 1 / theta * (
                (np.log(1 - (1 - w) ** theta)) * (1 - (1 - w) ** theta)
                / ((1 - w) ** (theta - 1)))

        result = cop._generate_one_pair(0, 0, 3, Kc=_Kc)
        expected = np.array([1, 0])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_n13(self):
        """
        Test N13 Copula class.
        """
        cop = copula_generate.N13(theta=3)
        # Check describe
        descr = cop.describe()
        self.assertEqual(descr['Descriptive Name'], 'Bivariate Nelsen 13 Copula')
        self.assertEqual(descr['Class Name'], 'N13')
        self.assertEqual(descr['theta'], 3)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.3, 0.7), 0.271918, delta=1e-4)

        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7, 0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.3, 0.7), 0.770034, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.3, 0.7), 0.134891, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(0.222657233776425), 2, delta=1e-4)

        # Check side-loading pairs generation
        unif_vec = np.random.uniform(low=0, high=1, size=(100, 2))
        sample_pairs = cop.generate_pairs(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.generate_pairs()

        def _Kc(w: float, theta: float):
            return w + 1 / theta * (
                w - w * np.power((1 - np.log(w)), 1 - theta) - w * np.log(w))

        result = cop._generate_one_pair(0, 0, 3, Kc=_Kc)
        expected = np.array([1, 0])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_n14(self):
        """
        Test N14 Copula class.
        """

        cop = copula_generate.N14(theta=3)
        # Check describe
        descr = cop.describe()
        self.assertEqual(descr['Descriptive Name'], 'Bivariate Nelsen 14 Copula')
        self.assertEqual(descr['Class Name'], 'N14')
        self.assertEqual(descr['theta'], 3)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.3, 0.7), 0.298358, delta=1e-4)

        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7, 0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.3, 0.7), 0.228089, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.3, 0.7), 0.0207363, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(3 / 5), 2, delta=1e-4)

        # Check side-loading pairs generation
        unif_vec = np.random.uniform(low=0, high=1, size=(100, 2))
        sample_pairs = cop.generate_pairs(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.generate_pairs()

        def _Kc(w: float, theta: float):
            return -w * (-2 + w ** (1 / theta))

        result = cop._generate_one_pair(0, 0, 3, Kc=_Kc)
        expected = np.array([1, 0])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_gaussian(self):
        """
        Test Gaussian copula class.
        """

        cov = [[2, 0.5], [0.5, 2]]
        cop = copula_generate.Gaussian(cov=cov)
        # Check describe
        descr = cop.describe()
        self.assertEqual(descr['Descriptive Name'], 'Bivariate Gaussian Copula')
        self.assertEqual(descr['Class Name'], 'Gaussian')
        self.assertEqual(descr['cov'], cov)
        self.assertAlmostEqual(descr['rho'], 0.25, delta=1e-5)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), 0.7, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.5, 0.7), 0.384944, delta=1e-4)

        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7, 0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.5, 0.7), 1.023371665778, delta=1e-4)
        self.assertAlmostEqual(cop.c(0.6, 0.7), 1.058011636928, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.446148, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(2 * np.arcsin(0.2) / np.pi), 0.2, delta=1e-4)

        # Check edge cases
        self.assertEqual(str(type(cop.generate_pairs(num=1))), "<class 'numpy.ndarray'>")

    def test_student(self):
        """
        Test Student copula class (Student-t).
        """

        cov = [[2, 0.5], [0.5, 2]]
        nu = 5
        cop = copula_generate.Student(cov=cov, nu=nu)
        # Check describe
        descr = cop.describe()
        self.assertEqual(descr['Descriptive Name'], 'Bivariate Student-t Copula')
        self.assertEqual(descr['Class Name'], 'Student')
        self.assertEqual(descr['cov'], cov)
        self.assertEqual(descr['nu (degrees of freedom)'], nu)
        self.assertAlmostEqual(descr['rho'], 0.25, delta=1e-5)
        # More to be added here for test on C(U<=u, V<=v)

        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7, 0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.5, 0.7), 1.09150554, delta=1e-4)
        self.assertAlmostEqual(cop.c(0.6, 0.7), 1.1416005, delta=1e-4)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0, 0.2), cop.C(0.2, 0), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(1, 1), 1, delta=1e-4)
        self.assertAlmostEqual(cop.C(0, 0), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.3, 0.7), 0.23534923332657925, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.4415184293094455, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(2 * np.arcsin(0.2) / np.pi), 0.2, delta=1e-4)

        # log_ml function in ccalc
        ll, _ = copula_calculation.log_ml(x=np.array([0.1, 0.21, 0.5, 0.8]),
                                          y=np.array([0.01, 0.25, 0.4, 0.7]),
                                          copula_name='Student',
                                          nu=5)
        self.assertAlmostEqual(ll, 2.1357117471178584, delta=1e-5)

        # Check edge cases
        self.assertEqual(str(type(cop.generate_pairs(num=1))), "<class 'numpy.ndarray'>")

    def test_random_gen_kendall_tau(self):
        """
        Test the random pairs generated by each copula has expected Kendall's tau
        """

        # Create copulas.
        cop_gumbel = copula_generate.Gumbel(theta=3)
        cop_frank = copula_generate.Frank(theta=3)
        cop_clayton = copula_generate.Clayton(theta=3)
        cop_joe = copula_generate.Joe(theta=3)
        cop_n13 = copula_generate.N13(theta=3)
        cop_n14 = copula_generate.N14(theta=3)
        cov = [[1, 0.5], [0.5, 1]]
        cop_gaussian = copula_generate.Gaussian(cov=cov)
        cop_t = copula_generate.Student(cov=cov, nu=3)

        copulas = [cop_gumbel, cop_frank, cop_clayton, cop_joe, cop_n13, cop_n14, cop_gaussian, cop_t]
        copula_names = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14', 'Gaussian', 'Student']
        archimedeans = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14']
        elliptics = ['Gaussian', 'Student']
        num_of_cop = len(copulas)

        # Generate pairs.
        np.random.seed(724)
        sample_pairs = {}
        for i in range(num_of_cop):
            sample_pairs[copula_names[i]] = copulas[i].generate_pairs(num=4000)

        # Calculate kendall's tau from generated data.
        kendalls_taus = {}
        for i in range(num_of_cop):
            x = sample_pairs[copula_names[i]][:, 0]
            y = sample_pairs[copula_names[i]][:, 1]
            kendalls_taus[copula_names[i]] = ss.kendalltau(x, y)[0]

        # Calculate theta (or correlation) for each copula based on tau.
        thetas = {}
        for i in range(num_of_cop):
            thetas[copula_names[i]] = copulas[i].theta_hat(kendalls_taus[copula_names[i]])

        # Compare results for copulas.
        for i in range(num_of_cop):
            if copula_names[i] in archimedeans:
                self.assertAlmostEqual(3, thetas[copula_names[i]], delta=0.3)  # Compare theta
                continue
            if copula_names[i] in elliptics:
                self.assertAlmostEqual(0.5, thetas[copula_names[i]], delta=0.05)  # Compare rho (corr)

    def test_signal(self):
        """
        Test trading signal generation.
        """

        CS = copula_strategy.CopulaStrategy()

        # Testing long.
        new_pos = CS.get_next_position(prob_u1=0.01, prob_u2=0.99,
                                       prev_prob_u1=0.5, prev_prob_u2=0.5,
                                       current_pos=0)
        self.assertEqual(new_pos, 1)
        new_pos = CS.get_next_position(prob_u1=0.01, prob_u2=0.99,
                                       prev_prob_u1=0.49, prev_prob_u2=0.51,
                                       current_pos=1)
        self.assertEqual(new_pos, 1)

        # Testing short.
        new_pos = CS.get_next_position(prob_u1=0.99, prob_u2=0.01,
                                       prev_prob_u1=0.5, prev_prob_u2=0.5,
                                       current_pos=0)
        self.assertEqual(new_pos, -1)
        new_pos = CS.get_next_position(prob_u1=0.99, prob_u2=0.01,
                                       prev_prob_u1=0.51, prev_prob_u2=0.49,
                                       current_pos=-1)
        self.assertEqual(new_pos, -1)

        # Testing exit.
        new_pos = CS.get_next_position(prob_u1=0.3, prob_u2=0.1,
                                       prev_prob_u1=0.6, prev_prob_u2=0.1,
                                       current_pos=0)
        self.assertEqual(new_pos, 0)
        new_pos = CS.get_next_position(prob_u1=0.99, prob_u2=0.01,
                                       prev_prob_u1=0.3, prev_prob_u2=0.1,
                                       current_pos=-1)
        self.assertEqual(new_pos, 0)
        new_pos = CS.get_next_position(prob_u1=0.99, prob_u2=0.01,
                                       prev_prob_u1=0.3, prev_prob_u2=0.1,
                                       current_pos=1)
        self.assertEqual(new_pos, 0)

    def test_cum_log_return(self):
        """
        Test calculation of cumulative log return.
        """

        CS = copula_strategy.CopulaStrategy()

        # Testing a constant series.
        const_series = np.ones(100) * 10
        expected_clr = np.zeros_like(const_series)
        clr = CS.cum_log_return(const_series)
        np.testing.assert_array_almost_equal(clr, expected_clr, decimal=6)

        # Testing a exponential series, with a custom start
        exp_series = np.exp(np.linspace(start=0, stop=1, num=101))
        expected_clr = np.linspace(start=0, stop=1, num=101) + 1
        clr = CS.cum_log_return(exp_series, start=np.exp(-1))
        np.testing.assert_array_almost_equal(clr, expected_clr, decimal=6)

        # Test __init__ for CopulaStrategy
        CS_1 = copula_strategy.CopulaStrategy(position_kind=[3, 4, 5])
        self.assertEqual(CS_1.position_kind, [3, 4, 5])

    @staticmethod
    def test_series_condi_prob():
        """
        Test calculating the conditional probabilities of a seires.
        """

        # Expected value.
        expected_probs = np.array([[0.00198373, 0.00198373],
                                   [0.5, 0.5],
                                   [0.998016, 0.998016]])
        # Initiate a Gaussian copula to test.
        cov = [[2, 0.5], [0.5, 2]]
        GaussianC = copula_generate.Gaussian(cov=cov)
        CS = copula_strategy.CopulaStrategy(GaussianC)
        s1 = np.linspace(0.0001, 1 - 0.0001, 3)  # Assume those are marginal cumulative densities already.
        s2 = np.linspace(0.0001, 1 - 0.0001, 3)
        cdf1 = lambda x: x  # Use identity mapping for cumulative density.
        cdf2 = lambda x: x
        prob_series = CS.series_condi_prob(s1_series=s1, s2_series=s2, cdf1=cdf1, cdf2=cdf2)

        np.testing.assert_array_almost_equal(prob_series, expected_probs, decimal=6)

    def test_ICs(self):
        """
        Test three information criterions.
        """

        aic = copula_calculation.aic
        sic = copula_calculation.sic
        hqic = copula_calculation.hqic
        log_likelihood = 1000
        n = 200
        k = 2

        self.assertAlmostEqual(aic(log_likelihood, n, k), -1995.9390862944163, delta=1e-5)
        self.assertAlmostEqual(sic(log_likelihood, n, k), -1989.4033652669038, delta=1e-5)
        self.assertAlmostEqual(hqic(log_likelihood, n, k), -1993.330442831434, delta=1e-5)

    def test_find_marginal_cdf(self):
        """
        Test find_marginal_cdf.
        """

        # Create data
        data = np.linspace(0, 1, 101)
        probflr = 0.001
        probcap = 1 - 0.001

        cdf1 = copula_calculation.find_marginal_cdf(data, prob_floor=probflr, prob_cap=probcap)
        self.assertAlmostEqual(cdf1(-1), probflr, delta=1e-5)
        self.assertAlmostEqual(cdf1(2), probcap, delta=1e-5)

        cdf2 = copula_calculation.find_marginal_cdf(data, prob_floor=probflr, prob_cap=probcap, empirical=False)
        self.assertIsNone(cdf2)

    def test_graph_copula(self):
        """
        Test graph_copula from copula_strategy.CopulaStrategy
        """

        CS = copula_strategy.CopulaStrategy()
        cov = [[1, 0.5], [0.5, 1]]
        nu = 4
        theta = 5

        # Expect None type
        axs = {}
        axs['Gumbel'] = CS.graph_copula(copula_name='Gumbel', theta=theta)
        axs['Frank'] = CS.graph_copula(copula_name='Frank', theta=theta)
        axs['Clayton'] = CS.graph_copula(copula_name='Clayton', theta=theta)
        axs['Joe'] = CS.graph_copula(copula_name='Joe', theta=theta)
        axs['N13'] = CS.graph_copula(copula_name='N13', theta=theta)
        axs['N14'] = CS.graph_copula(copula_name='N14', theta=theta)
        axs['Gaussian'] = CS.graph_copula(copula_name='Gaussian', cov=cov)
        axs['Student'] = CS.graph_copula(copula_name='Student', cov=cov, nu=nu)
        plt.close()

        for key in axs:
            self.assertIsNone(axs[key])

        # Expect plt.ax type
        _, ax = plt.subplots()
        axs = {}
        axs['Gumbel'] = CS.graph_copula(copula_name='Gumbel', theta=theta, ax=ax)
        axs['Frank'] = CS.graph_copula(copula_name='Frank', theta=theta, ax=ax)
        axs['Clayton'] = CS.graph_copula(copula_name='Clayton', theta=theta, ax=ax)
        axs['Joe'] = CS.graph_copula(copula_name='Joe', theta=theta, ax=ax)
        axs['N13'] = CS.graph_copula(copula_name='N13', theta=theta, ax=ax)
        axs['N14'] = CS.graph_copula(copula_name='N14', theta=theta, ax=ax)
        axs['Gaussian'] = CS.graph_copula(copula_name='Gaussian', cov=cov, ax=ax)
        axs['Student'] = CS.graph_copula(copula_name='Student', cov=cov, nu=nu, ax=ax)
        plt.close()

        for key in axs:
            self.assertEqual(str(type(axs[key])), "<class 'matplotlib.axes._subplots.AxesSubplot'>")

    def test_ml_theta_hat(self):
        """
        Test max likelihood fit of theta hat for each copula.
        """

        # Change price to cumulative log return. Here we fit the whole set.
        ml_theta_hat = copula_calculation.ml_theta_hat
        CS = copula_strategy.CopulaStrategy()
        BKD_clr = CS.cum_log_return(self.BKD_series)
        ESC_clr = CS.cum_log_return(self.ESC_series)

        # Fit through the copulas using theta_hat as its parameter
        copulas = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14', 'Gaussian', 'Student']
        theta_hats = np.array(
            [ml_theta_hat(x=BKD_clr, y=ESC_clr, copula_name=name) for name in copulas])

        # Expected values.
        expected_theta = np.array([4.823917032678924, 7.6478340653578485, 17.479858671919537, 8.416268109560686,
                                   13.006445455285089, 4.323917032678924, 0.9474504200741508, 0.9474504200741508])

        np.testing.assert_array_almost_equal(theta_hats, expected_theta, decimal=6)

    def test_fit_copula(self):
        """
        Test fit_copula in CopulaStrategy for each copula.
        """

        # Change price to cumulative log return. Here we fit the whole set.
        CS = copula_strategy.CopulaStrategy()
        BKD_clr = CS.cum_log_return(self.BKD_series)
        ESC_clr = CS.cum_log_return(self.ESC_series)

        # Fit through the copulas and the last one we do not update.
        copulas = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14', 'Gaussian', 'Student']
        aics = dict()

        for name in copulas:
            result_dict, _, _, _ = CS.fit_copula(s1_series=BKD_clr, s2_series=ESC_clr, copula_name=name)
            aics[name] = result_dict['AIC']

        # Check the renew functionality. It should still be a Student-t copula internally.
        _, _, _, _ = CS.fit_copula(s1_series=BKD_clr, s2_series=ESC_clr, copula_name='Gumbel', if_renew=False)
        self.assertIsInstance(CS.copula, copula_generate.Student)

        expeced_aics = {'Gumbel': -1996.8584204971112, 'Clayton': -1982.1106036413414,
                        'Frank': -2023.0991514138464, 'Joe': -1139.896265173598,
                        'N13': -2211.6295423299603, 'N14': -2111.9831835080827,
                        'Gaussian': -2211.4486204860873, 'Student': -2275.069087841567}

        for key in aics:
            self.assertAlmostEqual(aics[key], expeced_aics[key], delta=1)

    def test_analyze_time_series(self):
        """
        Test analyze_time_series in CopulaStrategy for each copula.
        """

        CS = copula_strategy.CopulaStrategy(default_lower_threshold=0.25, default_upper_threshold=0.75)
        BKD_clr = CS.cum_log_return(self.BKD_series)
        ESC_clr = CS.cum_log_return(self.ESC_series)

        # Training testing split
        training_length = 670

        BKD_train = BKD_clr[: training_length]
        ESC_train = ESC_clr[: training_length]
        BKD_test = BKD_clr[training_length:]
        ESC_test = ESC_clr[training_length:]

        # Compare their AIC values
        copulas = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14', 'Gaussian', 'Student']

        # For each copula type, fit, then analyze
        positions_data = {}
        for name in copulas:
            if name != 'Student':
                _, _, cdf1, cdf2 = CS.fit_copula(s1_series=BKD_train, s2_series=ESC_train, copula_name=name)
                positions = CS.analyze_time_series(s1_series=BKD_test, s2_series=ESC_test,
                                                   cdf1=cdf1, cdf2=cdf2)
                positions_data[name] = positions
            else:
                _, _, cdf1, cdf2 = CS.fit_copula(s1_series=BKD_train, s2_series=ESC_train, copula_name=name)
                positions = CS.analyze_time_series(s1_series=BKD_test, s2_series=ESC_test,
                                                   cdf1=cdf1, cdf2=cdf2, start_position=0,
                                                   lower_threshold=0.25, upper_threshold=0.75)
                positions_data[name] = positions

        # Load and compare with theoretical data
        expected_positions_df = pd.read_csv(self.data_path + r'/BKD_ESC_unittest_positions.csv')
        for name in copulas:
            np.testing.assert_array_almost_equal(positions_data[name],
                                                 expected_positions_df[name].to_numpy(),
                                                 decimal=3)

    def test_ic_test(self):
        """
        Test ic_test from CopulaStrategy for each copula.
        """

        # Change price to cumulative log return. Here we fit the whole set.
        CS = copula_strategy.CopulaStrategy()
        BKD_clr = CS.cum_log_return(self.BKD_series)
        ESC_clr = CS.cum_log_return(self.ESC_series)

        # 2. Fit to every copula, and get the SIC, AIC, HQIC data from ic_test
        copulas = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14', 'Gaussian', 'Student', 'Loaded-t']
        ic_type = ['SIC', 'AIC', 'HQIC']
        ic_dict = {copula: {ic: None for ic in ic_type} for copula in copulas}
        for name in copulas[: -1]:
            _, fitted_copula, cdf1, cdf2 = CS.fit_copula(s1_series=BKD_clr, s2_series=ESC_clr, copula_name=name)
            result_dict = CS.ic_test(s1_test=BKD_clr, s2_test=ESC_clr, cdf1=cdf1, cdf2=cdf2)
            for ic in ic_type:
                ic_dict[name][ic] = result_dict[ic]

        # Test side loading functionality. The last one should be a Student-t copula
        result_dict = CS.ic_test(s1_test=BKD_clr, s2_test=ESC_clr, cdf1=cdf1, cdf2=cdf2, copula=fitted_copula)
        for ic in ic_type:
            ic_dict['Loaded-t'][ic] = result_dict[ic]

        # 3. Hard coded theoretical value.
        ic_dict_expect = {'Gumbel':
                              {'SIC': -1991.9496657125103, 'AIC': -1996.8584204971112, 'HQIC': -1994.9956755450632},
                          'Clayton':
                              {'SIC': -1977.2018488567405, 'AIC': -1982.1106036413414, 'HQIC': -1980.2478586892935},
                          'Frank':
                              {'SIC': -2018.1903966292455, 'AIC': -2023.0991514138464, 'HQIC': -2021.2364064617984},
                          'Joe':
                              {'SIC': -1134.9875103889972, 'AIC': -1139.896265173598, 'HQIC': -1138.0335202215501},
                          'N13':
                              {'SIC': -2206.720787545359, 'AIC': -2211.6295423299603, 'HQIC': -2209.766797377912},
                          'N14':
                              {'SIC': -2107.0744287234816, 'AIC': -2111.9831835080827, 'HQIC': -2110.1204385560345},
                          'Gaussian':
                              {'SIC': -2206.539865701486, 'AIC': -2211.4486204860873, 'HQIC': -2209.585875534039},
                          'Student':
                              {'SIC': -2270.160333056966, 'AIC': -2275.069087841567, 'HQIC': -2273.206342889519},
                          'Loaded-t':
                              {'SIC': -2270.160333056966, 'AIC': -2275.069087841567, 'HQIC': -2273.206342889519}}

        # 4. Check with ic_test value.
        for name in copulas:
            for ic in ic_type:
                self.assertAlmostEqual(ic_dict[name][ic], ic_dict_expect[name][ic], delta=1)
