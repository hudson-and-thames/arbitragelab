"""
Unit tests for copula functions.
"""
# pylint: disable = invalid-name,  protected-access, too-many-locals, unexpected-keyword-arg, too-many-public-methods

import os
import unittest
import warnings
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss

from arbitragelab.copula_approach.archimedean import Clayton, Frank, Gumbel, Joe, N14, N13
from arbitragelab.copula_approach.elliptical import GaussianCopula, StudentCopula, fit_nu_for_t_copula
from arbitragelab.copula_approach import aic, sic, hqic, construct_ecdf_lin, find_marginal_cdf, \
    fit_copula_to_empirical_data


class TestCopulas(unittest.TestCase):
    """
    Test each copula class, calculations.
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

    @staticmethod
    def cum_log_return(price_series: np.array, start: float = None) -> np.array:
        """
        Convert a price time series to cumulative log return.
        clr[i] = log(S[i]/S[0]) = log(S[i]) - log(S[0]).

        :param price_series: (np.array) 1D price time series.
        :param start: (float) Initial price. Default to the starting element of price_series.
        :return: (np.array) 1D cumulative log return series.
        """

        if start is None:
            start = price_series[0]
        log_start = np.log(start)

        # Natural log of price series
        log_prices = np.log(price_series)
        # Calculate cumulative log return
        clr = np.array([log_price_now - log_start for log_price_now in log_prices])

        return clr

    def test_gumbel(self):
        """
        Test gumbel copula class.
        """

        cop = Gumbel(theta=2)
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
        sample_pairs = cop.sample(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.sample()

        def _Kc(w: float, theta: float):
            return w * (1 - np.log(w) / theta)

        result = cop._generate_one_pair(0, 0, 3, Kc=_Kc)
        expected = np.array([1, 1e10])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

        # Test plot methods
        ax = cop.plot_scatter(200)
        self.assertTrue(isinstance(ax, Axes))

        fig = cop.plot_pdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_pdf('contour')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('contour')
        self.assertTrue(isinstance(fig, Figure))

    def test_frank(self):
        """
        Test Frank copula class.
        """

        cop = Frank(theta=10)
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
        sample_pairs = cop.sample(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.sample()

        # Test plot methods
        ax = cop.plot_scatter(200)
        self.assertTrue(isinstance(ax, Axes))

        fig = cop.plot_pdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_pdf('contour')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('contour')
        self.assertTrue(isinstance(fig, Figure))

    def test_clayton(self):
        """
        Test Clayton copula class.
        """

        cop = Clayton(theta=2)
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
        sample_pairs = cop.sample(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.sample()

        # Test plot methods
        ax = cop.plot_scatter(200)
        self.assertTrue(isinstance(ax, Axes))

        fig = cop.plot_pdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_pdf('contour')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('contour')
        self.assertTrue(isinstance(fig, Figure))

    def test_joe(self):
        """
        Test Joe copula class.
        """

        cop = Joe(theta=6)
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
        sample_pairs = cop.sample(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.sample()

        def _Kc(w: float, theta: float):
            return w - 1 / theta * (
                    (np.log(1 - (1 - w) ** theta)) * (1 - (1 - w) ** theta)
                    / ((1 - w) ** (theta - 1)))

        result = cop._generate_one_pair(0, 0, 3, Kc=_Kc)
        expected = np.array([1, 0])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

        # Test plot methods
        ax = cop.plot_scatter(200)
        self.assertTrue(isinstance(ax, Axes))

        fig = cop.plot_pdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_pdf('contour')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('contour')
        self.assertTrue(isinstance(fig, Figure))


    def test_n13(self):
        """
        Test N13 Copula class.
        """
        cop = N13(theta=3)
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
        sample_pairs = cop.sample(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.sample()

        def _Kc(w: float, theta: float):
            return w + 1 / theta * (
                    w - w * np.power((1 - np.log(w)), 1 - theta) - w * np.log(w))

        result = cop._generate_one_pair(0, 0, 3, Kc=_Kc)
        expected = np.array([1, 0])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

        # Test plot methods
        ax = cop.plot_scatter(200)
        self.assertTrue(isinstance(ax, Axes))

        fig = cop.plot_pdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_pdf('contour')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('contour')
        self.assertTrue(isinstance(fig, Figure))

    def test_n14(self):
        """
        Test N14 Copula class.
        """

        cop = N14(theta=3)
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
        sample_pairs = cop.sample(unif_vec=unif_vec)
        self.assertEqual(str(type(sample_pairs)), "<class 'numpy.ndarray'>")
        self.assertEqual(sample_pairs.shape, (100, 2))

        # Check edge cases
        with self.assertRaises(ValueError):
            cop.sample()

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
        cop = GaussianCopula(cov=cov)
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
        self.assertEqual(str(type(cop.sample(num=1))), "<class 'numpy.ndarray'>")

        # Test plot methods
        ax = cop.plot_scatter(200)
        self.assertTrue(isinstance(ax, Axes))

        fig = cop.plot_pdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_pdf('contour')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('contour')
        self.assertTrue(isinstance(fig, Figure))

    def test_student(self):
        """
        Test Student copula class (Student-t).
        """

        cov = [[2, 0.5], [0.5, 2]]
        nu = 5
        cop = StudentCopula(cov=cov, nu=nu)
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
        u = np.array([0.1, 0.21, 0.5, 0.8])
        v = np.array([0.01, 0.25, 0.4, 0.7])
        new_cop = StudentCopula(nu=5)
        new_cop.fit(u, v)
        ll = new_cop.get_log_likelihood_sum(u=u,
                                            v=v)
        self.assertAlmostEqual(ll, 2.1357117471178584, delta=1e-5)

        # Check edge cases
        self.assertEqual(str(type(cop.sample(num=1))), "<class 'numpy.ndarray'>")

        # Test plot methods
        ax = cop.plot_scatter(200)
        self.assertTrue(isinstance(ax, Axes))

        fig = cop.plot_pdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_pdf('contour')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('3d')
        self.assertTrue(isinstance(fig, Figure))

        fig = cop.plot_cdf('contour')
        self.assertTrue(isinstance(fig, Figure))

    def test_plot_cdf(self):
        """
        Test plotting cdf.
        """

        # Creating copula
        cov = [[2, 0.5], [0.5, 2]]
        nu = 5
        cop = StudentCopula(cov=cov, nu=nu)

        # Test plotting with given levels
        levels = [0.01, 0.04, 0.08, 0.12, 0.5, 0.7]
        plot = cop.plot_cdf('contour', levels=levels)
        self.assertTrue(isinstance(plot, Figure))

        # Test error when wrong option is used
        with self.assertRaises(ValueError):
            cop.plot_cdf('4d')

    def test_plot_pdf(self):
        """
        Test plotting pdf.
        """

        # Creating copula
        cov = [[2, 0.5], [0.5, 2]]
        nu = 5
        cop = StudentCopula(cov=cov, nu=nu)

        # Test plotting with given levels
        levels = [0.01, 0.04, 0.08, 0.12, 0.5, 0.7]
        plot = cop.plot_pdf('contour', levels=levels)
        self.assertTrue(isinstance(plot, Figure))

        # Test error when wrong option is used
        with self.assertRaises(ValueError):
            cop.plot_pdf('4d')

    def test_random_gen_kendall_tau(self):
        """
        Test the random pairs generated by each copula has expected Kendall's tau
        """

        # Create copulas
        cop_gumbel = Gumbel(theta=3)
        cop_frank = Frank(theta=3)
        cop_clayton = Clayton(theta=3)
        cop_joe = Joe(theta=3)
        cop_n13 = N13(theta=3)
        cop_n14 = N14(theta=3)
        cov = [[1, 0.5], [0.5, 1]]
        cop_gaussian = GaussianCopula(cov=cov)
        cop_t = StudentCopula(cov=cov, nu=3)

        copulas = [cop_gumbel, cop_frank, cop_clayton, cop_joe, cop_n13, cop_n14, cop_gaussian, cop_t]
        copula_names = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14', 'Gaussian', 'Student']
        archimedeans = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14']
        elliptics = ['Gaussian', 'Student']
        num_of_cop = len(copulas)

        # Generate pairs
        np.random.seed(724)
        sample_pairs = {}
        for i in range(num_of_cop):
            sample_pairs[copula_names[i]] = copulas[i].sample(num=4000)

        # Calculate kendall's tau from generated data
        kendalls_taus = {}
        for i in range(num_of_cop):
            x = sample_pairs[copula_names[i]][:, 0]
            y = sample_pairs[copula_names[i]][:, 1]
            kendalls_taus[copula_names[i]] = ss.kendalltau(x, y)[0]

        # Calculate theta (or correlation) for each copula based on tau
        thetas = {}
        for i in range(num_of_cop):
            thetas[copula_names[i]] = copulas[i].theta_hat(kendalls_taus[copula_names[i]])

        # Compare results for copulas
        for i in range(num_of_cop):
            if copula_names[i] in archimedeans:
                self.assertAlmostEqual(3, thetas[copula_names[i]], delta=0.3)  # Compare theta
                continue
            if copula_names[i] in elliptics:
                self.assertAlmostEqual(0.5, thetas[copula_names[i]], delta=0.05)  # Compare rho (corr)

    @staticmethod
    def test_series_condi_prob():
        """
        Test calculating the conditional probabilities of a series.
        """

        # Expected value
        expected_probs = np.array([[0.00198373, 0.00198373],
                                   [0.5, 0.5],
                                   [0.998016, 0.998016]])
        # Initiate a Gaussian copula to test
        cov = [[2, 0.5], [0.5, 2]]
        gauss_cop = GaussianCopula(cov=cov)
        s1 = np.linspace(0.0001, 1 - 0.0001, 3)  # Assume those are marginal cumulative densities already.
        s2 = np.linspace(0.0001, 1 - 0.0001, 3)
        cdf1 = lambda x: x  # Use identity mapping for cumulative density
        cdf2 = lambda x: x
        prob_series = []
        for u, v in zip(s1, s2):
            prob_series.append([gauss_cop.get_condi_prob(cdf1(u), cdf2(v)), gauss_cop.get_condi_prob(cdf2(v), cdf1(u))])
        np.testing.assert_array_almost_equal(prob_series, expected_probs, decimal=6)

    def test_ICs(self):
        """
        Test three information criterions.
        """

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

        cdf1 = find_marginal_cdf(data, prob_floor=probflr, prob_cap=probcap)
        self.assertAlmostEqual(cdf1(-1), probflr, delta=1e-5)
        self.assertAlmostEqual(cdf1(2), probcap, delta=1e-5)

        cdf2 = find_marginal_cdf(data, prob_floor=probflr, prob_cap=probcap, empirical=False)
        self.assertIsNone(cdf2)

    def test_ml_theta_hat(self):
        """
        Test max likelihood fit of theta hat for each copula.
        """

        # Change price to cumulative log return. Here we fit the whole set.
        BKD_clr = self.cum_log_return(self.BKD_series)
        ESC_clr = self.cum_log_return(self.ESC_series)
        ecdf_x = construct_ecdf_lin(BKD_clr)
        ecdf_y = construct_ecdf_lin(ESC_clr)

        # Fit through the copulas using theta_hat as its parameter
        copulas = [Gumbel, Clayton, Frank, Joe, N13, N14, GaussianCopula, StudentCopula]
        theta_hats = []

        for cop in copulas:
            if cop == GaussianCopula:
                # These copulas can't be fit on returns! Only on [0,1] data
                theta_hats.append(cop().fit(ecdf_x(BKD_clr), ecdf_y(ESC_clr)))
            elif cop == StudentCopula:
                fitted_nu = fit_nu_for_t_copula(ecdf_x(BKD_clr), ecdf_y(ESC_clr), nu_tol=0.05)
                theta_hats.append(cop(nu=fitted_nu).fit(ecdf_x(BKD_clr), ecdf_y(ESC_clr)))
            else:
                theta_hats.append(
                    cop().fit(BKD_clr, ESC_clr))  # Using copula in this way is wrong! Use pseudo-observations

        # Expected values
        expected_theta = np.array([4.823917032678924, 7.6478340653578485, 17.479858671919537, 8.416268109560686,
                                   13.006445455285089, 4.323917032678924, 0.9431138949207484, 0.9471157685912241])

        np.testing.assert_array_almost_equal(theta_hats, expected_theta, decimal=3)

    def test_fit_copula(self):
        """
        Test  fit_copula_to_empirical_data for each copula.
        """

        # Change price to cumulative log return. Here we fit the whole set
        BKD_clr = self.cum_log_return(self.BKD_series)
        ESC_clr = self.cum_log_return(self.ESC_series)

        # Fit through the copulas and the last one we do not update
        copulas = [Gumbel, Clayton, Frank, Joe, N13, N14, GaussianCopula, StudentCopula]
        aics = dict()

        for cop in copulas:
            result_dict, _, _, _ = fit_copula_to_empirical_data(x=BKD_clr, y=ESC_clr, copula=cop)
            aics[result_dict['Copula Name']] = result_dict['AIC']

        expeced_aics = {'Gumbel': -1996.8584204971112, 'Clayton': -1982.1106036413414,
                        'Frank': -2023.0991514138464, 'Joe': -1139.896265173598,
                        'N13': -2211.6295423299603, 'N14': -2111.9831835080827,
                        'Gaussian': -2211.4486204860873, 'Student': -2275.069087841567}

        for key in aics:
            self.assertAlmostEqual(aics[key], expeced_aics[key], delta=1)

    @staticmethod
    def test_construct_ecdf_lin():
        """
        Testing the construct_ecdf_lin() method.
        """

        # Create sample data frame and compute the percentile
        data = {'col1': [0, 1, 2, 3, 4, 5], 'col2': [0, 2, 4, 6, np.nan, 10], 'col3': [np.nan, 2, 4, 6, 8, 10]}
        quantile_dict = {k: construct_ecdf_lin(v) for k, v in data.items()}
        # Expected result
        expected = {'col1': [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1],
                    'col2': [1 / 5, 2 / 5, 3 / 5, 4 / 5, np.nan, 5/5],
                    'col3': [np.nan, 1 / 5, 2 / 5, 3 / 5, 4 / 5, 5/5]}
        for col, values in data.items():
            np.testing.assert_array_almost_equal(quantile_dict[col](values), expected[col], decimal=4)

        # Checking the cdfs
        test_input = [-100, -1, 1.5, 2, 3, 10, np.nan]
        expec_qt1 = [0.1667, 0.3333, 0.5, 0.66667, 0.83333, 1, np.nan]
        np.testing.assert_array_almost_equal(expec_qt1, construct_ecdf_lin(test_input)(test_input), decimal=4)

    @staticmethod
    def test_get_cop_density_abs_class_method():
        """
        Testing the get_cop_density method in the Copula abstract class.
        """

        # u and v's for checking copula density
        us = [0, 1, 1, 0, 0.3, 0.7, 0.5]
        vs = [0, 1, 0, 1, 0.7, 0.3, 0.5]

        # Check for Frank
        cop = Frank(theta=15)
        expected_densities = [1.499551e+01, 1.499551e+01, 4.589913e-06, 4.589913e-06, 3.700169e-02, 3.700169e-02,
                              3.754150e+00]
        densities = [cop.get_cop_density(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_densities, densities, decimal=4)

        # Check for N14
        cop = N14(theta=5)
        expected_densities = [28447.02084968514, 114870.71560409002, 5.094257802793674e-28, 5.094257802793674e-28,
                              0.03230000161691527, 0.03230000161691527, 3.8585955174356292]
        densities = [cop.get_cop_density(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_densities, densities, decimal=4)

    @staticmethod
    def test_get_cop_eval_abs_class_method():
        """
        Testing the get_cop_eval method in the Copula abstract class.
        """

        # u and v's for checking copula density
        us = [0, 1, 1, 0, 0.3, 0.7, 0.5]
        vs = [0, 1, 0, 1, 0.7, 0.3, 0.5]

        # Check for Frank
        cop = Frank(theta=15)
        expected_cop_evals = [1.4997754984537027e-09, 0.9999800014802172, 9.999999999541836e-06, 9.999999999541836e-06,
                              0.2998385964795436, 0.2998385964795436, 0.4538270500610275]
        cop_evals = [cop.get_cop_eval(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_cop_evals, cop_evals, decimal=4)

        # Check for N14
        cop = N14(theta=5)
        expected_cop_evals = [5.3365811321695815e-06, 0.9999885130266996, 9.999999999999982e-06, 9.999999999999982e-06,
                              0.2999052240847225, 0.2999052240847225, 0.4545364421835644]
        cop_evals = [cop.get_cop_eval(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_cop_evals, cop_evals, decimal=4)

    @staticmethod
    def test_get_condi_prob_abs_class_method():
        """
        Testing the get_cop_eval method in the Copula abstract class.
        """

        # u and v's for checking copula density
        us = [0, 1, 1, 0, 0.3, 0.7, 0.5]
        vs = [0, 1, 0, 1, 0.7, 0.3, 0.5]

        # Check for Frank
        cop = Frank(theta=15)
        expected_condi_probs = [0.0001499663031859714, 0.999850033362625, 0.9999999999541043, 4.589568752277862e-11,
                                0.0024452891307218463, 0.9975547108692793, 0.5000000000000212]
        condi_probs = [cop.get_condi_prob(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_condi_probs, condi_probs, decimal=4)

        # Check for N14
        cop = N14(theta=5)
        expected_condi_probs = [0.2703284430766092, 0.5743481526129369, 0.9999999999999973, 2.4387404375289055e-33,
                                0.0019649735178081406, 0.9984409781303989, 0.5122647216509384]
        condi_probs = [cop.get_condi_prob(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_condi_probs, condi_probs, decimal=4)

    def test_plot_abs_class_method(self):
        """
        Testing the plot method in the Copula abstract class.
        """

        cov = [[1, 0.5], [0.5, 1]]
        nu = 4
        theta = 5
        gumbel = Gumbel(theta=theta)
        frank = Frank(theta=theta)
        clayton = Clayton(theta=theta)
        joe = Joe(theta=theta)
        n13 = N13(theta=theta)
        n14 = N14(theta=theta)
        gaussian = GaussianCopula(cov=cov)
        student = StudentCopula(cov=cov, nu=nu)

        # Initiate without an axes
        axs = dict()
        axs['Gumbel'] = gumbel.plot_scatter(200)
        axs['Frank'] = frank.plot_scatter(200)
        axs['Clayton'] = clayton.plot_scatter(200)
        axs['Joe'] = joe.plot_scatter(200)
        axs['N13'] = n13.plot_scatter(200)
        axs['N14'] = n14.plot_scatter(200)
        axs['Gaussian'] = gaussian.plot_scatter(200)
        axs['Student'] = student.plot_scatter(200)
        plt.close()

        for plot in axs.values():
            self.assertTrue(isinstance(plot, Axes))
