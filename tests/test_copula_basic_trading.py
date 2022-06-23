# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Unit tests for copula_strategy_basic, and additional features of copula_generate.
"""
# pylint: disable = invalid-name, protected-access, too-many-locals, unsubscriptable-object, too-many-statements, undefined-variable

import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import arbitragelab.copula_approach.archimedean as coparc
import arbitragelab.copula_approach.elliptical as copeli
from arbitragelab.copula_approach import construct_ecdf_lin


class TestBasicCopulaStrategy(unittest.TestCase):
    """
    Test the BasicCopulaStrategy class.
    """

    def setUp(self):
        """
        Get the correct directory and data.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + "/test_data/BKD_ESC_2009_2011.csv"
        self.stocks = pd.read_csv(data_path, parse_dates=True, index_col="Date")

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
        cop = coparc.Frank(theta=15)
        expected_densities = [1.499551e+01, 1.499551e+01, 4.589913e-06, 4.589913e-06, 3.700169e-02, 3.700169e-02,
                              3.754150e+00]
        densities = [cop.get_cop_density(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_densities, densities, decimal=4)

        # Check for N14
        cop = coparc.N14(theta=5)
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
        cop = coparc.Frank(theta=15)
        expected_cop_evals = [1.4997754984537027e-09, 0.9999800014802172, 9.999999999541836e-06, 9.999999999541836e-06,
                              0.2998385964795436, 0.2998385964795436, 0.4538270500610275]
        cop_evals = [cop.get_cop_eval(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_cop_evals, cop_evals, decimal=4)

        # Check for N14
        cop = coparc.N14(theta=5)
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
        cop = coparc.Frank(theta=15)
        expected_condi_probs = [0.0001499663031859714, 0.999850033362625, 0.9999999999541043, 4.589568752277862e-11,
                                0.0024452891307218463, 0.9975547108692793, 0.5000000000000212]
        condi_probs = [cop.get_condi_prob(u, v) for (u, v) in zip(us, vs)]
        np.testing.assert_array_almost_equal(expected_condi_probs, condi_probs, decimal=4)

        # Check for N14
        cop = coparc.N14(theta=5)
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
        gumbel = coparc.Gumbel(theta=theta)
        frank = coparc.Frank(theta=theta)
        clayton = coparc.Clayton(theta=theta)
        joe = coparc.Joe(theta=theta)
        n13 = coparc.N13(theta=theta)
        n14 = coparc.N14(theta=theta)
        gaussian = copeli.GaussianCopula(cov=cov)
        student = copeli.StudentCopula(cov=cov, nu=nu)

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

        for key in axs:
            self.assertEqual(str(type(axs[key])), "<class 'matplotlib.axes._subplots.AxesSubplot'>")
