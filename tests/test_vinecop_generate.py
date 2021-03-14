# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Unit tests for vinecop_generate, vinecop_strategy
"""
import unittest
import pandas as pd
import numpy as np
import os
import warnings
import pyvinecopulib as pv
import arbitragelab.copula_approach.vinecop_generate as vg
import arbitragelab.copula_approach.copula_calculation as ccalc


class TestVineCop(unittest.TestCase):
    """
    Testing vinecop_generate, vinecop_strategy. Not including the stocks selection module.
    """

    def setUp(self) -> None:
        project_path = os.path.dirname(__file__)
        data_path = project_path + "/test_data/prices_10y_SP500.csv"
        sp500_prices = pd.read_csv(data_path, index_col=0, parse_dates=True).fillna(method='ffill')
        sp500_returns = sp500_prices.pct_change().fillna(0)

        subsample = sp500_returns[['AAPL', 'ABT', 'V', 'AIZ']]
        returns_train = subsample.iloc[:400]
        returns_test = subsample.iloc[400:800]

        # Fit a c-vine copula in pv for future usage
        bicop_family = [pv.BicopFamily.bb1, pv.BicopFamily.bb6, pv.BicopFamily.bb7, pv.BicopFamily.bb8,
                        pv.BicopFamily.clayton, pv.BicopFamily.student, pv.BicopFamily.frank,
                        pv.BicopFamily.gaussian, pv.BicopFamily.gumbel, pv.BicopFamily.indep]
        cvine_structure = pv.CVineStructure(order=[1, 4, 3, 2])
        controls = pv.FitControlsVinecop(family_set=bicop_family)
        pv_cvine_cop = pv.Vinecop(structure=cvine_structure)  # Construct the C-vine copula
        data_np, _ = ccalc.to_quantile(returns_train)
        pv_cvine_cop.select(data=data_np, controls=controls)

        self.pv_cvine_cop = pv_cvine_cop

    @staticmethod
    def test_to_quantile():
        """
        Testing the to_quantile method from ccalc module.
        """

        # Create sample data frame and compute the percentile
        data = {'col1': [0, 1, 2, 3, 4, 5], 'col2': [0, 2, 4, 6, np.nan, 10], 'col3': [np.nan, 2, 4, 6, 8, 10]}
        df = pd.DataFrame.from_dict(data)
        quantile_df, cdfs = ccalc.to_quantile(df)
        # Expected result
        expected = {'col1': [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1],
                    'col2': [1 / 5, 2 / 5, 3 / 5, 4 / 5, np.nan, 1],
                    'col3': [np.nan, 1 / 5, 2 / 5, 3 / 5, 4 / 5, 1]}
        expected_df = pd.DataFrame.from_dict(expected)
        # Compare with expected result, up to 4 digits
        pd.testing.assert_frame_equal(quantile_df, expected_df, check_dtype=False, atol=4)

        # Checking the cdfs
        test_input = pd.Series([-100, -1, 1.5, 2, 3, 10, np.nan])
        quantiles_1 = test_input.map(cdfs[0])
        quantiles_2 = test_input.map(cdfs[1])
        expec_qt1 = pd.Series([0, 1e-5, 0.416667, 0.5, 0.666667, 1 - 1e-5, np.nan])
        expec_qt2 = pd.Series([1e-5, 0.1, 0.35, 0.4, 0.5, 1 - 1e-5, np.nan])
        pd.testing.assert_series_equal(expec_qt1, quantiles_1, check_dtype=False, atol=4)
        pd.testing.assert_series_equal(expec_qt2, quantiles_2, check_dtype=False, atol=4)

    def test_get_possible_cvine_structs(self) -> None:
        """
        Test CVineCop._get_possible_cvine_structs.
        """

        cvinecop = vg.CVineCop()
        structures_5 = cvinecop._get_possible_cvine_structs(data_dim=5, pv_target_idx=2)
        self.assertTrue(isinstance(structures_5, list))
        expected_structures_5 = {(2, 1, 3, 4, 5), (2, 1, 3, 5, 4), (2, 1, 4, 3, 5), (2, 1, 4, 5, 3), (2, 1, 5, 3, 4),
                                 (2, 1, 5, 4, 3), (2, 3, 1, 4, 5), (2, 3, 1, 5, 4), (2, 3, 4, 1, 5), (2, 3, 4, 5, 1),
                                 (2, 3, 5, 1, 4), (2, 3, 5, 4, 1), (2, 4, 1, 3, 5), (2, 4, 1, 5, 3), (2, 4, 3, 1, 5),
                                 (2, 4, 3, 5, 1), (2, 4, 5, 1, 3), (2, 4, 5, 3, 1), (2, 5, 1, 3, 4), (2, 5, 1, 4, 3),
                                 (2, 5, 3, 1, 4), (2, 5, 3, 4, 1), (2, 5, 4, 1, 3), (2, 5, 4, 3, 1)}
        structures_5_set = set(structures_5)
        self.assertEqual(structures_5_set, expected_structures_5)

        structures_3 = cvinecop._get_possible_cvine_structs(data_dim=3, pv_target_idx=1)
        expected_structures_3 = {(1, 2, 3), (1, 3, 2)}
        structures_3_set = set(structures_3)
        self.assertEqual(structures_3_set, expected_structures_3)

    def test_get_condi_probs_priv(self) -> None:
        """
        Test the _get_condi_probs function from vinecop_generate.
        """

        # Ignore possible integration warnings.
        warnings.filterwarnings(action='ignore', message='The integral is probably divergent')

        cvinecop = vg.CVineCop(self.pv_cvine_cop)
        # Various inputs in quantile
        u = np.array([[0, 0.1, 0.4, 0.6], [0.1, 0.1, 0.4, 0.6], [0.2, 0.1, 0.4, 0.6], [0.3, 0.1, 0.4, 0.6],
                      [0.9, 0.1, 0.4, 0.6], [1, 0.1, 0.4, 0.6], [0, 0.1, 0.4, 1], [0.1, 1, 0.4, 1], [0, 0, 1, 1],
                      [1, 0, 0, 0]])
        expected_results = np.array([0.0001, 0.07678050591604534, 0.17637106968139507, 0.283373927184108,
                                     0.9155297176502054, 0.9999, 0.0001, 0.03269281901703475, 0.0001, 0.9999])

        condi_probs = np.array([cvinecop._get_condi_prob(ui, pv_target_idx=1) for ui in u])
        np.testing.assert_array_almost_equal(condi_probs, expected_results, decimal=3)

    def test_get_condi_probs(self) -> None:
        """
        Test the get_condi_probs function from vinecop_generate.
        """

        # Ignore possible integration warnings.
        warnings.filterwarnings(action='ignore', message='The integral is probably divergent')

        cvinecop = vg.CVineCop(self.pv_cvine_cop)

        # Test 1D numpy array input
        u_array = np.array([0.2, 0.1, 0.4, 0.6])
        expected_condi_prob = 0.17637106968139507
        condi_prob_single = cvinecop.get_condi_probs(u=u_array)
        self.assertAlmostEqual(expected_condi_prob, condi_prob_single, places=4)

        # Test pd.DataFrame input
        u_df = pd.DataFrame([[0, 0.1, 0.4, 0.6], [0.1, 0.1, 0.4, 0.6], [0.2, 0.1, 0.4, 0.6], [0.3, 0.1, 0.4, 0.6],
                             [0.9, 0.1, 0.4, 0.6], [1, 0.1, 0.4, 0.6], [0, 0.1, 0.4, 1], [0.1, 1, 0.4, 1], [0, 0, 1, 1],
                             [1, 0, 0, 0]])
        expected_results = pd.Series([0.0001, 0.07678050591604534, 0.17637106968139507, 0.283373927184108,
                                      0.9155297176502054, 0.9999, 0.0001, 0.03269281901703475, 0.0001, 0.9999])
        condi_probs = cvinecop.get_condi_probs(u=u_df)
        pd.testing.assert_series_equal(expected_results, condi_probs, check_dtype=False, atol=4)

    def test_get_cop_densities(self) -> None:
        """
        Test the get_cop_densities function from vinecop_generate.
        """

        cvinecop = vg.CVineCop(self.pv_cvine_cop)

        # Test 1D numpy array input
        u_array = np.array([0.2, 0.1, 0.4, 0.6])
        expected_cop_density = 0.8593039707767826
        cop_density_single = cvinecop.get_cop_densities(u=u_array)
        self.assertAlmostEqual(expected_cop_density, cop_density_single, places=4)

        # Test pd.DataFrame input
        u_df = pd.DataFrame([[0, 0.1, 0.4, 0.6], [0.1, 0.1, 0.4, 0.6], [0.2, 0.1, 0.4, 0.6], [0.3, 0.1, 0.4, 0.6],
                             [0.9, 0.1, 0.4, 0.6], [1, 0.1, 0.4, 0.6], [0, 0.1, 0.4, 1], [0.1, 1, 0.4, 1], [0, 0, 1, 1],
                             [1, 0, 0, 0]])
        cop_densities = cvinecop.get_cop_densities(u_df)
        expected_cop_densities = pd.Series([3.272297e-03, 7.724320e-01, 8.593040e-01, 9.017561e-01, 7.418892e-01,
                                            4.308034e-01, 2.886153e-11, 6.266201e+07, 5.009017e-06, 1.229496e+15])
        pd.testing.assert_series_equal(expected_cop_densities, cop_densities, check_dtype=False, rtol=1e-3)

    def test_get_cop_evals(self) -> None:
        """
        Test the get_cop_evals function from vinecop_generate.
        """

        cvinecop = vg.CVineCop(self.pv_cvine_cop)

        # Test 1D numpy array input
        u_array = np.array([0.2, 0.1, 0.4, 0.6])
        expected_cop_eval = 0.037
        cop_eval_single = cvinecop.get_cop_evals(u=u_array)
        self.assertAlmostEqual(expected_cop_eval, cop_eval_single, places=2)

        # Test pd.DataFrame input
        u_df = pd.DataFrame([[0, 0.1, 0.4, 0.6], [0.1, 0.1, 0.4, 0.6], [0.2, 0.1, 0.4, 0.6], [0.3, 0.1, 0.4, 0.6],
                             [0.9, 0.1, 0.4, 0.6], [1, 0.1, 0.4, 0.6], [0, 0.1, 0.4, 1], [0.1, 1, 0.4, 1], [0, 0, 1, 1],
                             [1, 0, 0, 0]])
        cop_evals = cvinecop.get_cop_densities(u_df)
        expected_cop_evals = pd.Series([3.272297e-03, 7.724320e-01, 8.593040e-01, 9.017561e-01, 7.418892e-01,
                                        4.308034e-01, 2.886153e-11, 6.266201e+07, 5.009017e-06, 1.229496e+15])
        pd.testing.assert_series_equal(expected_cop_evals, cop_evals, check_dtype=False, rtol=1e-3)
