# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Unit tests for vinecop_generate, vinecop_strategy
"""

# pylint: disable = invalid-name, protected-access, consider-using-enumerate
import unittest
import os
import warnings
import pandas as pd
import numpy as np
import pyvinecopulib as pv
import arbitragelab.copula_approach.vinecop_generate as vg
import arbitragelab.copula_approach.vinecop_strategy as vs
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
        data_np, cdfs = ccalc.to_quantile(returns_train)
        pv_cvine_cop.select(data=data_np, controls=controls)

        self.pv_cvine_cop = pv_cvine_cop

        # Data stored as class attributes
        self.quantiles_data_train = data_np
        self.quantiles_data_test, _ = ccalc.to_quantile(returns_test)
        self.returns_train = returns_train
        self.returns_test = returns_test
        self.cdfs = cdfs
        self.target_stock_prices = sp500_prices['AAPL']
        self.index_prices = sp500_prices['SPY']

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

    def test_get_possible_cvine_structs_alt(self) -> None:
        """
        Test CVineCop._get_possible_cvine_structs_alt.
        """

        cvinecop = vg.CVineCop()
        structures_5 = cvinecop._get_possible_cvine_structs_alt(data_dim=5, pv_target_idx=2)
        self.assertTrue(isinstance(structures_5, list))
        expected_structures_5 = {(1, 3, 4, 5, 2), (1, 3, 5, 4, 2), (1, 4, 3, 5, 2), (1, 4, 5, 3, 2), (1, 5, 3, 4, 2),
                                 (1, 5, 4, 3, 2), (3, 1, 4, 5, 2), (3, 1, 5, 4, 2), (3, 4, 1, 5, 2), (3, 4, 5, 1, 2),
                                 (3, 5, 1, 4, 2), (3, 5, 4, 1, 2), (4, 1, 3, 5, 2), (4, 1, 5, 3, 2), (4, 3, 1, 5, 2),
                                 (4, 3, 5, 1, 2), (4, 5, 1, 3, 2), (4, 5, 3, 1, 2), (5, 1, 3, 4, 2), (5, 1, 4, 3, 2),
                                 (5, 3, 1, 4, 2), (5, 3, 4, 1, 2), (5, 4, 1, 3, 2), (5, 4, 3, 1, 2)}
        structures_5_set = set(structures_5)
        self.assertEqual(structures_5_set, expected_structures_5)

        structures_3 = cvinecop._get_possible_cvine_structs_alt(data_dim=3, pv_target_idx=1)
        expected_structures_3 = {(2, 3, 1), (3, 2, 1)}
        structures_3_set = set(structures_3)
        self.assertEqual(structures_3_set, expected_structures_3)

    def test_get_condi_probs_priv(self) -> None:
        """
        Test the _get_condi_probs function from CVineCop.
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
        Test the get_condi_probs function from CVineCop.
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
        Test the get_cop_densities function from CVineCop.
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
        Test the get_cop_evals function from CVineCop.
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
        cop_evals = cvinecop.get_cop_evals(u_df)
        expected_cop_evals = pd.Series([0., 0.0246, 0.0373, 0.0444, 0.0673, 0.069, 0., 0.0739, 0., 0.])
        pd.testing.assert_series_equal(expected_cop_evals, cop_evals, check_dtype=False, atol=1e-2)

    def test_simulate(self) -> None:
        """
        Test the simulate function from CVineCop.
        """

        cvinecop = vg.CVineCop(self.pv_cvine_cop)

        # Simulate with seeds, quasi random number
        samples_1 = cvinecop.simulate(n=5, qrn=True, seeds=[1], num_threads=2)
        expected_samples_1 = np.array([[0.81371918, 0.71286734, 0.04834333, 0.07197967],
                                       [0.30703092, 0.04620067, 0.23234878, 0.22256818],
                                       [0.60524091, 0.37953401, 0.22861292, 0.77214559],
                                       [0.48694196, 0.82397845, 0.88019649, 0.66608806],
                                       [0.73964397, 0.15731178, 0.25048662, 0.54705984]])
        np.testing.assert_array_almost_equal(samples_1, expected_samples_1, decimal=5)

        # Simulate with seeds, no quasi random number
        samples_2 = cvinecop.simulate(n=5, qrn=False, seeds=[1])
        expected_samples_2 = np.array([[0.28040356, 0.18609166, 0.18793118, 0.08634828],
                                       [0.42191178, 0.14888814, 0.10004586, 0.46644944],
                                       [0.41231885, 0.5561528, 0.98162344, 0.936266],
                                       [0.10936174, 0.51577398, 0.3811335, 0.69293515],
                                       [0.77400811, 0.5948096, 0.50113537, 0.31332545]])
        np.testing.assert_array_almost_equal(samples_2, expected_samples_2, decimal=5)

        # Simulate with no seeds, quasi random number
        samples_3 = cvinecop.simulate(n=5, qrn=True)
        self.assertIsInstance(samples_3, np.ndarray)
        self.assertEqual(samples_3.shape, (5, 4))

        # Simulate with no seeds, no quasi random number
        samples_4 = cvinecop.simulate(n=5, qrn=False)
        self.assertIsInstance(samples_4, np.ndarray)
        self.assertEqual(samples_4.shape, (5, 4))

    def test_aic_bic_loglik(self) -> None:
        """
        Test the aic, bic and loglik function in CVineCop class.
        """

        cvinecop = vg.CVineCop(self.pv_cvine_cop)

        # Calculate aic, bic, log-likelihood value on test data
        aic_value = cvinecop.aic(u=self.quantiles_data_train, num_threads=2)
        bic_value = cvinecop.bic(u=self.quantiles_data_train, num_threads=2)
        loglik_value = cvinecop.loglik(u=self.quantiles_data_train, num_threads=2)

        calculated = np.array([aic_value, bic_value, loglik_value])
        expected = np.array([-382.98496652926514, -355.04471469950926, 198.49248326463257])

        np.testing.assert_array_almost_equal(calculated, expected, decimal=3)

    def test_fit_auto(self) -> None:
        """
        Test the fit_auto function in CVinecop class.
        """

        cvinecop = vg.CVineCop()

        # 1. With renewal
        cvinecop.fit_auto(data=self.quantiles_data_train, pv_target_idx=1, if_renew=True)
        # Check its aic value on the training data
        aic_1 = cvinecop.aic(u=self.quantiles_data_train, num_threads=1)
        expected_aic_1 = -382.98496652926514
        self.assertAlmostEqual(aic_1, expected_aic_1, places=3)

        # 2. Without renewal, fit the cvine copula using the test set.
        cvinecop.fit_auto(data=self.quantiles_data_test, pv_target_idx=1, if_renew=False)
        # Check its aic value on the training data. Should still get the same result.
        aic_2 = cvinecop.aic(u=self.quantiles_data_train, num_threads=1)
        expected_aic_2 = -382.98496652926514
        self.assertAlmostEqual(aic_2, expected_aic_2, places=3)

    def test_fit_auto_alt(self) -> None:
        """
        Test the fit_auto function in CVinecop class using the alternative method.
        """

        cvinecop = vg.CVineCop()

        # 1. With renewal
        cvinecop.fit_auto(data=self.quantiles_data_train, pv_target_idx=1, if_renew=True, alt_cvine_structure=True)
        # Check its aic value on the training data
        aic_1 = cvinecop.aic(u=self.quantiles_data_train, num_threads=1)
        expected_aic_1 = -379.8244061045313
        self.assertAlmostEqual(aic_1, expected_aic_1, places=3)

        # 2. Without renewal, fit the cvine copula using the test set.
        cvinecop.fit_auto(data=self.quantiles_data_test, pv_target_idx=1, if_renew=False)
        # Check its aic value on the training data. Should still get the same result.
        aic_2 = cvinecop.aic(u=self.quantiles_data_train, num_threads=1)
        expected_aic_2 = -379.8244061045313
        self.assertAlmostEqual(aic_2, expected_aic_2, places=3)

    def test_strat_init(self) -> None:
        """
        Test the __init__ function from CVineCopStrat.
        """

        # 1. Instantiate with a given CVineCop, using default table
        cvinecop = vg.CVineCop(self.pv_cvine_cop)
        cvstrat = vs.CVineCopStrat(cvinecop)
        self.assertIsInstance(cvstrat.cvinecop, vg.CVineCop)
        default_signal_to_position_table = pd.DataFrame({1: {0: 1, 1: 1, -1: 0},
                                                         -1: {0: -1, 1: 0, -1: -1},
                                                         0: {0: 0, 1: 0, -1: 0},
                                                         2: {0: 0, 1: 1, -1: -1}})
        pd.testing.assert_frame_equal(default_signal_to_position_table, cvstrat.signal_to_position_table, atol=1e-4)

        # 2. Instantiate with no CVineCop, using a new table
        signal_to_position_table = pd.DataFrame({1: {0: 1, 1: 1, -1: 1},
                                                 -1: {0: -1, 1: 0, -1: -1},
                                                 0: {0: 0, 1: 0, -1: 0},
                                                 2: {0: 0, 1: 1, -1: -1}})
        cvinecop = vg.CVineCop()
        cvstrat = vs.CVineCopStrat(signal_to_position_table=signal_to_position_table)
        self.assertIsNone(cvstrat.cvinecop)
        pd.testing.assert_frame_equal(signal_to_position_table, cvstrat.signal_to_position_table, atol=1e-4)

    def test_calc_mpi(self) -> None:
        """
        Test the calc_mpi method for CVineCopStrat.
        """

        cvinecop = vg.CVineCop(self.pv_cvine_cop)
        cvstrat = vs.CVineCopStrat(cvinecop)
        dataset = self.returns_test.iloc[:10]  # Use only 10 data points to speed up calculation.

        # 1. MPIs with no mean subtraction
        mpis_1 = cvstrat.calc_mpi(returns=dataset, cdfs=self.cdfs, pv_target_idx=1, subtract_mean=False)
        expected_mpis_1 = np.array([0.319452, 0.510193, 0.498257, 0.864428, 0.169004, 0.669175, 0.479996, 0.216829,
                                    0.338173, 0.872803])
        np.testing.assert_array_almost_equal(mpis_1.to_numpy(), expected_mpis_1)

        # 2. MPIs with mean subtraction
        mpis_2 = cvstrat.calc_mpi(returns=dataset, cdfs=self.cdfs, pv_target_idx=1, subtract_mean=True)
        expected_mpis_2 = np.array([0.319452, 0.510193, 0.498257, 0.864428, 0.169004, 0.669175, 0.479996, 0.216829,
                                    0.338173, 0.872803]) - 0.5
        np.testing.assert_array_almost_equal(mpis_2.to_numpy(), expected_mpis_2)

        # 3. MPIs with target index being 2
        mpis_3 = cvstrat.calc_mpi(returns=dataset, cdfs=self.cdfs, pv_target_idx=2, subtract_mean=False)
        expected_mpis_3 = np.array([0.38365451, 0.64834723, 0.65025266, 0.89930077, 0.27221104, 0.79938626,
                                    0.25941534, 0.79929461, 0.65683873, 0.60624597])
        np.testing.assert_array_almost_equal(mpis_3.to_numpy(), expected_mpis_3)

    def test_signal_to_position_priv(self) -> None:
        """
        Test _signal_to_position method for CVineCopStrat.
        """

        past_positions = [0, 1, -1]
        signals = [0, 1, -1, 2]

        # 1. Default table.
        cvinecop = vg.CVineCop(self.pv_cvine_cop)
        cvstrat = vs.CVineCopStrat(cvinecop)
        new_positions = []
        for past_pos in past_positions:
            for signal in signals:
                new_positions.append(cvstrat._signal_to_position(past_pos, signal))

        new_positions = np.array(new_positions, dtype=int)
        expected_new_positions = np.array([0, 1, -1, 0, 0, 1, 0, 1, 0, 0, -1, -1], dtype=int)

        np.testing.assert_array_equal(new_positions, expected_new_positions)

        # 2. Custom table.
        signal_to_position_table = pd.DataFrame({1: {0: 1, 1: 1, -1: 1},
                                                 -1: {0: -1, 1: 0, -1: -1},
                                                 0: {0: 0, 1: 0, -1: 0},
                                                 2: {0: 0, 1: 1, -1: -1}})
        cvinecop = vg.CVineCop(self.pv_cvine_cop)
        cvstrat = vs.CVineCopStrat(cvinecop, signal_to_position_table)
        new_positions_2 = []
        for past_pos in past_positions:
            for signal in signals:
                new_positions_2.append(cvstrat._signal_to_position(past_pos, signal))

        new_positions_2 = np.array(new_positions_2, dtype=int)
        expected_new_positions_2 = np.array([0, 1, -1, 0, 0, 1, 0, 1, 0, 1, -1, -1], dtype=int)

        np.testing.assert_array_equal(new_positions_2, expected_new_positions_2)

    def test_get_cur_signal_bollinger(self) -> None:
        """
        Test the get_cur_signal_bollinger method for CVineCopStrat.
        """

        cvinecop = vg.CVineCop(self.pv_cvine_cop)
        cvstrat = vs.CVineCopStrat(cvinecop)

        # All possible past CMPI and current CMPI value.
        past_cmpis = [1.2, 1.2, 1.2, 1.2, 0.6, 0.6, 0.6, 0.6, -1.2, -1.2, -1.2, -1.2, -0.6, -0.6, -0.6, -0.6]
        cur_cmpis = [1.3, 0.8, -0.8, -1.2, 0.8, 1.2, -0.6, -1.2, -1.3, -0.8, 0.8, 1.2, -0.8, -1.2, 0.6, 1.2]

        # Set the Bollinger band be: lower bound=-1, running mean=0, upper_bound=1, for eaiser comparison.
        signals = []
        for p in range(len(past_cmpis)):
            signals.append(cvstrat.get_cur_signal_bollinger(past_cmpis[p], cur_cmpis[p], running_mean=0,
                                                            upper_threshold=1, lower_threshold=-1))

        signals = np.array(signals, dtype=int)
        expected_signals = np.array([-1, 2, 0, 0, 2, -1, 0, 0, 1, 2, 0, 0, 2, 1, 0, 0], dtype=int)

        np.testing.assert_array_equal(signals, expected_signals)

    def test_get_positions_bollinger(self) -> None:
        """
        Test the get_positions_bollonger method for CVineCopStrat.
        """

        cvinecop = vg.CVineCop(self.pv_cvine_cop)
        cvstrat = vs.CVineCopStrat(cvinecop)

        returns = self.returns_test.iloc[:50]
        cdfs = self.cdfs

        # 1. No side-loaded MPIs, No returning Bollinger band data
        positions_1 = cvstrat.get_positions_bollinger(returns=returns, cdfs=cdfs, init_pos=0, past_obs=20,
                                                      threshold_std=1.0, mpis=None, if_return_bollinger_band=False)
        positions_1 = np.array(positions_1)
        expected_positions_1 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0,
                                         0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                                         0, 0, -1, -1, -1, -1])
        np.testing.assert_array_almost_equal(positions_1, expected_positions_1)

        # 2. No side-loaded MPIs, returning Bollinger band data
        positions_2, bband_2 = cvstrat.get_positions_bollinger(returns=returns, cdfs=cdfs, init_pos=0, past_obs=20,
                                                               threshold_std=1.0, mpis=None,
                                                               if_return_bollinger_band=True)
        positions_2 = np.array(positions_2)
        expected_bband_2 = np.array(
            [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan], [-0.99804174, -0.51242996, -0.02681818],
             [-1.01389355, -0.53421688, -0.0545402], [-1.01410921, -0.53519172, -0.05627423],
             [-1.01585547, -0.54073564, -0.06561581], [-1.01443563, -0.5431882, -0.07194077],
             [-1.01440765, -0.53871901, -0.06303036], [-1.01355233, -0.54504716, -0.07654198],
             [-1.02009377, -0.56974981, -0.11940585], [-1.02381349, -0.57755538, -0.13129726],
             [-1.03164991, -0.58661169, -0.14157347], [-1.05140426, -0.62233172, -0.19325919],
             [-1.07200258, -0.64781845, -0.22363432], [-1.06415267, -0.63549678, -0.20684088],
             [-1.07106441, -0.64187137, -0.21267833], [-1.06701021, -0.63792586, -0.20884152],
             [-1.03268445, -0.61200584, -0.19132723], [-1.00132712, -0.59859125, -0.19585537],
             [-1.02406425, -0.60640481, -0.18874537], [-0.99048707, -0.59439837, -0.19830968],
             [-0.98614651, -0.59228322, -0.19841993], [-0.96288318, -0.57358595, -0.18428872],
             [-0.96237537, -0.57312583, -0.18387629], [-0.97926186, -0.59957237, -0.21988287],
             [-0.98888191, -0.61657377, -0.24426563], [-0.97483941, -0.64025171, -0.30566401],
             [-0.96810442, -0.65350625, -0.33890809], [-0.96208956, -0.67228683, -0.38248409],
             [-0.96345298, -0.65933248, -0.35521197], [-0.96573728, -0.64677872, -0.32782015],
             [-0.98176343, -0.60250981, -0.22325619], [-0.97292103, -0.55106804, -0.12921505]])

        np.testing.assert_array_almost_equal(positions_2, expected_positions_1)
        np.testing.assert_array_almost_equal(bband_2.to_numpy(), expected_bband_2)

        # 3. Side loading CMPIs, output Bollinger Band
        mpis = cvstrat.calc_mpi(returns=returns, cdfs=self.cdfs, pv_target_idx=1, subtract_mean=False)
        positions_3, bband_3 = cvstrat.get_positions_bollinger(
            returns=returns, cdfs=cdfs, init_pos=0, past_obs=20, threshold_std=1.0, mpis=mpis,
            if_return_bollinger_band=True)

        np.testing.assert_array_almost_equal(positions_3, expected_positions_1)
        np.testing.assert_array_almost_equal(bband_3.to_numpy(), expected_bband_2)

    def test_get_cur_pos_bollinger(self) -> None:
        """
        Test the get_cur_position_bollinger method in CVineCopStrat.
        """

        cvinecop = vg.CVineCop(self.pv_cvine_cop)
        cvstrat = vs.CVineCopStrat(cvinecop)

        new_position, cur_cmpi = cvstrat.get_cur_pos_bollinger(
            returns_slice=self.returns_test.iloc[:21], cdfs=self.cdfs, past_pos=0, pv_target_idx=1, past_cmpi=0,
            threshold_std=1.0)

        past_positions = [0, -1, 1]
        past_cmpis = [-1.6, -0.6, 0.4]
        new_positions = []
        cur_cmpis = []
        for past_pos in past_positions:
            for past_cmpi in past_cmpis:
                new_position, cur_cmpi = cvstrat.get_cur_pos_bollinger(
                    returns_slice=self.returns_test.iloc[:21], cdfs=self.cdfs, past_pos=past_pos, pv_target_idx=1,
                    past_cmpi=past_cmpi, threshold_std=1.0)
                new_positions.append(new_position)
                cur_cmpis.append(cur_cmpi)

        expected_new_positions = [0, 0, 0, -1, -1, -1, 1, 1, 1]
        expected_cur_cmpis = [-2.2162865028946586, -1.2162865028946583, -0.21628650289465834, -2.2162865028946586,
                              -1.2162865028946583, -0.21628650289465834, -2.2162865028946586, -1.2162865028946583,
                              -0.21628650289465834]

        np.testing.assert_array_almost_equal(new_positions, expected_new_positions)
        np.testing.assert_array_almost_equal(cur_cmpis, expected_cur_cmpis)

    def test_positions_to_units_against_index(self) -> None:
        """
        Test the positions_to_units_against_index method in CVineCopStrat.
        """

        cvinecop = vg.CVineCop(self.pv_cvine_cop)
        cvstrat = vs.CVineCopStrat(cvinecop)
        target_stock_prices = self.target_stock_prices.iloc[:50]
        index_prices = self.index_prices[:50]

        positions = cvstrat.get_positions_bollinger(returns=self.returns_test[:50], cdfs=self.cdfs, init_pos=0,
                                                    past_obs=10, threshold_std=1.0, mpis=None,
                                                    if_return_bollinger_band=False)

        units_df = cvstrat.positions_to_units_against_index(target_stock_prices, index_prices, positions,
                                                            multiplier=10000)
        expected_units = np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
                                   [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
                                   [np.nan, np.nan], [0., 0.], [0., 0.], [465.89015937, -42.29402765],
                                   [465.89015937, - 42.29402765], [465.89015937, - 42.29402765],
                                   [465.89015937, - 42.29402765], [465.89015937, - 42.29402765],
                                   [465.89015937, - 42.29402765], [465.89015937, - 42.29402765],
                                   [465.89015937, - 42.29402765], [465.89015937, - 42.29402765],
                                   [465.89015937, - 42.29402765], [0.,  0.], [-441.02822241, 40.68679327],
                                   [-441.02822241, 40.68679327], [-441.02822241, 40.68679327],
                                   [-441.02822241, 40.68679327], [-441.02822241, 40.68679327],
                                   [-441.02822241, 40.68679327], [-441.02822241, 40.68679327],
                                   [0., 0.], [437.00834394, - 40.29008913], [437.00834394, - 40.29008913],
                                   [0., 0.], [0., 0.], [0., 0.], [0., 0.], [432.63286684, - 39.80891768],
                                   [432.63286684, - 39.80891768], [432.63286684, - 39.80891768],
                                   [0., 0.], [0., 0.], [0., 0.], [0., 0.], [-422.59046387, 39.37627869],
                                   [-422.59046387, 39.37627869], [-422.59046387, 39.37627869],
                                   [-422.59046387, 39.37627869], [-422.59046387, 39.37627869],
                                   [-422.59046387, 39.37627869], [-422.59046387, 39.37627869]])

        np.testing.assert_array_almost_equal(units_df.to_numpy(), expected_units)
