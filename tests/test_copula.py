# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""Unit test for copula approach"""
from arbitragelab.copula_approach import copula_generate, copula_strategy, copula_calculation
import unittest
import numpy as np
import pandas as pd
import os

class TestCopulas(unittest.TestCase):
    """Test each copula class, calculations and strategy."""
    def setUp(self):
        """
        Get the correct directory.
        """
        project_path = os.path.dirname(__file__)
        self.data_path = project_path + r'/test_data'

    def test_gumbel(self):
        """Test gumbel copula class."""
        cop = copula_generate.Gumbel(theta=2)
        
        # Check copula joint cumulative density C(U=u,V=v)
        self.assertEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7))
        self.assertEqual(cop.C(0.7, 1), cop.C(1, 0.7))
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertEqual(cop.C(0.7, 1), 0.7)
        self.assertAlmostEqual(cop.C(0.7, 0.5), 0.458621, delta=1e-4)
        
        # Check copula joint probability density c(U=u,V=v)
        self.assertEqual(cop.c(0.5, 0.7), cop.c(0.7,0.5))
        self.assertAlmostEqual(cop.c(0.5, 0.7), 1.21699, delta=1e-4)
        
        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.299774, delta=1e-4)
        
        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(0.5), 2, delta=1e-4)
        
    def test_frank(self):
        """Test Frank copula class."""
        cop = copula_generate.Frank(theta=10)
        
        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), 0.7, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.5, 0.7), 0.487979, delta=1e-4)
        
        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7,0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.5, 0.7), 1.06418, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.119203, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(0.21389456921960), 2, delta=1e-4)

    def test_clayton(self):
        """Test Clayton copula class."""
        cop = copula_generate.Clayton(theta=2)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), 0.7, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 0.5), 0.445399, delta=1e-4)

        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7,0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.5, 0.7), 1.22649, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.257605, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(0.5), 2, delta=1e-4)
    
    def test_joe(self):
        """Test Joe copula class."""
        cop = copula_generate.Joe(theta=6)
        
        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), 0.7, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.5, 0.7), 0.496244, delta=1e-4)
        
        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7,0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.5, 0.7), 0.71849, delta=1e-4)
        
        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.0737336, delta=1e-4)
        
        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(0.35506593315175), 2, delta=1e-4)

    def test_n13(self):
        """Test N14 Copula class."""
        cop = copula_generate.N13(theta=3)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.3, 0.7), 0.271918, delta=1e-4)

        # # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7,0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.3, 0.7), 0.770034, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.3, 0.7), 0.134891, delta=1e-4)
        
        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(0.222657233776425), 2, delta=1e-4)

    def test_n14(self):
        """Test N14 Copula class."""
        cop = copula_generate.N14(theta=3)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.3, 0.7), 0.298358, delta=1e-4)

        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7,0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.3, 0.7), 0.228089, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.3, 0.7), 0.0207363, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(3/5), 2, delta=1e-4)

    def test_gaussian(self):
        """Test Gaussian copula class."""
        cov = [[2, 0.5], [0.5, 2]]
        cop = copula_generate.Gaussian(cov=cov)

        # Check copula joint cumulative density C(U=u,V=v)
        self.assertAlmostEqual(cop.C(0.7, 1e-4), cop.C(1e-4, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), cop.C(1, 0.7), delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1e-8), 0, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.7, 1), 0.7, delta=1e-4)
        self.assertAlmostEqual(cop.C(0.5, 0.7), 0.384944, delta=1e-4)

        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7,0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.5, 0.7), 1.0327955, delta=1e-4)
        self.assertAlmostEqual(cop.c(0.6, 0.7), 0.999055, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.446148, delta=1e-4)

        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(2*np.arcsin(0.2)/np.pi), 0.2, delta=1e-4)


    def test_student(self):
        """Test Student copula class (Student-t)."""
        cov = [[2, 0.5], [0.5, 2]]
        nu = 5
        cop = copula_generate.Student(cov=cov, nu=nu)
        
        # More to be added here for test on C(U<=u, V<=v)

        # Check copula joint probability density c(U=u,V=v)
        self.assertAlmostEqual(cop.c(0.5, 0.7), cop.c(0.7,0.5), delta=1e-8)
        self.assertAlmostEqual(cop.c(0.5, 0.7), 1.09150554, delta=1e-4)
        self.assertAlmostEqual(cop.c(0.6, 0.7), 1.1416005, delta=1e-4)

        # Check copula conditional cdf Prob(U<=u|V=v)
        self.assertAlmostEqual(cop.condi_cdf(0.5, 0.7), 0.4415184293094455, delta=1e-4)
        
        # Check theta(tau)
        self.assertAlmostEqual(cop.theta_hat(2*np.arcsin(0.2)/np.pi), 0.2, delta=1e-4)

    def test_signal(self):
        """Test trading signal generation."""
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
        """Test calculation of cumulative log return."""
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
        
    def test_series_condi_prob(self):
        """Test calculating the conditional probabilities of a seires."""
        # Expected value.
        expected_probs = np.array([[0.00198373, 0.00198373],
                                   [0.5, 0.5],
                                   [0.998016, 0.998016]])
        # Initiate a Gaussian copula to test.
        cov = [[2, 0.5], [0.5, 2]]
        GaussianC = copula_generate.Gaussian(cov=cov)
        CS = copula_strategy.CopulaStrategy(GaussianC)
        s1 = np.linspace(0.0001, 1-0.0001, 3)  # Assume those are marginal cumulative densities already.
        s2 = np.linspace(0.0001, 1-0.0001, 3)
        cdf1 = lambda x: x  # Use identity mapping for cumulative density.
        cdf2 = lambda x: x
        prob_series = CS.series_condi_prob(s1_series=s1, s2_series=s2, cdf1=cdf1, cdf2=cdf2)
        np.testing.assert_array_almost_equal(prob_series, expected_probs, decimal=6)
        
    def test_ICs(self):
        """Test three information criterions."""
        aic = copula_calculation.aic
        sic = copula_calculation.sic
        hqic = copula_calculation.hqic
        log_likelihood = 1000
        n = 200
        k = 2
        self.assertAlmostEqual(aic(log_likelihood, n, k), -1995.9390862944163, delta=1e-5)
        self.assertAlmostEqual(sic(log_likelihood, n, k), -1989.4033652669038, delta=1e-5)
        self.assertAlmostEqual(hqic(log_likelihood, n, k), -1993.330442831434, delta=1e-5)

    def test_ml_theta_hat(self):
        """Test max likelihood fit of theta hat for each copula."""
        # Import data.
        pair_prices = pd.read_csv(self.data_path + r'/BKD_ESC_2009_2011.csv')
        BKD_series = pair_prices['BKD'].to_numpy()
        ESC_series = pair_prices['ESC'].to_numpy()
        # Change price to cumulative log return. Here we fit the whole set.
        ml_theta_hat = copula_calculation.ml_theta_hat
        CS = copula_strategy.CopulaStrategy()
        BKD_clr = CS.cum_log_return(BKD_series)
        ESC_clr = CS.cum_log_return(ESC_series)
        # Fit through the copulas using theta_hat as its parameter
        copulas = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14', 'Gaussian', 'Student']
        theta_hats = np.array(
            [ml_theta_hat(x=BKD_clr, y=ESC_clr, copula_name=name) for name in copulas])
        # Expected values.
        expected_theta = np.array([4.823917032678924, 7.6478340653578485, 17.479858671919537, 8.416268109560686,
                                   13.006445455285089, 4.323917032678924, 0.9474504200741508, 0.9474504200741508])

        np.testing.assert_array_almost_equal(theta_hats, expected_theta, decimal=6)
        
    def test_fit_copula(self):
        """Test fit_copula in CopulaStrategy for each copula."""
        # Import data.
        pair_prices = pd.read_csv(self.data_path + r'/BKD_ESC_2009_2011.csv')
        BKD_series = pair_prices['BKD'].to_numpy()
        ESC_series = pair_prices['ESC'].to_numpy()
        # Change price to cumulative log return. Here we fit the whole set.
        CS = copula_strategy.CopulaStrategy()
        BKD_clr = CS.cum_log_return(BKD_series)
        ESC_clr = CS.cum_log_return(ESC_series)
        # Fit through the copulas and watch out for Student-t
        copulas = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14', 'Gaussian', 'Student']
        aics = dict()
        for name in copulas:
            if name != 'Student':
                result_dict,_,_,_ = CS.fit_copula(s1_series=BKD_clr, s2_series=ESC_clr, copula_name=name)
                aics[name] = result_dict['AIC']
            else:  # For Student copula
                result_dict,_,_,_ = CS.fit_copula(s1_series=BKD_clr, s2_series=ESC_clr, copula_name=name, nu=3)
                aics[name] = result_dict['AIC']
        
        expeced_aics = {'Gumbel': -1996.8584204971112, 'Clayton': -1982.1106036413414,
                        'Frank': -2023.0991514138464, 'Joe': -1139.896265173598,
                        'N13': -2211.6295423299603, 'N14': -2111.9831835080827,
                        'Gaussian': -413.9148808046805, 'Student': -2204.0928279630475}

        for key in aics:
            self.assertAlmostEqual(aics[key], expeced_aics[key], delta = 1e-5)
            
    def test_analyze_time_series(self):
        """Test analyze_time_series in CopulaStrategy for each copula."""
        pair_prices = pd.read_csv(self.data_path + r'/BKD_ESC_2009_2011.csv')
        BKD_series = pair_prices['BKD'].to_numpy()
        ESC_series = pair_prices['ESC'].to_numpy()

        CS = copula_strategy.CopulaStrategy()
        BKD_clr = CS.cum_log_return(BKD_series)
        ESC_clr = CS.cum_log_return(ESC_series)

        # Training testing split
        training_length = 670

        BKD_train = BKD_clr[: training_length]
        ESC_train = ESC_clr[: training_length]
        BKD_test = BKD_clr[training_length :]
        ESC_test = ESC_clr[training_length :]

        # Compare their AIC values
        copulas = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14', 'Gaussian', 'Student']

        # For each copula type, fit, then analyze
        positions_data = {}
        for name in copulas:
            if name != 'Student':
                _, _, cdf1, cdf2 = CS.fit_copula(s1_series=BKD_train, s2_series=ESC_train, copula_name=name)
                positions = CS.analyze_time_series(s1_series=BKD_test, s2_series=ESC_test,
                                                    cdf1=cdf1, cdf2=cdf2,
                                                    lower_threshold=0.75, upper_threshold=0.25)
                positions_data[name] = positions
            else:
                _, _, cdf1, cdf2 = CS.fit_copula(s1_series=BKD_train, s2_series=ESC_train, copula_name=name, nu=3)
                positions = CS.analyze_time_series(s1_series=BKD_test, s2_series=ESC_test,
                                                    cdf1=cdf1, cdf2=cdf2,
                                                    lower_threshold=0.75, upper_threshold=0.25)
                positions_data[name] = positions

        # Load and compare with theoretical data
        expected_positions_df = pd.read_csv(self.data_path + r'/BKD_ESC_unittest_positions.csv')
        for name in copulas:
            np.testing.assert_array_almost_equal(positions_data[name],
                                                  expected_positions_df[name].to_numpy(),
                                                  decimal=3)

    def test_ic_test(self):
        """Test ic_test from CopulaStrategy for each copula."""
        # 1. Get and process price pairs data
        pair_prices = pd.read_csv(self.data_path + r'/BKD_ESC_2009_2011.csv')
        BKD_series = pair_prices['BKD'].to_numpy()
        ESC_series = pair_prices['ESC'].to_numpy()
        # Change price to cumulative log return. Here we fit the whole set.
        CS = copula_strategy.CopulaStrategy()
        BKD_clr = CS.cum_log_return(BKD_series)
        ESC_clr = CS.cum_log_return(ESC_series)

        # 2. Fit to every copula, and get the SIC, AIC, HQIC data from ic_test
        copulas = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14', 'Gaussian', 'Student']
        ic_type = ['SIC', 'AIC', 'HQIC']
        ic_dict = {copula: {ic: None for ic in ic_type} for copula in copulas}
        for name in copulas:
            if name != 'Student':
                _,_,cdf1,cdf2 = CS.fit_copula(s1_series=BKD_clr, s2_series=ESC_clr, copula_name=name)
                result_dict = CS.ic_test(s1_test=BKD_clr, s2_test=ESC_clr, cdf1=cdf1, cdf2=cdf2)
                for ic in ic_type:
                    ic_dict[name][ic] = result_dict[ic]
            else:  # For Student copula, use nu=3
                _,_,cdf1,cdf2 = CS.fit_copula(s1_series=BKD_clr, s2_series=ESC_clr, copula_name=name, nu=3)
                result_dict = CS.ic_test(s1_test=BKD_clr, s2_test=ESC_clr, cdf1=cdf1, cdf2=cdf2)
                for ic in ic_type:
                    ic_dict[name][ic] = result_dict[ic]

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
                          {'SIC': -409.00612602007965, 'AIC': -413.9148808046805, 'HQIC': -412.0521358526326},
                          'Student':
                          {'SIC': -2199.1840731784464, 'AIC': -2204.0928279630475, 'HQIC': -2202.2300830109994}}

        # 4. Check with ic_test value.
        for name in copulas:
            for ic in ic_type:
                self.assertAlmostEqual(ic_dict[name][ic], ic_dict_expect[name][ic], delta=1e-5)

# if __name__ == "__main__":
#     unittest.main()