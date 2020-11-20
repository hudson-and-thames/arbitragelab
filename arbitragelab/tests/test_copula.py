# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""Unit test for copula approach"""
import unittest
from arbitragelab.copula_approach import copula_generate, copula_strategy, copula_calculation

class TestCopulas(unittest.TestCase):
    """Test each copula class."""
    
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
        
        # Testing exiting
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

if __name__ == "__main__":
    unittest.main()