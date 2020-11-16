"""
This module implements Copula-based Statistical Arbitrage strategies.
"""

from arbitragelab.copula_approach.copula_calculation import (find_marginal_cdf, ml_theta_hat, log_ml,
                                                             sic, aic, hqic)
from arbitragelab.copula_approach.copula_generate import (Copula, Gumbel, Frank, Clayton, Joe, N13,
                                                          N14, Gaussian, Student, Switcher)
from arbitragelab.copula_approach.copula_strategy import CopulaStrategy
