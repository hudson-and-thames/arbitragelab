"""
This module implements Copula-based Statistical Arbitrage tools.
"""

from arbitragelab.copula_approach.copula_calculation import (
    find_marginal_cdf, sic, aic, hqic, construct_ecdf_lin, scad_penalty,
    scad_derivative, adjust_weights, to_quantile, fit_copula_to_empirical_data)
from arbitragelab.copula_approach import archimedean
from arbitragelab.copula_approach import elliptical
from arbitragelab.copula_approach import mixed_copulas
from arbitragelab.copula_approach.copula_strategy_mpi import CopulaStrategyMPI
from arbitragelab.copula_approach.vine_copula_partner_selection import PartnerSelection
from arbitragelab.copula_approach.vinecop_generate import (RVineCop, CVineCop)
from arbitragelab.copula_approach.vinecop_strategy import CVineCopStrat
