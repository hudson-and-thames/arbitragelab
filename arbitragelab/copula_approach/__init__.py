"""
This module implements Copula-based Statistical Arbitrage strategies.
"""

from arbitragelab.copula_approach.copula_calculation import (
    find_marginal_cdf, sic, aic, hqic, construct_ecdf_lin,scad_penalty,
    scad_derivative, adjust_weights, to_quantile)
from arbitragelab.copula_approach.copula_generate import (
    Copula, Gumbel, Frank, Clayton, Joe, N13, N14, Gaussian)
from arbitragelab.copula_approach.elliptical.student import StudentCopula
from arbitragelab.copula_approach.copula_strategy import CopulaStrategy
from arbitragelab.copula_approach.copula_strategy_basic import BasicCopulaStrategy
from arbitragelab.copula_approach.copula_strategy_mpi import CopulaStrategyMPI
from arbitragelab.copula_approach.copula_generate_mixedcopula import (MixedCopula, CFGMixCop, CTGMixCop)
from arbitragelab.copula_approach.vine_copula_partner_selection import PartnerSelection
from arbitragelab.copula_approach.vinecop_generate import (RVineCop, CVineCop)
from arbitragelab.copula_approach.vinecop_strategy import CVineCopStrat
