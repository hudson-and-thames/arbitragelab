"""
Module which implements various hedge ratios calculations.
"""

from arbitragelab.hedge_ratios.linear import get_ols_hedge_ratio, get_tls_hedge_ratio
from arbitragelab.hedge_ratios.half_life import get_minimum_hl_hedge_ratio
from arbitragelab.hedge_ratios.adf_optimal import get_adf_optimal_hedge_ratio
from arbitragelab.hedge_ratios.johansen import get_johansen_hedge_ratio
from arbitragelab.hedge_ratios.spread_construction import construct_spread
