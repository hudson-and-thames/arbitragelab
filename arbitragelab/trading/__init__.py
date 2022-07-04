"""
Various trading rules implementation.
"""

from arbitragelab.trading.basic_copula import BasicCopulaTradingRule
from arbitragelab.trading.copula_strategy_mpi import MPICopulaTradingRule
from arbitragelab.trading.minimum_profit import MinimumProfitTradingRule
from arbitragelab.trading.multi_coint import MultivariateCointegrationTradingRule
from arbitragelab.trading.z_score import BollingerBandsTradingRule
