"""
This module implements Cointegration-based Statistical Arbitrage strategies
"""
from arbitragelab.cointegration_approach.johansen import JohansenPortfolio
from arbitragelab.cointegration_approach.engle_granger import EngleGrangerPortfolio
from arbitragelab.cointegration_approach.signals import (get_half_life_of_mean_reversion,
                                                         bollinger_bands_trading_strategy, linear_trading_strategy)
