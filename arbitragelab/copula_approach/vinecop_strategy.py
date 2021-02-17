# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module that generates vine copulas.

Built on top of the :code:`pyvinecoplib` package. See https://github.com/vinecopulib/pyvinecopulib for more details.
"""

from typing import List, Union, Callable, Tuple
from itertools import permutations
from arbitragelab.copula_approach.vinecop_generate import CVineCop
import numpy as np
import pandas as pd
import pyvinecopulib as pv

class CVineCopStrat():
    """
    Trading strategy class using vine copulas.
    """
    def __init__(self, cvinecop: CVineCop = None):
        """
        Create the trading strategy class.
        """
        self.cvine_cop = cvinecop
        self.past_obs = None
        self.threshold_std = None
        # Signal to position table: Row indexed by previous position, column indexed by signal
        # Refer to doc string of self._signal_to_position() for more detail
        self.signal_to_position_table = pd.DataFrame({1: {0: 1, 1: 1, -1: 0},
                                                      -1: {0: -1, 1: 0, -1: -1},
                                                      0: {0: 0, 1: 0, -1: 0},
                                                      2: {0: 0, 1: 1, -1: -1}})
    
    def calc_mpi(self, returns: pd.DataFrame, cdfs: List[Callable], pv_target_idx: int = 1,
                 subtract_mean: bool = False) -> pd.Series:
        """
        Calculate mispricing indices from returns for the target stock.

        Mispricing indices are technically cumulative conditional probabilities calculated from a copula based on
        returns data.
        """
        
        # 1. Map all the returns to quantiles
        quantiles = pd.DataFrame(data=0, index=returns.index, columns=returns.columns)
        for i, column_name in enumerate(returns):
            quantiles[column_name] = returns[column_name].apply(cdfs[i])
        
        # 2. Calculate the conditional probabilities
        mpis = self.cvine_cop.get_condi_probs(quantiles, pv_target_idx)
        
        if subtract_mean:
            mpis = mpis.subtract(0.5)

        return mpis
        
    def select_candidates(self):
        pass
    
    def get_positions_bollinger(self, returns: pd.DataFrame, cdfs: List[Callable], pv_target_idx: int = 1,
                                init_pos: int = 0, past_obs: int = 20, threshold_std: float = 1.0,
                                mpis: pd.Series = None, if_return_bollinger_band = False) -> pd.Series:
        """
        Get positions based on the strategy suggested by [Bollinger, 1992].
        
        Calculate the running mean and standard deviation of Cumulative mispricing index (CMPI) for the past 20
        observations (one month if it is daily data) and create the running Bollinger band as 
        (mean - k*std, mean + k*std), k is the threshold_std variable. Trading logic is as follows:
            
            - Long if the CMPI crosses its lower Bollinger band (undervalued)
            - Short if the CMPI crosses its upper Bollinger band (overvalued)
            - Exit if the CMPI crosses the running mean of the Bollinger band (reversion to equilibrium)
            
        This is used for backtesting when one has access to all testing data at once. The first 20 observations
        ([0, 19]) will be used for calculating the Bollinger band and the trading signal generation starts from
        observation 20. For live data feed please use self.get_cur_pos_bollinger().
        """
        
        self.past_obs = past_obs
        self.threshold_std = threshold_std
        
        # 1. Get the initial mean and std of CMPI
        if mpis is None:
            cmpis = self.calc_mpi(returns, cdfs, pv_target_idx, subtract_mean=True).cumsum()
        else:
            cmpis = mpis.cumsum()

        init_mean = cmpis.iloc[:past_obs].mean()
        init_std = cmpis.iloc[:past_obs].std()
        
        # 2. Get positions
        # Initiating
        trading_cmpis = cmpis.iloc[past_obs-1:]
        bollinger_band = pd.DataFrame(data=0, index=trading_cmpis.index, columns=['LowerBound', 'Mean', 'UpperBound'])
        bollinger_band.iloc[0] = [init_mean - threshold_std*init_std, init_mean, init_mean + threshold_std*init_std]
        positions = pd.Series(data=0, index=trading_cmpis.index)
        positions.iloc[0] = init_pos
        # Loop through trading_cmpis to calculate positions
        for idx in range(1, len(trading_cmpis)):
            # Get signal
            signal = self.get_cur_signal_bollinger(
                past_pos=positions.iloc[idx-1], past_cmpi=trading_cmpis[idx-1], cur_cmpi=trading_cmpis[idx],
                running_mean=bollinger_band['Mean'].iloc[idx-1],
                upper_threshold=bollinger_band['UpperBound'].iloc[idx-1],
                lower_threshold=bollinger_band['LowerBound'].iloc[idx-1])
            # Translate signal to current position
            positions.iloc[idx] = self._signal_to_position(past_pos=positions.iloc[idx-1], signal=signal)

            # Update the bollinger band
            cur_returns_slice = cmpis.iloc[idx: idx+past_obs]
            cur_mean = cur_returns_slice.mean()
            cur_std = cur_returns_slice.std()
            # Format: LowerBound, Mean, UpperBound
            bollinger_band.iloc[idx] = [cur_mean - threshold_std*cur_std, cur_mean, cur_mean + threshold_std*cur_std]
        
        # 3. Return positions (and the bollinger band)
        if if_return_bollinger_band:
            return positions, bollinger_band

        return positions
            
    @staticmethod
    def get_cur_signal_bollinger(past_pos: int, past_cmpi: float, cur_cmpi: float, running_mean: float,
                                 upper_threshold: float, lower_threshold: float) -> int:
        """
        Get the currrent trading signal based.
        
        Signal types {1: long, -1: short, 0: exit, 2: do nothing}
        """
        
        signal = 2  # By default do nothing
        
        # Open signal
        if_long = (cur_cmpi < lower_threshold)
        if_short = (cur_cmpi > upper_threshold)
        
        # Exit signal: when the cmpi crosses with the running mean
        if_exit = ((past_cmpi - running_mean) * (cur_cmpi - running_mean) < 0)
        
        # Assemble the signal together
        if_any_signal = any([if_long, if_short, if_exit])
        if if_any_signal:
            signal = (int(if_long) - int(if_short)) * int(not if_exit)
            # Return the updated signal
            return signal
        # Return the default signal, which is do nothing
        return signal

    def _signal_to_position(self, past_pos: int, signal: int, signal_to_position_table: pd.DataFrame = None) -> int:
        """
        Map the signal to position, given the past position.
        
        The default map is:
            signal [long, short, exit, do-nothing], acting on previous positions being [0, 1, -1] respectively
            
            * long signal: {0: 1, 1: 1, -1: 0}
            * short signal: {0: -1, 1: 0, -1: 0}
            * exit signal: {0: 0, 1: 0, -1: 0}
            * do-nothing signal: {0: 0, 1: 1, -1: -1}
        """
        
        if signal_to_position_table is not None:
            self.signal_to_position_table = signal_to_position_table
        
        table = self.signal_to_position_table
        
        new_position = table.loc[past_pos, signal]
        
        return new_position
        
    def get_cur_pos_bollinger(self, returns_slice: pd.DataFrame, cdfs: List[Callable], past_pos: int,
                              pv_target_idx: int = 1, past_cmpi: float = 0, threshold_std: float = 1.0) -> Tuple:
        """
        Get the suggested position and updated bollinger band from cur_returns pandas DataFrame.
        
        If the dataframe has 21 rows, then the first 20 rows will be used to calculate the Bollinger band, then the
        position will be generated based on the last row. Then we calculate the updated Bollinger band. This is used
        for live data feed.
        """
        
        # 1. Calculate the Bollinger band: mean and std
        cmpis = past_cmpi + self.calc_mpi(returns_slice, cdfs, subtract_mean=True).cumsum()
        init_mean = cmpis.iloc[: -1].mean()
        init_std = cmpis.iloc[: -1].std()

        # 2. Get positions from the last row of the dataframe
        past_cmpi = cmpis.iloc[-2]
        cur_cmpi = cmpis.iloc[-1]
        signal = self.get_cur_signal_bollinger(past_pos, past_cmpi, cur_cmpi, running_mean=init_mean,
                                               upper_threshold=init_mean + threshold_std*init_std,
                                               lower_threshold=init_mean - threshold_std*init_std)
        new_position = self._signal_to_position(past_pos, signal)
        
        return new_position, cur_cmpi
    
    @staticmethod
    def positions_to_units_cohort(prices_df: pd.DataFrame, positions: pd.Series, pv_target_idx: int = 1,
                                  multiplier: float = 1) -> pd.DataFrame:
        """
        Change the positions series into units held for each security for a dollar neutral strategy.
        
        For example., for 1 dollar investment of a 4-stock-cohort, long the target stock by 1/2 dollar and short every
        other 3 by 1/6 dollar each.

        Originally the positions calculated by this strategy is given with values in {0, 1, -1}. To be able to actually
        trade using the dollar neutral strategy as given by the authors in the paper, one needs to know at any given
        time how much units to hold for each stock. The result will be returned in a pd.DataFrame. The user can also
        multiply the final result by changing the multiplier input. It means by default it uses 1 dollar for
        calculation unless changed. It also means there is no reinvestment on gains.

        Note: This method assumes the 0th column in prices_df is the long unit (suppose it is called stock 1), 1st
        column the shrot unit (suppose it is called stock 2). For example, 1 in positions means buy stock 1 with 0.5
        dollar and sell stock 2 to gain 0.5 dollar.

        Note2: The short units will be given in its actual value. i.e., short 0.54 units is given as -0.54 in the
        output.

        :param prices_df: (pd.DataFrame) Prices data frame for the two securities.
        :param positions: (pd.Series) The suggested positions with values in {0, 1, -1}. Need to have the same length
            as prices_df.
        :param multiplier: (float) Optional. Multiply the calculated result by this amount. Defalts to 1.
        :return: (pd.DataFrame) The calculated positions for each security. The row and column index will be taken
            from prices_df.
        """

        target_stock_prices = prices_df.iloc[:, pv_target_idx-1]
        other_stocks_prices = prices_df.drop(columns=prices_df.columns[pv_target_idx-1])
        num_of_others = len(other_stocks_prices.columns)
        target_stock_prices_np = target_stock_prices.to_numpy(dtype=np.float16)
        other_stocks_prices_np = other_stocks_prices.to_numpy(dtype=np.float16)

        target_stock_units = pd.Series(data=0, index=target_stock_prices.index)
        other_stocks_units = pd.DataFrame(data=0, index=other_stocks_prices.index, columns=other_stocks_prices.columns)
        target_stock_units_np = target_stock_units.to_numpy(dtype=np.float16)
        other_stocks_units_np = other_stocks_units.to_numpy(dtype=np.float16)

        target_stock_units_np[0] = 0.5 / target_stock_prices_np[0] * positions[0]
        other_stocks_units_np[0, :] = - 0.5 / num_of_others /other_stocks_prices_np[0, :] * positions[0]

        nums = len(positions)
        for i in range(1, nums):
            # By default the new amount of units to be held is the same as the previous step.
            target_stock_units_np[i] = target_stock_units_np[i-1]
            other_stocks_units_np[i, :] = other_stocks_units_np[i-1, :]
            # Updating if there are position changes.
            # From not short to short.
            if positions[i-1] != -1 and positions[i] == -1:  # Short target, long cohort
                long_units = 0.5 / num_of_others / other_stocks_prices_np[i, :]
                short_units = 0.5 / target_stock_prices_np[i]
                target_stock_units_np[i] = - short_units
                other_stocks_units_np[i, :] = long_units
            # From not long to long.
            if positions[i-1] != 1 and positions[i] == 1:  # Short cohort, long target
                long_units = 0.5 / target_stock_prices_np[i]
                short_units = 0.5 / num_of_others / other_stocks_prices_np[i, :]
                target_stock_units_np[i] = long_units
                other_stocks_units_np[i, :] = - short_units
            # From long/short to none.
            if positions[i-1] != 0 and positions[i] == 0:  # Exiting
                target_stock_units_np[i] = 0
                other_stocks_units_np[i, :] = 0

        return target_stock_units_np * multiplier, other_stocks_units_np * multiplier

#%% test
import time
import arbitragelab.copula_approach.copula_strategy_basic as csb

cvinecop = CVineCop()
stocks_universe = pd.read_csv(r'D:\Apprenticeship\data\dow_stocks.csv', parse_dates=True,
                              index_col='Date')
# Fit a vine copula
# Change to returns
returns_universe = stocks_universe.pct_change()
returns_universe.iloc[0] = 0

returns_sample = returns_universe[['AAPL', 'MSFT', 'BA', 'V']]

train_returns = returns_sample.iloc[:1200]
test_returns = returns_sample.iloc[1200:]

start_time = time.time()
# Change returns to pct
CSB = csb.BasicCopulaStrategy()
train_quantiles, cdfs = CSB.to_quantile(data=train_returns)
cvine_cop = cvinecop.fit_auto(train_quantiles)
print("--- %s seconds ---" % (time.time() - start_time))
#%%
cvinecop = CVineCop(cvine_cop)
#%% Test mpi generation
import time
start_time = time.time()
cvcs = CVineCopStrat(cvinecop=cvinecop)
mpis = cvcs.calc_mpi(returns=test_returns, cdfs=cdfs, subtract_mean=True)
print("--- %s seconds ---" % (time.time() - start_time))
#%%
start_time = time.time()
positions, bband = cvcs.get_positions_bollinger(
    returns=test_returns, cdfs=cdfs, pv_target_idx=1,
    init_pos=0, past_obs=20, threshold_std=2.0, mpis=mpis,
    if_return_bollinger_band = True)
print("--- %s seconds ---" % (time.time() - start_time))
#%%
import matplotlib.pyplot as plt
plt.style.use('seaborn')
fig, ax = plt.subplots(nrows=3, dpi=300, figsize=(10, 7), gridspec_kw={'height_ratios': [0.5, 0.2, 0.7]}, sharex=True)
ax[0].plot(stocks_universe['AAPL'].iloc[1200:] / stocks_universe['AAPL'].iloc[1200], label='Target Stock Price')
ax[0].plot(stocks_universe['MSFT'].iloc[1200:] / stocks_universe['MSFT'].iloc[1200], label='Cohort Stock 2', linewidth=1)
ax[0].plot(stocks_universe['BA'].iloc[1200:] / stocks_universe['BA'].iloc[1200], label='Cohort Stock 3', linewidth=1)
ax[0].plot(stocks_universe['V'].iloc[1200:] / stocks_universe['V'].iloc[1200], label='Cohort Stock 4',linewidth=1)
ax[1].plot(positions, label='positions')
ax[2].plot(mpis.cumsum(), label='CMPI')
ax[2].plot(bband['LowerBound'], label='B Band Bound', linewidth=1, color='brown')
ax[2].plot(bband['Mean'], label='B Band Mean')
ax[2].plot(bband['UpperBound'], linewidth=1, color='brown')
plt.tight_layout()
ax[0].legend()
ax[2].legend()
fig.show()

#%% Positions to units
shifted_positions = positions.shift(1)
shifted_positions.iloc[0] = 0
target_units, others_units = cvcs.positions_to_units_cohort(
    prices_df=stocks_universe[['AAPL', 'MSFT', 'BA', 'V']].iloc[1219:], positions=shifted_positions, multiplier=4)

#%% Equity curve plot
portfolio_pnl = (returns_sample['AAPL'].iloc[1219:].to_numpy() * target_units
                 + (returns_sample[['MSFT', 'BA', 'V']].iloc[1219:].to_numpy() * others_units).sum(axis=1))
equity_pnl = pd.Series(data=portfolio_pnl.cumsum(), index=returns_sample['AAPL'].iloc[1219:].index)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=3, dpi=300, figsize=(10, 7), gridspec_kw={'height_ratios': [0.5, 0.2, 0.7]}, sharex=True)
ax[0].plot(stocks_universe['AAPL'].iloc[1200:] / stocks_universe['AAPL'].iloc[1200], label='Target Stock Price')
ax[0].plot(stocks_universe['MSFT'].iloc[1200:] / stocks_universe['MSFT'].iloc[1200], label='Cohort Stock 2', linewidth=1)
ax[0].plot(stocks_universe['BA'].iloc[1200:] / stocks_universe['BA'].iloc[1200], label='Cohort Stock 3', linewidth=1)
ax[0].plot(stocks_universe['V'].iloc[1200:] / stocks_universe['V'].iloc[1200], label='Cohort Stock 4',linewidth=1)
ax[1].plot(positions, label='positions')
ax[2].plot(equity_pnl, label='P&L', linewidth=1)

plt.tight_layout()
ax[0].legend()
ax[2].legend()
fig.show()