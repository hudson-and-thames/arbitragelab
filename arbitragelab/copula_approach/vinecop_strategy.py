# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module that generates vine copulas.

Built on top of the :code:`pyvinecoplib` package. See https://github.com/vinecopulib/pyvinecopulib for more details.
"""

from typing import List, Union, Callable, Tuple
from arbitragelab.copula_approach.vinecop_generate import CVineCop
import arbitragelab.copula_approach.copula_calculation as ccalc
import pandas as pd


class CVineCopStrat:
    """
    Trading strategy class using vine copulas.
    """

    def __init__(self, cvinecop: CVineCop = None, signal_to_position_table: pd.DataFrame = None):
        """
        Create the trading strategy class.

        :param cvinecop: (CVineCop) Optional. A fitted C-vine copula model from the vinecop_generate module.
        :param signal_to_position_table: (pd.DataFrame) Optional. The table that instructs how to generate positions
            based on current signal and the immediate past position. By default it uses the table described above.
        """

        self.cvine_cop = cvinecop
        self.past_obs = None  # Number of past observations for Bollinger band.
        self.threshold_std = None  # Standard deviation threshold for Bollinger band.
        # Signal to position table: Row indexed by previous position, column indexed by signal.
        # Refer to doc string of self._signal_to_position() for more detail.
        if signal_to_position_table is None:
            self.signal_to_position_table = pd.DataFrame({1: {0: 1, 1: 1, -1: 0},
                                                          -1: {0: -1, 1: 0, -1: -1},
                                                          0: {0: 0, 1: 0, -1: 0},
                                                          2: {0: 0, 1: 1, -1: -1}})
        else:
            self.signal_to_position_table = signal_to_position_table

    def calc_mpi(self, returns: pd.DataFrame, cdfs: List[Callable], pv_target_idx: int = 1,
                 subtract_mean: bool = False) -> pd.Series:
        """
        Calculate mispricing indices from returns for the target stock.

        Mispricing indices are technically cumulative conditional probabilities calculated from a vine copula based on
        returns data. Note that MPIs are model dependent since they are conditional probabilities inferred from vine
        copulas.

        :param returns: (pd.DataFrame) The returns data for stocks.
        :param cdfs: (pd.DataFrame) The list of cdf functions for each stocks return, used to map returns to their
            quantiles.
        :param pv_target_idx: (int) Optional. The target stock's index. Defaults to 1.
        :param subtract_mean: (bool) Optional. Whether to subtract the mean 0.5 of MPI from the calculation.
        :return: (pd.Series) The MPIs calculated from returns and vine copula.
        """

        # 1. Map all the returns to quantiles.
        quantiles = pd.DataFrame(data=0, index=returns.index, columns=returns.columns)
        for i, column_name in enumerate(returns):
            quantiles[column_name] = returns[column_name].apply(cdfs[i])

        # 2. Calculate the conditional probabilities. This step is relatively slow.
        mpis = self.cvine_cop.get_condi_probs(quantiles, pv_target_idx)

        if subtract_mean:
            mpis = mpis.subtract(0.5)

        return mpis

    def get_positions_bollinger(self, returns: pd.DataFrame, cdfs: List[Callable], pv_target_idx: int = 1,
                                init_pos: int = 0, past_obs: int = 20, threshold_std: float = 1.0,
                                mpis: pd.Series = None, if_return_bollinger_band=False) \
            -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
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
        observation 20. For live data feed you can  use self.get_cur_pos_bollinger() to save computation time.

        :param returns: (pd.DataFrame) The returns data for stocks.
        :param cdfs: (pd.DataFrame) The list of cdf functions for each stocks return, used to map returns to their
            quantiles.
        :param pv_target_idx: (int) Optional. The target stock's index. Defaults to 1.
        :param init_pos: (int) Optional. Initial trading position, 0 is None, 1 is Long, -1 is Short. Defaults to 0.
        :param past_obs: (int) Optional. The number of observations used to calculate Bollinger band. Defaults to 20.
        :param threshold_std: (float) Optional. How many std away from the running avg for the Bollinger band. Defaults
            to 1.
        :param mpis: (pd.Series) Optional. The MPI data for the target stock. Defaults to None, and it will be
            from the returns automatically. This is used to preload MPIs to save computation time.
        :param if_return_bollinger_band: (bool) Optional. Whether to return the Bollinger band data series together
            with the positions data series. Defaults to False.
        :return: (Union[pd.Series, Tuple[pd.Series]]) The MPIs calculated from returns and vine copula. Or the MPIs
            together with the Bollinger band data.
        """

        self.past_obs = past_obs  # Number of past observations for the Bollinger band.
        self.threshold_std = threshold_std  # How many std away from the running avg for the Bollinger band.

        # 1. Get the initial mean and std of CMPI
        if mpis is None:
            cmpis = self.calc_mpi(returns, cdfs, pv_target_idx, subtract_mean=True).cumsum()
        else:
            cmpis = mpis.cumsum()

        # init_mean = cmpis.iloc[:past_obs].mean()
        # init_std = cmpis.iloc[:past_obs].std()

        # 2. Get positions
        running_avg = cmpis.rolling(self.past_obs).mean()
        running_std = cmpis.rolling(self.past_obs).std()
        bollinger_band = pd.DataFrame({'LowerBound': running_avg - self.threshold_std * running_std,
                                       'Mean': running_avg,
                                       'UpperBound': running_avg + self.threshold_std * running_std})
        # Loop through the bollinger_band to get positions
        positions = pd.Series(data=None, index=bollinger_band.index)
        positions.iloc[self.past_obs - 1] = init_pos
        for idx in range(past_obs, len(positions)):
            # Get current trading signal from Bollinger band
            signal = self.get_cur_signal_bollinger(
                past_cmpi=cmpis[idx - 1], cur_cmpi=cmpis[idx],
                running_mean=bollinger_band['Mean'].iloc[idx - 1],
                upper_threshold=bollinger_band['UpperBound'].iloc[idx - 1],
                lower_threshold=bollinger_band['LowerBound'].iloc[idx - 1])
            # Translate signal to current position
            positions.iloc[idx] = self._signal_to_position(past_pos=positions.iloc[idx - 1], signal=signal)

        # 3. Return positions (and the bollinger band)
        if if_return_bollinger_band:
            return positions, bollinger_band

        return positions

    @staticmethod
    def get_cur_signal_bollinger(past_cmpi: float, cur_cmpi: float, running_mean: float,
                                 upper_threshold: float, lower_threshold: float) -> int:
        """
        Get the current trading signal based on the bollinger band over CMPIs.

        Signal types {1: long, -1: short, 0: exit, 2: do nothing}. If the current CMPI > upper threshold, then short;
        If the current CMPI < lower threshold, then long; If the current CMPI crosses with the running mean, then
        exit; else do nothing.

        :param past_cmpi: (float) The immediate past CMPI value.
        :param cur_cmpi: (float) The current CMPI value.
        :param running_mean: (float) The running average for CMPI in the bollinger band.
        :param upper_threshold: (float) The upper threshold of the bollinger band.
        :param lower_threshold: (float) The lower threshold of the bollinger band.
        :return: (int) The derived signal based on all the input info.
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

    def _signal_to_position(self, past_pos: int, signal: int) -> int:
        """
        Map the signal to position, given the past position.

        The default map is:
            signal [long, short, exit, do-nothing], acting on previous positions being [0, 1, -1] respectively

            * long signal: {0: 1, 1: 1, -1: 0}
            * short signal: {0: -1, 1: 0, -1: 0}
            * exit signal: {0: 0, 1: 0, -1: 0}
            * do-nothing signal: {0: 0, 1: 1, -1: -1}

        :param signal: (int) Current signal {1: long, -1: short, 0: exit, 2: do nothing}.
        :param past_pos: (int) The immediate past position {0: no position, 1: long, -1: short}.
        :return: (int) The current position.
        """

        # Generate a new position according to the table.
        new_position = self.signal_to_position_table.loc[past_pos, signal]

        return new_position

    def get_cur_pos_bollinger(self, returns_slice: pd.DataFrame, cdfs: List[Callable], past_pos: int,
                              pv_target_idx: int = 1, past_cmpi: float = 0, threshold_std: float = 1.0) \
            -> Tuple[int, float]:
        """
        Get the suggested position and updated bollinger band from cur_returns pandas DataFrame.

        If the dataframe has 21 rows, then the first 20 rows will be used to calculate the Bollinger band, then the
        position will be generated based on the last row. Then we calculate the updated Bollinger band. This is used
        for live data feed.

        :param returns_slice: (pd.DataFrame) The slice of the returns data frame. Everything except for the last data
            point is used to calculate the bollinger band, and the last data point is used to generate the current
            position.
        :param cdfs: (List[Callable]) The list of CDFs for the returns data.
        :param past_pos: (int) The immediate past position.
        :param pv_target_idx: (int) Optional. The target stock index. For example, 1 means the target stock is the 0th
            column in returns_slice. Defaults to 1.
        :param past_cmpi: (float) Optional. The immediate past CMPI value. Defaults to 0.
        :param threshold_std: (float) Optional. How many std away from the running avg for the Bollinger band. Defalts
            to 1.
        :return: (Tuple[int, float]) The current position and the new cmpi.
        """

        # 1. Calculate the Bollinger band: mean and std
        cmpis = past_cmpi + self.calc_mpi(returns_slice, cdfs, subtract_mean=True, pv_target_idx=pv_target_idx).cumsum()
        init_mean = cmpis.iloc[: -1].mean()
        init_std = cmpis.iloc[: -1].std()

        # 2. Get positions from the last row of the dataframe
        past_cmpi = cmpis.iloc[-2]
        cur_cmpi = cmpis.iloc[-1]
        signal = self.get_cur_signal_bollinger(past_cmpi, cur_cmpi, running_mean=init_mean,
                                               upper_threshold=init_mean + threshold_std * init_std,
                                               lower_threshold=init_mean - threshold_std * init_std)
        new_position = self._signal_to_position(past_pos, signal)

        return new_position, cur_cmpi

    @staticmethod
    def positions_to_units_against_index(target_stock_prices: pd.Series, index_prices: pd.Series, positions: pd.Series,
                                         multiplier: float = 1) -> pd.DataFrame:
        """
        Translate positions to units held for the target stock against an index fund.

        The translation is conducted under a dollar-neutral strategy against an index fund (typically SP500 index). For
        example, for a long position, for each 1 dollar investment, long the target stock by 1/2 dollar, and short the
        index fund by 1 dollar.

        Originally the positions calculated by this strategy is given with values in {0, 1, -1}. To be able to actually
        trade using the dollar neutral strategy as given by the authors in the paper, one needs to know at any given
        time how much units to hold for the stock. The result will be returned in a pd.DataFrame. The user can also
        multiply the final result by changing the multiplier input. It means by default it uses 1 dollar for
        calculation unless changed. It also means there is no reinvestment on gains.

        Note: The short units will be given in its actual value. i.e., short 0.54 units is given as -0.54 in the
        output.

        :param target_stock_prices: (pd.Series) The target stock's price series.
        :param index_prices: (pd.Series) The index fund's price.
        :param positions: (pd.Series) The positions suggested by the strategy in integers.
        :param multiplier: (float) Optional. Multiply the calculated result by this amount. Defaults to 1.
        :return: (pd.DataFrame) The units to hold for the target stock and the index fund.
        """

        units_df = pd.DataFrame(data=0, index=target_stock_prices.index, columns=[target_stock_prices.name,
                                                                                  index_prices.name])
        prices_df = pd.concat([target_stock_prices, index_prices], axis=1)

        units_df.iloc[0, 0] = 0.5 / prices_df.iloc[0, 0] * positions[0]
        units_df.iloc[0, 1] = - 0.5 / prices_df.iloc[1, 0] * positions[0]
        for i in range(1, len(positions)):
            # By default the new amount of units to be held is the same as the previous step.
            units_df.iloc[i, :] = units_df.iloc[i - 1, :]
            # Updating if there are position changes.
            # From not short to short.
            if positions[i - 1] != -1 and positions[i] == -1:  # Short 1, long 2
                long_units = 0.5 / prices_df.iloc[i, 1]
                short_units = 0.5 / prices_df.iloc[i, 0]
                units_df.iloc[i, 0] = - short_units
                units_df.iloc[i, 1] = long_units
            # From not long to long.
            if positions[i - 1] != 1 and positions[i] == 1:  # Short 2, long 1
                long_units = 0.5 / prices_df.iloc[i, 0]
                short_units = 0.5 / prices_df.iloc[i, 1]
                units_df.iloc[i, 0] = long_units
                units_df.iloc[i, 1] = - short_units
            # From long/short to none.
            if positions[i - 1] != 0 and positions[i] == 0:  # Exiting
                units_df.iloc[i, 0] = 0
                units_df.iloc[i, 1] = 0

        return units_df.multiply(multiplier)
