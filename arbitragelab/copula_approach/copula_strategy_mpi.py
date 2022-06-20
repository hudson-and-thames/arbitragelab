# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
Module that uses copula for trading strategy based on (cumulative) mispricing index.

`Xie, W., Liew, R.Q., Wu, Y. and Zou, X., 2014. Pairs Trading with Copulas.
<https://efmaefm.org/0efmameetings/EFMA%20ANNUAL%20MEETINGS/2014-Rome/papers/EFMA2014_0222_FullPaper.pdf>`__
"""

# pylint: disable = invalid-name, too-many-locals, dangerous-default-value
from typing import Callable, Sequence, Union
import numpy as np
import pandas as pd

from arbitragelab.copula_approach.copula_strategy_basic import BasicCopulaStrategy
import arbitragelab.copula_approach.base as cop
import arbitragelab.copula_approach.mixed_copulas.base as copmix
from arbitragelab.util import segment


class CopulaStrategyMPI(BasicCopulaStrategy):
    """
    Copula trading strategy based on mispricing index(MPI).

    This strategy uses mispricing indices from a pair of stocks to form positions.
    It is more specific than the original BacicCopulaStrategy as its logic is built upon the usage of return series,
    not price series from stocks. Indeed, it uses flag series, defined as the cumulative centered mispricing index,
    with certain reset conditions to form positions. A very important note is that, flag series are not uniquely
    defined based on the authors' description. In some cases the reset conditions depends on whether the reset priority
    is higher or opening a position priority is higher. In this implementation as CopulaStrategyMPI, the reset priority
    is the highest. If one wishes to change the precedence, it is in method _get_position_and_reset_flag.

    Compared to the original BasicCopulaStrategy class, it includes the following fundamental functionalities:

        1. Convert price series to return series.
        2. Calculate MPI and flags (essentially cumulative mispricing index).
        3. Use flags to form positions.

    """

    def __init__(self, copula: Union[cop.Copula, copmix.MixedCopula] = None,
                 opening_triggers: tuple = (-0.6, 0.6), stop_loss_positions: tuple = (-2, 2)):
        """
        Initiate a CopulaStrategyMPI class.

        One can choose to initiate with no arguments, or to initiate with a given copula as the system's
        Copula.

        :param copula: (Copula, MixedCopula) Optional. A copula object that the class will use for all analysis. If
            there is no input then fit_copula method will create one when called.
        :param opening_triggers: (tuple) Optional. The thresholds for MPI to trigger a long/short position for the
            pair's trading framework. Format is (long trigger, short trigger). Defaults to (-0.6, 0.6).
        :param stop_loss_positions: (tuple) Optional. One of the conditions for MPI to trigger an exiting
            trading signal. Defaults to (-2, 2).
        """

        super().__init__(copula=copula)
        self.opening_triggers = opening_triggers
        self.stop_loss_positions = stop_loss_positions

        # Counters on how many times each position is triggered.
        self._long_count = 0
        self._short_count = 0
        self._exit_count = 0

        segment.track('CopulaStrategyMPI')

    @staticmethod
    def to_returns(pair_prices: pd.DataFrame, fill_init_nan: Sequence[float] = (0, 0)) -> pd.DataFrame:
        r"""
        Convert a pair's prices DataFrame to its returns DataFrame.

        Returns (excess) defined as: r(t) = P(t) / P(t-1) - 1.

        Note that the 0th row will be NaN value, and needs to be filled.

        :param pair_prices: (pd.DataFrame) Prices data frame of the stock pair.
        :param fill_init_nan: (Sequence[float]) Optional. What to fill the NaN value at the initial row. Defaults
            to (0, 0).
        :return: (pd.DataFrame) Returns data frame for the stock pair.
        """

        returns = pair_prices.pct_change()
        returns.iloc[0, 0] = fill_init_nan[0]
        returns.iloc[0, 1] = fill_init_nan[1]

        return returns

    def calc_mpi(self, returns: pd.DataFrame, cdf1: Callable[[float], float],
                 cdf2: Callable[[float], float]) -> pd.DataFrame:
        r"""
        Calculate mispricing indices from returns.

        Mispricing indices are technically cumulative conditional probabilities calculated from a copula based on
        returns data. i.e., MPI_1(r1, r2) = P(R1 <= r1 | R2 = r2), where r1, r2 are the value of returns for two stocks.
        Similarly MPI_2(r1, r2) = P(R2 <= r2 | R1 = r1).

        :param returns: (pd.DataFrame) Return data frame for the stock pair.
        :param cdf1: (func) Cumulative density function for stock 1's returns series.
        :param cdf2: (func) Cumulative density function for stock 2's returns series.
        :return: (pd.DataFrame) Mispricing indices for the pair of stocks.
        """

        # Convert to quantile data
        quantile_c1 = returns.iloc[:, 0].map(cdf1)
        quantile_c2 = returns.iloc[:, 1].map(cdf2)
        quantile_data = pd.concat([quantile_c1, quantile_c2], axis=1)
        # Calculate conditional probabilities using returns and cdfs. This is the definition of MPI.
        mpis = super().get_condi_probs(quantile_data)

        return mpis

    @staticmethod
    def positions_to_units_dollar_neutral(prices_df: pd.DataFrame, positions: pd.Series,
                                          multiplier: float = 1) -> pd.DataFrame:
        """
        Change the positions series into units held for each security for a dollar neutral strategy.

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

        units_df = pd.DataFrame(data=0, index=prices_df.index, columns=prices_df.columns)
        units_df.iloc[0, 0] = 0.5 / prices_df.iloc[0, 0] * positions[0]
        units_df.iloc[0, 1] = - 0.5 / prices_df.iloc[1, 0] * positions[0]
        nums = len(positions)
        for i in range(1, nums):
            # By default the new amount of units to be held is the same as the previous step.
            units_df.iloc[i, :] = units_df.iloc[i-1, :]
            # Updating if there are position changes.
            # From not short to short.
            if positions[i-1] != -1 and positions[i] == -1:  # Short 1, long 2
                long_units = 0.5 / prices_df.iloc[i, 1]
                short_units = 0.5 / prices_df.iloc[i, 0]
                units_df.iloc[i, 0] = - short_units
                units_df.iloc[i, 1] = long_units
            # From not long to long.
            if positions[i-1] != 1 and positions[i] == 1:  # Short 2, long 1
                long_units = 0.5 / prices_df.iloc[i, 0]
                short_units = 0.5 / prices_df.iloc[i, 1]
                units_df.iloc[i, 0] = long_units
                units_df.iloc[i, 1] = - short_units
            # From long/short to none.
            if positions[i-1] != 0 and positions[i] == 0:  # Exiting
                units_df.iloc[i, 0] = 0
                units_df.iloc[i, 1] = 0

        return units_df.multiply(multiplier)

    def get_positions_and_flags(self, returns: pd.DataFrame,
                                cdf1: Callable[[float], float], cdf2: Callable[[float], float],
                                init_pos: int = 0, enable_reset_flag: bool = True,
                                open_rule: str = 'or', exit_rule: str = 'or', opening_triggers: tuple = None,
                                stop_loss_positions: tuple = None) -> (pd.Series, pd.DataFrame):
        """
        Get the positions and flag series based on returns series.

        Flags are defined as the accumulative, corrected MPIs. i.e., flag(t) = flag(t-1) + (mpi(t)-0.5). Note that flags
        reset when an exiting signal is present, so it is not a markov chain, a.k.a. it depends on history.
        This method at first calculates the MPIs based on return series. Then it loops through the mpi series to form
        flag series and positions. Suppose the upper opening trigger is D_u and the lower opening trigger is D_l, the
        stop-loss has upper threshold slp_u and lower threshold slp_l.

        For the open OR and exit OR logic (method default) as described in [Xie et al. 2014], it goes as follows:

            - If flag1 >= D_u, short stock 1 and long stock 2. i.e., position = -1;
            - If flag1 <= D_l, short stock 2 and long stock 1. i.e., position = 1;
            - If flag1 >= D_u, short stock 2 and long stock 1. i.e., position = 1;
            - If flag1 >= D_l, short stock 1 and long stock 2. i.e., position = -1;

            - If trades are open based on flag1, then exit if flag1 returns to 0, or reaches slp_u or slp_l;
            - If trades are open based on flag2, then exit if flag2 returns to 0, or reaches slp_u or slp_l;

            - Once an exit trigger is activated, then BOTH flag1 and flag2 are reset to 0.

        We also implemented OR-AND, AND-OR, AND-AND options for open-exit logic. For all those three methods, it does
        not keep track which stock opened the position, since it makes no logical sense. The AND-OR logic is the one
        used more often in other literatures such as [Rad et al. 2016], and is much more stable.

        Note 1: The original description of the strategy in the paper states that the position should be interpreted as
        dollar neutral. i.e., buying stock A and sell B in equal dollar amounts. Here in this class we do not have this
        feature built-in to calculate ratios for forming positions and we still use -1, 1, 0 to indicate short, long
        and no position, as we think it offers better flexibility for the user to choose.

        Note 2: The positions calculated on a certain day are corresponds to information given on *THAT DAY*. Thus for
        forming an equity curve, backtesting or actual trading, one should forward-roll the position by at least 1.

        :param returns: (pd.DataFrame) Returns data frame for the stock pair.
        :param cdf1: (func) Cumulative density function for stock 1's returns series.
        :param cdf2: (func) Cumulative density function for stock 2's returns series.
        :param init_pos: (int) Optional. Initial position. Takes value 0, 1, -1, corresponding to no
            position, long or short. Defaults to 0.
        :param enable_reset_flag: (bool) Optional. Whether allowing the flag series to be reset by
            exit triggers. Defaults to True.
        :param open_rule: (str) Optional. The logic for deciding to open a position from combining mispricing info from
            the two stocks. Choices are ['and', 'or']. 'and' means both stocks need to be mispriced to justify an
            opening. 'or' means only one stock need to be mispriced to open a position. Defaults to 'or'.
        :param exit_rule: (str) Optional. The logic for deciding to exit a position from combining mispricing info from
            the two stocks. Choices are ['and', 'or']. 'and' means both stocks need to be considered to justify an
            exit. 'or' means only one stock need to be considered to exit a position. Defaults to 'or'.
        :param opening_triggers: (tuple) Optional. The thresholds for MPI to trigger a long/short position for the
            pair's trading framework. Format is (long trigger, short trigger). Defaults to (-0.6, 0.6).
        :param stop_loss_positions: (tuple) Optional. One of the conditions for MPI to trigger an exiting
            trading signal. Defaults to (-2, 2).
        :return: (pd.Series, pd.DataFrame)
            The calculated position series in a pd.Series, and the two flag series in a pd.DataFrame.
        """

        # Initialization
        # Update default opening triggers
        if opening_triggers is not None:
            self.opening_triggers = opening_triggers
        # Update default stop loss positions
        if stop_loss_positions is not None:
            self.stop_loss_positions = stop_loss_positions

        open_based_on = [0, 0]  # Initially no position was opened based on stocks.
        mpis = self.calc_mpi(returns, cdf1, cdf2)  # Mispricing indices from stock 1 and 2.
        flags = pd.DataFrame(data=0, index=returns.index, columns=returns.columns)  # Initialize flag values.
        positions = pd.Series(data=[np.nan]*len(returns), index=returns.index)
        positions[0] = init_pos
        # Reset the counters
        self._long_count, self._short_count, self._exit_count = 0, 0, 0

        # Calculate positions and flags
        for i in range(1, len(returns)):
            mpi = mpis.iloc[i, :]
            pre_flag = flags.iloc[i - 1, :]
            pre_position = positions[i - 1]

            cur_flag, cur_position, open_based_on = \
                self._cur_flag_and_position(mpi, pre_flag, pre_position, open_based_on, enable_reset_flag,
                                            open_rule, exit_rule)
            flags.iloc[i, :] = cur_flag
            positions[i] = cur_position

        return positions, flags

    def _cur_flag_and_position(self, mpi: pd.Series, pre_flag: pd.Series, pre_position: int,
                               open_based_on: list, enable_reset_flag: bool,
                               open_rule: str, exit_rule: str) -> (pd.Series, int, list):
        """
        Get the current flag value and position for the two stocks.

        :param mpi: (pd.Series) The pair of mispricing indices from the stocks pair for the current time.
        :param pre_flag: (pd.Series) The pair of flag values from the stocks pair for the immediate previous time.
        :param pre_position: (pd.Series) The pair of positions from the stocks pair for the immediate previous time.
        :param open_based_on: (list) Len 2 list describing which stock did the current long or short position based on.
            position 0 takes value 1, -1, 0: 1 means long, -1 means short, 0 means no position.
            position 1 takes value 1, 2, 0: 1 means from stock 1, 2 means from stock 2, 0 means no position.
        :param enable_reset_flag: (bool) Optional. Whether allowing the flag series to be reset by
            exit triggers. Defaults to True.
        :param open_rule: (str) The logic for deciding to open a position from combining mispricing info from
            the two stocks. Choices are ['and', 'or']. 'and' means both stocks need to be mispriced to justify an
            opening. 'or' means only one stock need to be mispriced to open a position. Defaults to 'or'.
        :param exit_rule: (str) The logic for deciding to exit a position from combining mispricing info from
            the two stocks. Choices are ['and', 'or']. 'and' means both stocks need to be considered to justify an
            exit. 'or' means only one stock need to be considered to exit a position. Defaults to 'or'.
        :return: (pd.Series, int, list)
            Flag value at the current time.
            Current position.
            Updated open_based_on history information.
        """

        centered_mpi = mpi.subtract(0.5)  # Center to 0
        # Raw value means it is not (potentially) reset by exit triggers.
        raw_cur_flag = centered_mpi + pre_flag  # Definition.

        cur_position, if_reset_flag, open_based_on = self._get_position_and_reset_flag(
            pre_flag, raw_cur_flag, pre_position, open_rule, exit_rule, open_based_on)

        # if if_reset_flag: reset.
        # if not if_reset_flag: do nothing.
        cur_flag = raw_cur_flag  # If not enable flag reset, then current flag value is just its raw value.
        if enable_reset_flag:
            cur_flag = raw_cur_flag * int(not if_reset_flag)

        return cur_flag, cur_position, open_based_on

    def _get_position_and_reset_flag(self, pre_flag: pd.Series, raw_cur_flag: pd.Series,
                                     pre_position: int, open_rule: str, exit_rule: str,
                                     open_based_on: list = [0, 0],) -> (int, bool, list):
        """
        Get the next position, and check if one should reset the flag. Suppose the upper opening trigger is D_u and the
        lower opening trigger is D_l, the stop-loss has upper threshold slp_u and lower threshold slp_l.

        For the open OR and exit OR logic (method default) as described in [Xie et al. 2014], it goes as follows:

            - If flag1 >= D_u, short stock 1 and long stock 2. i.e., position = -1;
            - If flag1 <= D_l, short stock 2 and long stock 1. i.e., position = 1;
            - If flag1 >= D_u, short stock 2 and long stock 1. i.e., position = 1;
            - If flag1 >= D_l, short stock 1 and long stock 2. i.e., position = -1;

            - If trades are open based on flag1, then exit if flag1 returns to 0, or reaches slp_u or slp_l;
            - If trades are open based on flag2, then exit if flag2 returns to 0, or reaches slp_u or slp_l;

            - Once an exit trigger is activated, then BOTH flag1 and flag2 are reset to 0.

        We also implemented OR-AND, AND-OR, AND-AND options for open-exit logic. For all those three methods, it does
        not keep track which stock opened the position, since it makes no logical sense. The AND-OR logic is the one
        used more often in other literatures such as [Rad et al. 2016], and is much more stable. Thus you do not need
        to input value for open_based_on.

        :param pre_flag: (pd.Series) The pair of flag values from the stocks pair for the immediate previous time.
        :param raw_cur_flag: (pd.Series) The pair of raw flag values from the stocks pair for the current time. It is
            raw value because it is not (potentially) corrected by an exit trigger.
        :param pre_position: (pd.Series) The pair of positions from the stocks pair for the immediate previous time.
        :param open_based_on: (list) Len 2 list describing which stock did the current long or short position based on.
            position 0 takes value 1, -1, 0: 1 means long, -1 means short, 0 means no position.
            position 1 takes value 1, 2, 0: 1 means from stock 1, 2 means from stock 2, 0 means no position.
            This value is only needed if the open-exit logic is OR-OR.
        :param open_rule: (str) The logic for deciding to open a position from combining mispricing info from
            the two stocks. Choices are ['and', 'or']. 'and' means both stocks need to be mispriced to justify an
            opening. 'or' means only one stock need to be mispriced to open a position. Defaults to 'or'.
        :param exit_rule: (str) The logic for deciding to exit a position from combining mispricing info from
            the two stocks. Choices are ['and', 'or']. 'and' means both stocks need to be considered to justify an
            exit. 'or' means only one stock need to be considered to exit a position. Defaults to 'or'.
        :return: (int, bool, list)
            Suggested current position.
            Whether to reset the flag.
            Updated open_based_on.
        """

        flag_1 = raw_cur_flag[0]
        flag_2 = raw_cur_flag[1]
        lower_open_threshold = self.opening_triggers[0]
        upper_open_threshold = self.opening_triggers[1]

        # Check if positions should be open. If so, based on which stock.
        # Uncomment for the next four lines to allow for openinig positions only when there's no position currently.
        long_based_on_1 = (flag_1 <= lower_open_threshold)  # and (pre_position == 0)
        long_based_on_2 = (flag_2 >= upper_open_threshold)  # and (pre_position == 0)
        short_based_on_1 = (flag_1 >= upper_open_threshold)  # and (pre_position == 0)
        short_based_on_2 = (flag_2 <= lower_open_threshold)  # and (pre_position == 0)

        # Forming triggers, OR open logic
        if open_rule == 'or':
            long_trigger = (long_based_on_1 or long_based_on_2)
            short_trigger = (short_based_on_1 or short_based_on_2)
        # Forming triggers, AND open logic
        if open_rule == 'and':
            long_trigger = (long_based_on_1 and long_based_on_2)
            short_trigger = (short_based_on_1 and short_based_on_2)
        exit_trigger = self._exit_trigger_mpi(pre_flag, raw_cur_flag, open_based_on, open_rule, exit_rule)
        any_trigger = any([long_trigger, short_trigger, exit_trigger])
        # Updating trigger counts.
        self._long_count += int(long_trigger)
        self._short_count += int(short_trigger)
        self._exit_count += int(exit_trigger)

        # Updating open_based_on variable.
        # This is only useful when used with OR-OR logic for open and exit. In other cases please ignore it.
        # Logic (and precedence. The sequence at which below are executed has influence on flag values.):
        # if long_based_on_1:
        #     open_based_on = [1, 1]
        # if short_based_on_1:
        #     open_based_on = [-1, 1]
        # if long_based_on_2:
        #     open_based_on = [1, 2]
        # if short_based_on_2:
        #     open_based_on = [-1, 2]
        # if exit_trigger:
        #     open_based_on = [0, 0]
        open_exit_triggers = [long_based_on_1, short_based_on_1, long_based_on_2, short_based_on_2, exit_trigger]
        open_based_on_values = [[1, 1], [-1, 1], [1, 2], [-1, 2], [0, 0]]
        for i in range(5):
            if open_exit_triggers[i]:
                open_based_on = open_based_on_values[i]

        # Update positions. Defaults to previous position unless there is a trigger to update it.
        cur_position = pre_position
        # Updating logic:
        # If there is a long trigger, take long position (1);
        # If there is a short trigger, take short position (-1);
        # If there is an exit trigger, take no position (0).
        if any_trigger:
            cur_position = (int(long_trigger) - int(short_trigger)) * int(not exit_trigger)

        # When there is an exit_trigger, we reset the flag value.
        if_reset_flag = exit_trigger

        return cur_position, if_reset_flag, open_based_on

    def _exit_trigger_mpi(self, pre_flag: pd.Series, raw_cur_flag: pd.Series, open_based_on: list,
                          open_rule: str, exit_rule: str) -> bool:
        """
        Check if the exit signal is triggered.

        The exit signal will be triggered:

            - If trades are open based on flag1, then exit if flag1 returns to 0, or reaches slp_u or slp_l;
            - If trades are open based on flag2, then exit if flag2 returns to 0, or reaches slp_u or slp_l;

        :param pre_flag: (pd.Series) The pair of flag values from the stocks pair for the immediate previous time.
        :param raw_cur_flag: (pd.Series) The pair of raw flag values from the stocks pair for the current time. It is
            raw value because it is not (potentially) corrected by an exit trigger.
        :param open_based_on: (int) Len 2 list describing which stock did the current long or short position based on.
            position 0 takes value 1, -1, 0: 1 means long, -1 means short, 0 means no position.
            position 1 takes value 1, 2, 0: 1 means from stock 1, 2 means from stock 2, 0 means no position.
        :param open_rule: (str) The logic for deciding to open a position from combining mispricing info from
            the two stocks. Choices are ['and', 'or']. 'and' means both stocks need to be mispriced to justify an
            opening. 'or' means only one stock need to be mispriced to open a position. Defaults to 'or'.
        :param exit_rule: (str) The logic for deciding to exit a position from combining mispricing info from
            the two stocks. Choices are ['and', 'or']. 'and' means both stocks need to be considered to justify an
            exit. 'or' means only one stock need to be considered to exit a position. Defaults to 'or'.
        :return: (bool) The exit trigger.
        """

        pre_flag_1 = pre_flag[0]  # Previous flag1 value
        pre_flag_2 = pre_flag[1]  # Previous flag2 value
        raw_cur_flag_1 = raw_cur_flag[0]  # Current raw flag1 value
        raw_cur_flag_2 = raw_cur_flag[1]  # Current raw flag2 value

        slp_lower = self.stop_loss_positions[0]  # Lower end of stop loss value for flag.
        slp_upper = self.stop_loss_positions[1]  # Upper end of stop loss value for flag.

        # Check if crossing 0 from above & below 0
        stock_1_x_from_above = (pre_flag_1 > 0 >= raw_cur_flag_1)  # flag1 crossing 0 from above.
        stock_1_x_from_below = (pre_flag_1 < 0 <= raw_cur_flag_1)  # flag1 crossing 0 from below.
        stock_2_x_from_above = (pre_flag_2 > 0 >= raw_cur_flag_2)  # flag2 crossing 0 from above.
        stock_2_x_from_below = (pre_flag_2 < 0 <= raw_cur_flag_2)  # flag2 crossing 0 from below.

        # Check if current flag reaches stop-loss positions.
        # If flag >= slp_upper or flag <= slp_lower, then it reaches the stop-loss position.
        stock_1_stop_loss = (raw_cur_flag_1 <= slp_lower or raw_cur_flag_1 >= slp_upper)
        stock_2_stop_loss = (raw_cur_flag_2 <= slp_lower or raw_cur_flag_2 >= slp_upper)

        # Determine whether one should exit the current open position.
        exit_trigger = None
        # Case: open OR, exit OR (method in the paper [Xie et al. 2014])
        # If trades were open based on flag1, then they are closed if flag1 returns to 0, or reaches stop loss
        # position. Same for flag2. Thus in total there are 4 possibilities:
        # 1. If current pos is long based on 1: flag 1 returns to 0 from below, or reaches stop loss.
        # 2. If current pos is short based on 1: flag 1 returns to 0 from above, or reaches stop loss.
        # 3. If current pos is long based on 2: flag 2 returns to 0 from below, or reaches stop loss.
        # 4. If current pos is short based on 2: flag 2 returns to 0 from above, or reaches stop loss.
        # Hence, as long as 1 of the 4 exit condition is satisfied, we exit.
        if open_rule == 'or' and exit_rule == 'or':
            exit_based_on_1 = any([open_based_on == [1, 1] and (stock_1_x_from_below or stock_1_stop_loss),
                                   open_based_on == [-1, 1] and (stock_1_x_from_above or stock_1_stop_loss)])
            exit_based_on_2 = any([open_based_on == [1, 2] and (stock_2_x_from_above or stock_2_stop_loss),
                                   open_based_on == [-1, 2] and (stock_2_x_from_below or stock_2_stop_loss)])
            exit_trigger = (exit_based_on_1 or exit_based_on_2)
            return exit_trigger

        exit_for_1 = any([stock_1_x_from_below or stock_1_stop_loss, stock_1_x_from_above or stock_1_stop_loss])
        exit_for_2 = any([stock_2_x_from_above or stock_2_stop_loss, stock_2_x_from_below or stock_2_stop_loss])
        # Case: open AND, exit OR (method in the paper [Rad et al. 2016])
        # In this case, it makes no sense to have the open_based_on variable. So we are just directly looking at the
        # thresholds. If the flag1 OR flag2 series reaches the thresholdsm, then exit.
        if open_rule == 'and' and exit_rule == 'or':
            exit_trigger = exit_for_1 or exit_for_2

        # Case: open AND or OR, exit OR (method in the paper [Rad et al. 2016])
        # In this case, it makes no sense to have the open_based_on variable. So we are just directly looking at the
        # thresholds. If the flag1 AND flag2 series reaches the thresholdsm, then exit.
        if exit_rule == 'and':
            exit_trigger = exit_for_1 and exit_for_2

        return exit_trigger
