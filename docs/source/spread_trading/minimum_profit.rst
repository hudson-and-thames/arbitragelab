.. _spread_trading-minimum_profit:

=======================
Minimum Profit Strategy
=======================

Introduction
############

This trading strategy takes new spread values one by one and allows checking if the conditions to open a position
are fulfilled with each new timestamp and value provided. This allows for easier integration of these strategies
into an existing data pipeline. Also, the strategy object keeps track of open and closed trades and the supporting
information related to them.

Minimum Profit Strategy
#######################

The Minimum Profit class from the :ref:`Cointegration Approach <cointegration_approach-minimum_profit>` generates
optimal entry and exit levels (``buy_level``, ``close_level``, ``sell_level``) as well as number of
shares (``shares_A``, ``shares_B``) to trade per asset in a cointegrated pair. The strategy described in this
section of the documentation can be used to trade those generated signals.

The signal generation process relies on the performance of the spread, which is calculated as:

.. math::

    Spread = PriceA + beta * PriceB

Where the :math:`beta` parameter is given by the Minimum Profit class. If the spread value drops below
the ``buy_level``, a buy trade should be opened, meaning we should go ``shares_A`` long and ``shares_B`` short.
If the spread value rises above the ``sell_level``, a sell trade should be opened, so we short ``shares_B`` and
go long ``shares_A``. Upon reaching the ``close_level``, a buy or a sell trade should be closed. This strategy
assumes only one open trade at a time.

.. figure:: images/AME-DOV.png
    :width: 100 %
    :align: center

    An example of pair trading Ametek Inc. (AME) and Dover Corp. (DOV) from January 2nd, 2019 to date. The green line is the sell level, the red line is the buy level, and the black line is the close level.

The strategy object is initialized with a number of shares and optimal levels.

The ``update_spread_value`` method allows adding new spread values one by one - when they are available.
At each stage, the ``check_entry_signal`` method checks if the trade should be entered according to the
above-described logic. If the trade should be opened, it can be added to the internal dictionary using the
``add_trade`` method.

As well, the ``update_trades`` method can be used to check if any trades should be closed.
If so, the internal dictionaries are updated, and the list of the closed trades at this stage is returned.

Implementation
**************

.. py:currentmodule:: arbitragelab.spread_trading.minimum_profit
.. autoclass:: MinimumProfitTradingRule
    :members:
    :inherited-members:

    .. automethod:: __init__

Example
*******

.. code-block::

    # Importing packages
    import pandas as pd
    import numpy as np

    # Importing ArbitrageLab tools
    from arbitragelab.cointegration_approach.minimum_profit import MinimumProfit
    from arbitragelab.spread_trading import MinimumProfitTradingRule

    # Using MinimumProfit as optimizer ...

    # Generate optimal trading levels and number of shares to trade
    num_of_shares, optimal_levels = optimizer.get_optimal_levels(optimal_ub,
                                                                 minimum_profit,
                                                                 beta_eg,
                                                                 epsilon_t_eg)

    # Calculating spread
    spread = optimizer.construct_spread(data, beta_eg)

    # Creating a strategy
    strategy = MinimumProfitTradingRule(num_of_shares, optimal_levels)

    # Adding initial spread value
    strategy.update_spread_value(spread[0])

    # Feeding spread values to the strategy one by one
    for time, value in spread.iteritems():
        strategy.update_spread_value(value)

        # Checking if logic for opening a trade is triggered
        trade, side = strategy.check_entry_signal()

        # Adding a trade if we decide to trade signal
        if trade:
            strategy.add_trade(start_timestamp=time, side_prediction=side)

        # Update trades, close if logic is triggered
        close = strategy.update_trades(update_timestamp=time)

    # Checking currently open trades
    open_trades = strategy.open_trades

    # Checking all closed trades
    closed_trades = strategy.closed_trades

Research Notebooks
##################

The following research notebook can be used to better understand the Strategy described above.

* `Minimum Profit Optimization`_

.. _`Minimum Profit Optimization`: https://hudsonthames.org/notebooks/arblab/minimum_profit_optimization.html

.. raw:: html

    <a href="https://hudthames.tech/3iIGDvv"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
    <a href="https://hudthames.tech/2S03R58"><button style="margin: 20px; margin-top: 0px">Download Sample Data</button></a>

References
##########

* `Lin, Y.-X., McCrae, M., and Gulati, C., 2006. Loss protection in pairs trading through minimum profit bounds: a cointegration approach <http://downloads.hindawi.com/archive/2006/073803.pdf>`_
* `Puspaningrum, H., Lin, Y.-X., and Gulati, C. M. 2010. Finding the optimal pre-set boundaries for pairs trading strategy based on cointegration technique <https://ro.uow.edu.au/cgi/viewcontent.cgi?article=1040&context=cssmwp>`_
