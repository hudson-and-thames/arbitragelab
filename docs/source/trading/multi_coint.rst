.. _trading-multi_coint:

===================================
Multivariate Cointegration Strategy
===================================

Introduction
############

This trading strategy takes new spread values one by one and allows checking if the conditions to open a position
are fulfilled with each new timestamp and value provided. This allows for easier integration of these strategies
into an existing data pipeline. Also, the strategy object keeps track of open and closed trades and the supporting
information related to them.

Multivariate Cointegration Strategy
###################################

The trading strategy logic is described in more detail in the
:ref:`Multivariate Cointegration Framework <cointegration_approach-multivariate_cointegration>`
section of the documentation.

**The trading strategy itself works as follows:**

1. Estimate the cointegration vector :math:`\hat{\mathbf{b}}` with Johansen test using training data. This step is done by the ``MultivariateCointegration`` class.
2. Construct the realization :math:`\hat{Y}_t` of the process :math:`Y_t` by calculating :math:`\hat{\mathbf{b}}^T \ln P_t`, and calculate :math:`\hat{Z}_t = \hat{Y}_t - \hat{Y}_{t-1}`.
3. Compute the finite sum :math:`\sum_{p=1}^P \hat{Z}_{t-p}`, where the lag :math:`P` is the length of a data set.
4. Partition the assets into two sets :math:`L` and :math:`S` according to the sign of the element in the cointegration vector :math:`\hat{\mathbf{b}}`.
5. Following the formulae below, calculate the number of assets to trade so that the notional of the positions would equal to :math:`C`.

.. math::

    \Bigg \lfloor \frac{-b^i C \text{ sgn} \bigg( \sum_{p=1}^{P} Z_{t-p} \bigg)}{P_t^i \sum_{j \in L} b^j} \Bigg \rfloor, \: i \in L

    \Bigg \lfloor \frac{b^i C \text{ sgn} \bigg( \sum_{p=1}^{P} Z_{t-p} \bigg)}{P_t^i \sum_{j \in L} b^j} \Bigg \rfloor, \: i \in S

.. note::

    The trading signal is determined by :math:`\sum_{p=1}^{\infty} Z_{t-p}`, which sums to time period :math:`t-1`.
    The price used to convert the notional to the number of shares/contracts to trade is the closing price of time :math:`t`.
    This ensures that no look-ahead bias will be introduced.

6. Open the positions on time :math:`t` and close the positions on time :math:`t+1`.
7. Every once in a while - once per month (22 trading days) for example, re-estimate the cointegration vector. If it is time for a re-estimate, go to step 1; otherwise, go to step 2.

The strategy is trading at daily frequency and always in the market.

The strategy object is initialized with the cointegration vector.

The ``update_price_values`` method allows adding new price values one by one - when they are available.
At each stage, the ``get_signal`` method generates the number of shares to trade per asset according to the
above-described logic. A new trade can be added to the internal dictionary using the
``add_trade`` method.

As well, the ``update_trades`` method can be used to close the previously opened trade.
If so, the internal dictionaries are updated, and the list of the closed trades at this stage is returned.

Implementation
**************

.. py:currentmodule:: arbitragelab.trading.multi_coint
.. autoclass:: MultivariateCointegrationTradingRule
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
    from arbitragelab.cointegration_approach.multi_coint import MultivariateCointegration
    from arbitragelab.trading.multi_coint import MultivariateCointegrationTradingRule

    # Using MultivariateCointegration as optimizer ...

    # Generating the cointegration vector to later use in a trading strategy
    coint_vec = optimizer.get_coint_vec()

    # Creating a strategy
    strategy = MultivariateCointegrationTradingRule(coint_vec)

    # Adding initial price values
    strategy.update_price_values(data.iloc[0])

    # Feeding price values to the strategy one by one
    for ind in range(data.shape[0]):

        time = spread.index[ind]
        value = spread.iloc[ind]

        strategy.update_price_values(value)

        # Getting signal - number of shares to trade per asset
        pos_shares, neg_shares, pos_notional, neg_notional = strategy.get_signal()

        # Close previous trade
        strategy.update_trades(update_timestamp=time)

        # Add a new trade
        strategy.add_trade(start_timestamp=time, pos_shares=pos_shares, neg_shares=neg_shares)

    # Checking currently open trades
    open_trades = strategy.open_trades

    # Checking all closed trades
    closed_trades = strategy.closed_trades

Research Notebooks
##################

The following research notebook can be used to better understand the Strategy described above.

* `Multivariate Cointegration Strategy`_

.. _`Multivariate Cointegration Strategy`: https://hudsonthames.org/notebooks/arblab/multivariate_cointegration.html

.. raw:: html

    <a href="https://hudthames.tech/3iIGDvv"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
    <a href="https://hudthames.tech/2S03R58"><button style="margin: 20px; margin-top: 0px">Download Sample Data</button></a>

References
##########

* `Lin, Y.-X., McCrae, M., and Gulati, C., 2006. Loss protection in pairs trading through minimum profit bounds: a cointegration approach <http://downloads.hindawi.com/archive/2006/073803.pdf>`_
* `Puspaningrum, H., Lin, Y.-X., and Gulati, C. M. 2010. Finding the optimal pre-set boundaries for pairs trading strategy based on cointegration technique <https://ro.uow.edu.au/cgi/viewcontent.cgi?article=1040&context=cssmwp>`_
