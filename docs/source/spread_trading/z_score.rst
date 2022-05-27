.. _spread_trading-z_score:

=========================
Spread Trading Strategies
=========================

The following are some strategies for trading the spread.

.. note::
   These descriptions closely follow the book by Ernest P. Chan:
   `Algorithmic Trading: Winning Strategies and Their Rationale <https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146>`__.

Introduction
############

The Hedge Ratios module, alongside the Spread Selection and ML Based Pairs Selection modules, helps find assets
for a well-performing strategy and construct spreads. The current module provides strategies to use on the
constructed spreads.

The trading strategies presented in this section follow a different signal generation logic than that proposed in
the previous classes of the ArbitrageLab package. These trading strategies take new spread values one by one and allow
checking if the conditions to open a position are fulfilled with each new timestamp and value provided. This allows
for easier integration of these strategies into an existing data pipeline. Also, the strategy object keeps track
of open and closed trades and the supporting information related to them.

This module will be updated in the future as other strategies will support continuous data updates.

References
##########

* `Chan, Ernie. Algorithmic trading: winning strategies and their rationale. Vol. 625. John Wiley & Sons, 2013. <https://www.wiley.com/en-us/Algorithmic+Trading:+Winning+Strategies+and+Their+Rationale-p-9781118460146>`_
