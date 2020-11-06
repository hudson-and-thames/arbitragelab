.. _other_approaches-introduction:

.. Note::

    These descriptions closely follow a book by Ernest P. Chan:
    `Algorithmic Trading: Winning Strategies and Their Rationale <https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146>`__.

============
Introduction
============

Kalman Filter
#############

While for truly cointegrating price series we can use the tools described in the Mean Reversion
section. For real price series we might want to use other tools to estimate the hedge ratio, as
the cointegration property can be hard to achieve or the hedge ratio it can be changing in time.

Using a look-back period, as in the Mean Reversion approaches to estimate the parameters of a model
has its disadvantages, as a short period can cut a part of the information. Then we might improve these
methods by using an exponential weighting of observations, but it's not obvious if this weighting is
optimal either.

.. figure:: images/kalman_cumulative_returns.png
    :scale: 80 %
    :align: center

    Cumulative returns of Kalman Filter Strategy on a EWC-EWA pair.
    An example from `"Algorithmic Trading: Winning Strategies and Their Rationale" <https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146>`__
    by Ernest P. Chan.

This module describes a scheme that allows using the Kalman filter for hedge ratio updating, as presented in the
book by Ernest P. Chan "Algorithmic Trading: Winning Strategies and Their Rationale". One of the advantages
of this approach is that we don't have to pick a weighting scheme for observations in the look-back period.
Based on this scheme a Kalman Filter Mean Reversion Strategy can be created, which it is also described in
this module.
