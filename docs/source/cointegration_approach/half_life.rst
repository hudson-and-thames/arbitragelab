.. _cointegration_approach-half_life:

===========================
Half-life of Mean-Reversion
===========================

This module contains a function that allows calculating a half-life of the mean-reversion process
under the assumption that data follows the Ornstein-Uhlenbeck process.

The Ornstein-Uhlenbeck process can be described using a formula:

.. math::

    dy(t) = ( \lambda y(t-1) + \mu ) dt + d \varepsilon

where :math:`d \varepsilon` is some Gaussian noise.

Implementation
##############

.. py:currentmodule:: arbitragelab.cointegration_approach.signals

.. autofunction:: get_half_life_of_mean_reversion

Examples
########

.. code-block::

   # Importing the function
   from arbitragelab.cointegration_approach.signals import get_half_life_of_mean_reversion

   # Finding the half-life of mean-reversion
   half_life = get_half_life_of_mean_reversion(portfolio_price)
