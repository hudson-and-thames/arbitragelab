.. _cointegration_approach-minimum_profit_simulation:

.. note::
    The following documentation follows closely the paper:

    - `Loss protection in pairs trading through minimum profit bounds: a cointegration approach <http://downloads.hindawi.com/archive/2006/073803.pdf>`__ by Lin, Y.-X., McCrae, M., and Gulati, C. (2006)

==================================
Simulation of Cointegratred Series
==================================

This module allows users to simulate:

- AR(1) processes
- Cointegrated series pairs where the cointegration error follows an AR(1) process

Cointegration simulations are based on the following cointegration model:

.. math::

    P_{S_1}(t) + \beta P_{S_2}(t) = \varepsilon_t

    P_{S_2}(t) - P_{S_2}(t-1) = e_t

where :math:`\varepsilon_t` and :math:`e_t` are AR(1) processes.

.. math::

    \varepsilon_t - \phi_1 \varepsilon_{t-1} = c_1 + \delta_{1,t} \qquad \delta_{1,t} \sim N(0, \sigma_1^2)

    e_t - \phi_2 e_{t-1} = c_2 + \delta_{2,t} \qquad \delta_{2,t} \sim N(0, \sigma_2^2)

The parameters :math:`\phi_1`, :math:`\phi_2`, :math:`c_1`, :math:`c_2`, :math:`\sigma_1`, :math:`\sigma_2`, and
:math:`\beta` can be defined by users.

Implementation
**************

.. automodule:: arbitragelab.cointegration_approach.minimum_profit_simulation

    .. autoclass:: MinimumProfitSimulation
        :members:
        :inherited-members:

        .. automethod:: __init__
