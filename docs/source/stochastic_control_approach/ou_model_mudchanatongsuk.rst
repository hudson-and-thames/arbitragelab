.. _stochastic_control_approach_strategies-ou_model_mudchanatongsuk:

.. note::
   The following implementations and documentation closely following work:

    `Mudchanatongsuk, S., Primbs, J.A. and Wong, W., 2008, June. Optimal pairs trading: A stochastic control approach.
    <http://folk.ntnu.no/skoge/prost/proceedings/acc08/data/papers/0479.pdf>`__



========================
OU Model Mudchanatongsuk
========================

Introduction
############

In this module, we implement a stochastic control based approach to the problem of pairs trading.
The paper models the log-relationship between a pair of stock prices as an Ornstein-Uhlenbeck process
and use this to formulate a portfolio optimization based stochastic control problem.
This problem is constructed in such a way that one may either
trade based on the spread (by buying and selling equal amounts of the stocks in the pair) or
place money in a risk free asset. Then the optimal solution to this control problem
is obtained in closed form via the corresponding Hamilton-Jacobi-Bellman equation under a power utility on terminal wealth.

Modelling
#########

Let :math:`A(t)` and :math:`B(t)` denote respectively the prices of the
pair of stocks :math:`A` and :math:`B` at time :math:`t`. We assume that stock :math:`B`
follows a geometric Brownian motion,

.. math::

    d B(t)=\mu B(t) d t+\sigma B(t) d Z(t)

where :math:`\mu` is the drift, :math:`\sigma` is the volatility, and :math:`Z(t)` is a standard
Brownian motion.

Let :math:`X(t)` denote the spread of the two stocks at time :math:`t`,
defined as

.. math::

    X(t) = \ln(A(t)) − \ln(B(t))

We assume that the spread follows an Ornstein-Uhlenbeck process

.. math::

    d X(t)=k(\theta-X(t)) d t+\eta d W(t)

where :math:`k` is the rate of reversion, :math:`\eta` is the standard deviation and
:math:`\theta` is the long-term equilibrium level to which the spread reverts.

:math:`\rho` denotes the instantaneous correlation coefficient between :math:`Z(t)` and :math:`W(t)`.

Let :math:`V(t)` be the value of a self-financing pairs-trading portfolio and
let :math:`h(t)` and :math:`-h(t)` denote respectively the
portfolio weights for stocks :math:`A` and :math:`B` at time :math:`t`.


The wealth dynamics of the portfolio value is given by,

.. math::
    d V(t)= V(t)\left\{\left[h(t)\left(k(\theta-X(t))+\frac{1}{2} \eta^{2}+\rho \sigma \eta\right)+
    r\right] d t+\eta d W(t)\right\}



Given below is the formulation of the portfolio optimization pair-trading problem
as a stochastic optimal control problem. We assume that an investor’s preference
can be represented by the utility function :math:`U(x) = \frac{1}{\gamma} x^\gamma`
with :math:`x ≥ 0` and :math:`\gamma < 1`. In this formulation, our objective is to maximize expected utility at
the final time :math:`T`. Thus, we seek to solve


.. math::
    \begin{aligned}
    \sup _{h(t)} \quad & E\left[\frac{1}{\gamma}(V(T))^{\gamma}\right] \\[0.8em]
    \text { subject to: } \quad & V(0)=v_{0}, \quad X(0)=x_{0} \\[0.5em]
    d X(t)=& k(\theta-X(t)) d t+\eta d W(t) \\
    d V(t)=& V(t)((h(t)(k(\theta-X(t))+\frac{1}{2} \eta^{2}\\
    &+\rho \sigma \eta)+r) d t+\eta d W(t))
    \end{aligned}

Finally, the optimal weights are given by,

.. math::
    h^{*}(t, x)=\frac{1}{1-\gamma}\left[\beta(t)+2 x \alpha(t)-\frac{k(x-\theta)}{\eta^{2}}+
    \frac{\rho \sigma}{\eta}+\frac{1}{2}\right]



How to use this submodule
#########################


Step 1: Model fitting
*********************


Implementation
==============

.. automodule:: arbitragelab.stochastic_control_approach.ou_model_mudchanatongsuk

.. autoclass:: StochasticControlMudchanatongsuk
   :members: __init__

.. automethod:: StochasticControlMudchanatongsuk.fit


Step 2: Getting the Optimal Portfolio Weights
*********************************************


Implementation
==============

.. automethod:: StochasticControlMudchanatongsuk.optimal_portfolio_weights


Example
#######


Research Notebook
#################



References
##########



