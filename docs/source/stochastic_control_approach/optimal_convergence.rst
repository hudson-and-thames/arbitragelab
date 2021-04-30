.. _stochastic_control_approach-optimal_convergence:

.. note::
    The following implementations and documentation closely follow the below work:

    `Liu, J. and Timmermann, A., 2013. Optimal convergence trade strategies. <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.905.236&rep=rep1&type=pdf>`__


===================
Optimal Convergence
===================

Introduction
############

Convergence trades resemble the standard long-short arbitrage strategy
popular in industry and assumed in academic studies. Conventionally, such strategies take positions of equal size but opposite signs
either in portfolio weight or in number of shares. This seems intuitively
reasonable and ensures that future liabilities offset. However, such strategies
will typically not be optimal.

The objective of optimally trading off risk and returns can lead to quite different solutions compared
with the standard arbitrage objective assumed in convergence trades.
Standard arbitrage strategies and/or delta neutral convergence trades are
designed to explore long-term arbitrage opportunities but do typically not
optimally exploit the short-run risk return trade-off. By placing arbitrage
opportunities in the context of a portfolio maximization problem, the
optimal convergence strategy accounts for both arbitrage opportunities and
diversification benefits.

ADD THIS LATER : In some cases the price difference between two assets in a convergence
trade will disappear permanently after it reaches zero. To capture this, we use
a stopped cointegration process whereby asset prices follow a cointegrated
process before the difference reaches zero and the difference remains at zero
afterwards. Stopped cointegrated prices can also be used to model the strategies
of convergence traders who close out their position when prices converge. This case with
nonrecurring arbitrage opportunities gives rise to a set of very different
boundary conditions when solving for the optimal portfolio weights.


Modelling
#########


Implementation
##############

Model fitting
*************

We input the training data to the fit method which calculates the spread
and the estimators of the parameters of the model.

Implementation
==============


.. automodule:: arbitragelab.stochastic_control_approach.optimal_convergence


.. autoclass:: OptimalConvergence
   :members: __init__


.. automethod:: OptimalConvergence.fit

.. tip::
    To view the estimated model parameters from training data, call the ``describe`` function.

    .. automethod:: OptimalConvergence.describe


Optimal Unconstrained Portfolio Weights with recurring arbitrage opportunities
******************************************************************************

In this step we input the evaluation data and specify the utility function parameter :math:`\gamma`.

.. warning::
    Please make sure the value of ``gamma`` is positive.


Implementation
==============

.. automethod:: OptimalConvergence.unconstrained_portfolio_weights_continuous


Delta Neutral Portfolio Weights with recurring arbitrage opportunities
**********************************************************************

In this step we input the evaluation data and specify the utility function parameter :math:`\gamma`.

.. warning::
    Please make sure the value of ``gamma`` is positive.


Implementation
==============

.. automethod:: OptimalConvergence.delta_neutral_portfolio_weights_continuous


Example
#######


Research Notebook
#################

The following research notebook can be used to better understand the approach described above.

* `Optimal Convergence Trade Strategies`_

.. _`Optimal Convergence Trade Strategies`:

References
##########

* `Liu, J. and Timmermann, A., 2013. Optimal convergence trade strategies. The Review of Financial Studies, 26(4), pp.1048-1086. <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.905.236&rep=rep1&type=pdf>`__