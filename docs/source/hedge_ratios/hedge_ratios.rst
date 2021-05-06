.. _hedge_ratios-hedge_ratios:

========================
Hedge Ratio Calculations
========================

There are various ways to find mean-reverting spread including distance, cointegration, copula and ML approach.
On the next step, when a trade signal is generated (Z-score > or < then threshold value) a researcher needs to understand
how to trade the spread. How many units of X stock to buy and how many units of Y to sell.

Let's consider the example: a filter system detected a potentially profitable mean-reverting pair of **AMZN** and
**AMD** stock. However, AMZN price is 3200$/stock and AMD is only 78$. If we trade 1 unit of AMZN vs 1 unit of AMD and both
prices revert back - we will face a negative P&L. Why so?

1% change in AMZN results in 32$ change in your position value, however 1% in AMD results in only 0.78$ P&L change.
This problem is solved by calculating the **hedge ratio** which will balance dollar value differences between the spread
legs. One way to solve this problem is to divide AMZN price by AMD price and use it as a hedge ratio.

In this case, for each AMZN stock we trade 3200/78 = 41 units of AMD stock. This approach is called **ratio method**.
In ArbitrageLab we have implemented several methods which are used in hedge ratio calculations.

Ordinary Least Squares (OLS)
############################

One way to find a hedge ratio is to fit a linear regression and use slope coefficient. In our example, we fit the following regression:

:math:`AMZN = \beta * AMD`

In several studies, you may also find that intercept term is included in the linear regression:

:math:`AMZN = \alpha + \beta * AMD`

One critic `Armstrong (2001) <http://doi.org/10.1007/978-0-306-47630-3>`__ points
at the OLS hedge ratios sensitivity to the ordering of variables. It is a possibility that one of
the relationships will be cointegrated, while the other will not. This is troublesome because we would
expect that if the variables are truly cointegrated the two equations will yield the same conclusion.

.. py:currentmodule:: arbitragelab.hedge_ratios.linear
.. autofunction:: get_ols_hedge_ratio

Total Least Squares (TLS)
#########################

A better solution is proposed and implemented, based on `Gregory et al. (2011) <http://dx.doi.org/10.2139/ssrn.1663703>`__
to use orthogonal regression – also referred to as Total Least Squares (TLS) – in which the residuals
of both dependent and independent variables are taken into account. That way, we incorporate the volatility
of both legs of the spread when estimating the relationship so that hedge ratios are consistent, and thus
the cointegration estimates will be unaffected by the ordering of variables.

.. py:currentmodule:: arbitragelab.hedge_ratios.linear
.. autofunction:: get_tls_hedge_ratio

Minimum Half-Life
#################

Half-life of mean reversion is one of the key parameters which describe how often the spread will return to its mean-value.
As a result, instead of using linear approach (OLS, TLS) a researcher may want to minimize spread's half-life of mean-reversion.
We have implemented the algorithm which find the hedge ratio by minimizing half-life of mean-reversion.

.. py:currentmodule:: arbitragelab.hedge_ratios.half_life
.. autofunction:: get_minimum_hl_hedge_ratio


