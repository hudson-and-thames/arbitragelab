.. _hedge_ratios-hedge_ratios:

========================
Hedge Ratio Calculations
========================

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                margin-bottom: 5%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://www.youtube.com/embed/odh5rH3WYJM"
                frameborder="0"
                allowfullscreen
                style="position: absolute;
                       top: 0;
                       left: 0;
                       width: 100%;
                       height: 100%;">
        </iframe>
        <br/>
    </div>

There are various ways to find mean-reverting spread, including distance, cointegration, copula and ML approach.
In the next step, when a trade signal is generated (Z-score > or < then threshold value), a researcher needs to understand
how to trade the spread. How many units of X stock to buy and how many units of Y to sell.

Let's consider the example: a filter system detected a potentially profitable mean-reverting pair of **AMZN** and
**AMD** stock. However, AMZN price is 3200$/stock, and AMD is only 78$. If we trade 1 unit of AMZN vs 1 unit of AMD and both
prices revert back - we will face a negative P&L. Why so?

1% change in AMZN results in 32$ change in your position value, however 1% in AMD results in only 0.78$ P&L change.
This problem is solved by calculating the **hedge ratio**, which will balance dollar value differences between the spread
legs. One way to solve this problem is to divide AMZN price by AMD price and use it as a hedge ratio.

In this case, for each AMZN stock, we trade 3200/78 = 41 units of AMD stock. This approach is called the **ratio method**.
In ArbitrageLab, we have implemented several methods which are used in hedge ratio calculations.

Ordinary Least Squares (OLS)
############################

One way to find a hedge ratio is to fit a linear regression and use a slope coefficient. In our example, we fit the following regression:

:math:`AMZN = \beta * AMD`

In several studies, you may also find that intercept term is included in the linear regression:

:math:`AMZN = \alpha + \beta * AMD`

One critic `Armstrong (2001) <http://doi.org/10.1007/978-0-306-47630-3>`__ points
at the OLS hedge ratios sensitivity to the ordering of variables. It is a possibility that one of
the relationships will be cointegrated, while the other will not. This is troublesome because we would
expect that if the variables are truly cointegrated, the two equations will yield the same conclusion.

Implementation
**************

.. py:currentmodule:: arbitragelab.hedge_ratios.linear
.. autofunction:: get_ols_hedge_ratio

Total Least Squares (TLS)
#########################

A better solution is proposed and implemented, based on `Gregory et al. (2011) <http://dx.doi.org/10.2139/ssrn.1663703>`__
to use orthogonal regression – also referred to as Total Least Squares (TLS) – in which the residuals
of both dependent and independent variables are taken into account. That way, we incorporate the volatility
of both legs of the spread when estimating the relationship so that hedge ratios are consistent, and thus
the cointegration estimates will be unaffected by the ordering of variables.

Implementation
**************

.. py:currentmodule:: arbitragelab.hedge_ratios.linear
.. autofunction:: get_tls_hedge_ratio

Minimum Half-Life
#################

Half-life of mean reversion is one of the key parameters which describe how often the spread will return to its mean value.
As a result, instead of using a linear approach (OLS, TLS) a researcher may want to minimize the spread's half-life of mean-reversion.
We have implemented the algorithm which finds the hedge ratio by minimizing half-life of mean-reversion.

Implementation
**************

.. py:currentmodule:: arbitragelab.hedge_ratios.half_life
.. autofunction:: get_minimum_hl_hedge_ratio

Examples
########

.. code-block::

    # Importing packages
    import pandas as pd
    import numpy as np
    from arbitragelab.hedge_ratios import (get_ols_hedge_ratio, get_tls_hedge_ratio,
                                           get_minimum_hl_hedge_ratio)

    # Getting the dataframe with time series of asset prices
    data = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])
    data = data[['SPY', 'QQQ']] # Filter out to 2 pairs

    ols_model, _, _, _ = get_ols_hedge_ratio(data, dependent_variable='SPY', add_constant=False)
    print(f'OLS hedge ratio for SPY/QQQ spred is {ols_model.coef_[0]})

    tls_model, _, _, _ = get_tls_hedge_ratio(data, dependent_variable='SPY')
    print(f'TLS hedge ratio for SPY/QQQ spred is {tls_model.beta[0]})

    half_life_fit, _, _, _ = get_tls_hedge_ratio(data, dependent_variable='SPY')
    print(f'Minimum HL hedge ratio for SPY/QQQ spred is {half_life_fit.x[0]})

Presentation Slides
###################

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQgWsguF9SOa59ugYCpu3nNMTv4LoP-xbqw7hqlr0WhdkVeiisnLPaewm0Jf6qE5cTxFrJdIsQ4RILK/embed?start=false&loop=false&delayms=3000"
                frameborder="0"
                allowfullscreen
                style="position: absolute;
                       top: 0;
                       left: 0;
                       width: 100%;
                       height: 100%;">
        </iframe>
    </div>

|

References
##########

* `Armstrong, J.S. ed., 2001. Principles of forecasting: a handbook for researchers and practitioners (Vol. 30). Springer Science & Business Media. <http://doi.org/10.1007/978-0-306-47630-3>`_

* `Gregory, I., Ewald, C.O. and Knox, P., 2010, November. Analytical pairs trading under different assumptions on the spread and ratio dynamics. In 23rd Australasian Finance and Banking Conference. <http://doi.org/10.1007/978-0-306-47630-3>`_
