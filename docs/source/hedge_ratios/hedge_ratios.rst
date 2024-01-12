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

|

There are various ways to find mean-reverting spread, including distance,
cointegration, copula, and ML approach.  In the next step, when a trade signal
is generated (Z-score above or below some threshold value), a researcher needs
to understand how to trade the spread. In other words, how many units of X stock to buy and how
many units of Y to sell.

Let's consider an example: a filter system detected a potentially profitable mean-reverting pair of **AMZN** and
**AMD** stock. However, AMZN price is $3200, and AMD is only $78. If we trade 1 unit of AMZN vs 1 unit of AMD and both
prices revert back, we will face a negative P&L.

Why?

A 1% change in AMZN results in $32 change in your position value, however 1% in AMD results in only $0.78 P&L change.
This problem is solved by calculating the *hedge ratio*, which will balance dollar value differences between the spread
legs. A simple solution is to divide the AMZN price by AMD price, and use this resulting value as a hedge ratio.

In this case, for each AMZN stock, we trade 3200/78 = 41 units of AMD stock. This approach is called the **ratio method**.
In ArbitrageLab, we have implemented several methods which are used in hedge ratio calculations.

Spread construction methodology
###############################

.. note::
    In the ArbitrageLab package, all hedge ratio calculations methods normalize outputs such that a dependent variable asset has a hedge
    ratio of **1** and a spread is constructed using the following formula:

    .. math::
        S = leg1 - (hedgeratio_2) * leg2 - (hedgeratio_3) * leg3 - .....

.. note::
    All hedge ratio calculation methods assume that the first asset is a dependent variable unless a user specifies which asset
    should be used as dependent.

One can use the `construct_spread` function from the ArbitrageLab hedge ratio module to construct spread series from generated
hedge ratios.

.. py:currentmodule:: arbitragelab.hedge_ratios.spread_construction
.. autofunction:: construct_spread


.. doctest::

    >>> import pandas as pd
    >>> import numpy as np
    >>> from arbitragelab.hedge_ratios import construct_spread
    >>> url = "https://raw.githubusercontent.com/hudson-and-thames/example-data/main/arbitrage_lab_data/sp100_prices.csv"
    >>> data = pd.read_csv(url, index_col=0, parse_dates=[0])
    >>> hedge_ratios = pd.Series({"A": 1, "AVB": 0.832406370860649})
    >>> spread = construct_spread(data[["AVB", "A"]], hedge_ratios=hedge_ratios)
    >>> inverted_spread = construct_spread(
    ...     data[["AVB", "A"]], hedge_ratios=hedge_ratios, dependent_variable="A"
    ... )
    >>> inverted_spread # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Date
    2017-01-03 -100.529...

Ordinary Least Squares (OLS)
############################

One way to find a hedge ratio is to fit a linear regression and use a slope coefficient. In our example, we fit the following regression:

.. math::
    AMZN = \beta * AMD

In several studies, you may also find that intercept term is included in the linear regression:

.. math::
    AMZN = \alpha + \beta * AMD

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

Johansen Test Eigenvector
#########################

One of the big advantages of the Johansen cointegration test is the resulting eigenvector which serves as a hedge ratio to construct a spread.
A researcher can either use the `JohansenPortfolio` class from the ArbitrageLab cointegration module to find all eigenvectors or use a function
from hedge ratios module to get the first-order eigenvector which yields the most stationary portfolio.

Implementation
**************

.. py:currentmodule:: arbitragelab.hedge_ratios.johansen
.. autofunction:: get_johansen_hedge_ratio


Box-Tiao Canonical Decomposition (BTCD)
#######################################

First, Box and Tiao introduced a canonical transformation of an :math:`N`-dimensional stationary autoregressive process.
The components of the transformed process can then be ordered from least to most predictable, according to the research by Box and Tiao.

The estimation of this method goes as follows. For the :math:`VAR(L)` equation, which is called a forecasting equation,
this method fits :math:`\beta` and estimates :math:`\hat{P_t}` from the beta. With the estimated :math:`P_t`,
it undergoes a decomposition process and solves for optimal weight.
In short, the objective is to come up with the matrix of coefficients that deliver a vector of forecasts
with the most predictive power over the next observation.

.. math::
    \sum_{l=1}^{L} \sum_{i=1}^{N} \beta_{i, l, n} P_{t-l, i}+\beta_{n, 0} X_{t-1, n}+\varepsilon_{t, n}

Implementation
**************

.. py:currentmodule:: arbitragelab.hedge_ratios.box_tiao
.. autofunction:: get_box_tiao_hedge_ratio


Minimum Half-Life
#################

Half-life of mean reversion is one of the key parameters which describe how often the spread will return to its mean value.
As a result, instead of using a linear approach (OLS, TLS) a researcher may want to minimize the spread's half-life of mean-reversion.
We have implemented the algorithm which finds the hedge ratio by minimizing half-life of mean-reversion.

Implementation
**************

.. py:currentmodule:: arbitragelab.hedge_ratios.half_life
.. autofunction:: get_minimum_hl_hedge_ratio

Minimum ADF Test T-statistic Value
##################################

In the same fashion as the minimum half-life is calculated, one can find a hedge ratio that minimizes the t-statistic of the
Augmented Dickey-Fuller Test (ADF).

.. note::
    As Minimum Half-Life and Minimum ADF T-statistics algorithms rely on numerical optimization, sometimes output results can be
    unstable due to the fact the optimization algorithm did not converge. In order to control this issue, `get_minimum_hl_hedge_ratio` and
    `get_adf_optimal_hedge_ratio` return scipy optimization object, which contains logs and a status flag if the method managed to
    converge.

Implementation
**************

.. py:currentmodule:: arbitragelab.hedge_ratios.adf_optimal
.. autofunction:: get_adf_optimal_hedge_ratio


Examples
########

Code Example
************

.. doctest::

    >>> import pandas as pd
    >>> import numpy as np
    >>> from arbitragelab.hedge_ratios import (
    ...     get_ols_hedge_ratio,
    ...     get_tls_hedge_ratio,
    ...     get_johansen_hedge_ratio,
    ...     get_box_tiao_hedge_ratio,
    ...     get_minimum_hl_hedge_ratio,
    ...     get_adf_optimal_hedge_ratio,
    ... )
    >>> # Fetch time series of asset prices
    >>> url = "https://raw.githubusercontent.com/hudson-and-thames/example-data/main/arbitrage_lab_data/gld_gdx_data.csv"
    >>> data = pd.read_csv(url, index_col=0, parse_dates=[0])
    >>> ols_hedge_ratio, _, _, _ = get_ols_hedge_ratio(
    ...     data, dependent_variable="GLD", add_constant=False
    ... )
    >>> print(f"OLS hedge ratio for GLD/GDX spread is {ols_hedge_ratio}")  # doctest: +ELLIPSIS
    OLS hedge ratio for GLD/GDX spread is {'GLD': 1.0, 'GDX': 7.6...}
    >>> tls_hedge_ratio, _, _, _ = get_tls_hedge_ratio(data, dependent_variable="GLD")
    >>> print(f"TLS hedge ratio for GLD/GDX spread is {tls_hedge_ratio}")  # doctest: +ELLIPSIS
    TLS hedge ratio for GLD/GDX spread is {...}
    >>> joh_hedge_ratio, _, _, _ = get_johansen_hedge_ratio(data, dependent_variable="GLD")
    >>> print(
    ...     f"Johansen hedge ratio for GLD/GDX spread is {joh_hedge_ratio}"
    ... )  # doctest: +ELLIPSIS
    Johansen hedge ratio for GLD/GDX spread is {...}
    >>> box_tiao_hedge_ratio, _, _, _ = get_box_tiao_hedge_ratio(data, dependent_variable="GLD")
    >>> print(
    ...     f"Box-Tiao hedge ratio for GLD/GDX spread is {box_tiao_hedge_ratio}"
    ... )  # doctest: +ELLIPSIS
    Box-Tiao hedge ratio for GLD/GDX spread is {...}
    >>> hl_hedge_ratio, _, _, _, opt_object = get_minimum_hl_hedge_ratio(
    ...     data, dependent_variable="GLD"
    ... )
    >>> print(
    ...     f"Minimum HL hedge ratio for GLD/GDX spread is {hl_hedge_ratio}"
    ... )  # doctest: +ELLIPSIS
    Minimum HL hedge ratio for GLD/GDX spread is {...}
    >>> print(opt_object.status)
    0
    >>> adf_hedge_ratio, _, _, _, opt_object = get_adf_optimal_hedge_ratio(
    ...     data, dependent_variable="GLD"
    ... )
    >>> print(
    ...     f"Minimum ADF t-statistic hedge ratio for GLD/GDX spread is {adf_hedge_ratio}"
    ... )  # doctest: +ELLIPSIS
    Minimum ADF t-statistic hedge ratio for GLD/GDX spread is {...}
    >>> print(opt_object.status)
    0


Research Notebooks
******************

The following research notebook can be used to better understand the hedge ratios described above.

* `Hedge Ratios`_

.. _`Hedge Ratios`: https://hudsonthames.org/notebooks/arblab/hedge_ratios.html

.. raw:: html

    <a href="https://hudsonthames.org/notebooks_zip/arblab/hedge_ratios.zip"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
    <a href="https://hudsonthames.org/notebooks_zip/arblab/Sample-Data.zip"><button style="margin: 20px; margin-top: 0px">Download Sample Data</button></a>

Research Article
################

.. raw:: html

    <style>
      .special {
        display: inline-block;
        background-color: #0399AB;
        color: #eeeeee;
        text-align: center;
        font-size: 180%;
        padding: 15px;
        width: 100%;
        transition: all 0.5s;
        cursor: pointer;
        font-family: 'Josefin Sans';
      }
      .special span {
        cursor: pointer;
        display: inline-block;
        position: relative;
        transition: 0.5s;
      }
      .special span:after {
        content: '\00bb';
        position: absolute;
        opacity: 0;
        top: 0;
        right: -20px;
        transition: 0.5s;
      }
      .special:hover {
        background-color: #e7f2fa;
        color: #000000;
      }
      .special:hover span {
        padding-right: 25px;
      }
      .special:hover span:after {
        opacity: 1;
        right: 0;
      }
    </style>

    <button class="special" onclick="window.open('https://hudsonthames.org/introduction-to-hedge-ratio-estimation-methods/','_blank')">
      <span>Read our article on the topic</span>
    </button>

|

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
