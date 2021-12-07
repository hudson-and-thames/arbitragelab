.. _other_approaches-pca_approach:

.. note::
   The following documentation closely follows a paper by Marco Avellaneda and Jeong-Hyun Lee:
   `Statistical Arbitrage in the U.S. Equities Market <https://math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb20090616.pdf>`__.

============
PCA Approach
============

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                margin-bottom: 5%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://www.youtube.com/embed/IVAmm34eKWQ"
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

Introduction
############

This module shows how the Principal Component Analysis can be used to create mean-reverting portfolios
and generate trading signals. It's done by considering residuals or idiosyncratic components
of returns and modeling them as mean-reverting processes.

The original paper presents the following description:

The returns for different stocks are denoted as :math:`\{ R_{i} \}^{N}_{i=1}`. The :math:`F` represents
the return of a "market portfolio" over the same period. For each stock in the universe:

.. math::

    R_{i} = \beta_{i} F + \widetilde{R_{i}}

which is a regression, decomposing stock returns into a systematic component :math:`\beta_{i} F` and
an (uncorrelated) idiosyncratic component :math:`\widetilde{R_{i}}`.

This can also be extended to a multi-factor model with :math:`m` systematic factors:

.. math::

    R_{i} = \sum^{m}_{j=1} \beta_{ij} F_{j} + \widetilde{R_{i}}

A trading portfolio is a market-neutral one if the amounts :math:`\{ Q_{i} \}^{N}_{i=1}` invested in
each of the stocks are such that:

.. math::

    \bar{\beta}_{j} = \sum^{N}_{i=1} \beta_{ij} Q_{i} = 0, j = 1, 2,, ..., m.

where :math:`\bar{\beta}_{j}` correspond to the portfolio betas - projections of the
portfolio returns on different factors.

As derived in the original paper,

.. math::

    \sum^{N}_{i=1} Q_{i} R_{i} = \sum^{N}_{i=1} Q_{i} \widetilde{R_{i}}

So, a market-neutral portfolio is only affected by idiosyncratic returns.

PCA Approach
############

This approach was originally proposed by Jolliffe (2002). It is using a historical share price data
on a cross-section of :math:`N` stocks going back :math:`M` days in history. The stocks return data
on a date :math:`t_{0}` going back :math:`M + 1` days can be represented as a matrix:

.. math::

    R_{ik} = \frac{S_{i(t_{0} - (k - 1) \Delta t)} - S_{i(t_{0} - k \Delta t)}}{S_{i(t_{0} - k \Delta t)}}; k = 1, ..., M; i = 1, ..., N.

where :math:`S_{it}` is the price of stock :math:`i` at time :math:`t` adjusted for dividends. For
daily observations :math:`\Delta t = 1 / 252`.

Returns are standardized, as some assets may have greater volatility than others:

.. math::

    Y_{ik} = \frac{R_{ik} - \bar{R_{i}}}{\bar{\sigma_{i}}}

where

.. math::

   \bar{R_{i}} = \frac{1}{M} \sum^{M}_{k=1}R_{ik}

and

.. math::

   \bar{\sigma_{i}}^{2} = \frac{1}{M-1} \sum^{M}_{k=1} (R_{ik} - \bar{R_{i}})^{2}

And the empirical correlation matrix is defined by

.. math::

   \rho_{ij} = \frac{1}{M-1} \sum^{M}_{k=1} Y_{ik} Y_{jk}

.. Note::

    It's important to standardize data before inputting it to PCA, as the PCA seeks to maximize the
    variance of each component. Using unstandardized input data will result in worse results.
    The *get_signals()* function in this module automatically standardizes input returns before
    feeding them to PCA.

The original paper mentions that picking long estimation windows for the correlation matrix
(:math:`M \gg N`, :math:`M` is the estimation window, :math:`N` is the number of assets in a portfolio)
don't make sense because they take into account the distant past which is economically irrelevant.
The estimation windows used by the authors is fixed at 1 year (252 trading days) prior to the trading date.

The eigenvalues of the correlation matrix are ranked in the decreasing order:

.. math::

   N \ge \lambda_{1} \ge \lambda_{2} \ge \lambda_{3} \ge ... \ge \lambda_{N} \ge 0.

And the corresponding eigenvectors:

.. math::

   v^{(j)} = ( v^{(j)}_{1}, ..., v^{(j)}_{N} ); j = 1, ..., N.

Now, for each index :math:`j` we consider a corresponding "eigen portfolio", in which we
invest the respective amounts invested in each of the stocks as:

.. math::

    Q^{(j)}_{i} = \frac{v^{(j)}_{i}}{\bar{\sigma_{i}}}

And the eigen portfolio returns are:

.. math::

   F_{jk} = \sum^{N}_{i=1} \frac{v^{(j)}_{i}}{\bar{\sigma_{i}}} R_{ik}; j = 1, 2, ..., m.

.. figure:: images/pca_approach_portfolio.png
    :scale: 80 %
    :align: center

    Performance of a portfolio composed using the PCA approach in comparison to the market cap portfolio.
    An example from `Statistical Arbitrage in the U.S. Equities Market <https://math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb20090616.pdf>`__.
    by Marco Avellaneda and Jeong-Hyun Lee.

In a multi-factor model we assume that stock returns satisfy the system of stochastic
differential equations:

.. math::

   \frac{dS_{i}(t)}{S_{i}(t)} = \alpha_{i} dt + \sum^{N}_{j=1} \beta_{ij} \frac{dI_{j}(t)}{I_{j}(t)} + dX_{i}(t),

where :math:`\beta_{ij}` are the factor loadings.

The idiosyncratic component of the return with drift :math:`\alpha_{i}` is:

.. math::

   d \widetilde{X_{i}}(t) = \alpha_{i} dt + d X_{i} (t).

Based on the previous descriptions, a model for :math:`X_{i}(t)` is estimated as the Ornstein-Uhlenbeck
process:

.. math::

   dX_{i}(t) = \kappa_{i} (m_{i} - X_{i}(t))dt + \sigma_{i} dW_{i}(t), \kappa_{i} > 0.

which is stationary and auto-regressive with lag 1.

.. Note::

   To find out more about the Ornstein-Uhlenbeck model and optimal trading under this model
   please check out our section on Trading Under the Ornstein-Uhlenbeck Model.

The parameters :math:`\alpha_{i}, \kappa_{i}, m_{i}, \sigma_{i}` are specific for each stock.
They are assumed to *de facto* vary slowly in relation to Brownian motion increments :math:`dW_{i}(t)`,
in the chosen time-window. The authors of the paper were using a 60-day window to estimate the residual
processes for each stock and assumed that these parameters were constant over the window.

However, the hypothesis of parameters being constant over the time-window is being accepted
for stocks which mean reversion (the estimate of :math:`\kappa`) is sufficiently high and is
rejected for stocks with a slow speed of mean-reversion.

An investment in a market long-short portfolio is being constructed by going long $1 on the stock and
short :math:`\beta_{ij}` dollars on the :math:`j` -th factor. Expected 1-day return of such portfolio
is:

.. math::

   \alpha_{i} dt + \kappa_{i} (m_{i} - X_{i}(t))dt

The parameter :math:`\kappa_{i}` is called the speed of mean-reversion. If :math:`\kappa \gg 1` the
stock reverts quickly to its means and the effect of drift is negligible. As we are assuming that
the parameters of our model are constant, we are interested in stocks with fast mean-reversion,
such that:

.. math::

   \frac{1}{\kappa_{i}} \ll T_{1}

where :math:`T_{1}` is the estimation window to estimate residuals in years.

Implementation
**************
.. py:currentmodule:: arbitragelab.other_approaches.pca_approach

.. autoclass:: PCAStrategy
    :noindex:
    :members: __init__, standardize_data, get_factorweights, get_sscores

PCA Trading Strategy
####################

The strategy implemented sets a default estimation window for the correlation matrix as 252 days, a window for residuals
estimation of 60 days (:math:`T_{1} = 60/252`) and the threshold for the mean reversion speed of an eigen portfolio for
it to be traded so that the reversion time is less than :math:`1/2` period (:math:`\kappa > 252/30 = 8.4`).

For the process :math:`X_{i}(t)` the equilibrium variance is defined as:

.. math::

   \sigma_{eq,i} = \frac{\sigma_{i}}{\sqrt{2 \kappa_{i}}}

And the following variable is defined:

.. math::

   s_{i} = \frac{X_{i}(t)-m_{i}}{\sigma_{eq,i}}

This variable is called the S-score. The S-score measures the distance to the equilibrium of the
cointegrated residual in units standard deviations, i.e. how far away a given asset eigen portfolio
is from the theoretical equilibrium value associated with the model.

.. figure:: images/pca_approach_s_score.png
    :scale: 80 %
    :align: center

    Evolution of the S-score of JPM ( vs. XLF ) from January 2006 to December 2007.
    An example from `Statistical Arbitrage in the U.S. Equities Market <https://math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb20090616.pdf>`__.
    by Marco Avellaneda and Jeong-Hyun Lee.

If the eigen portfolio shows a mean reversion speed above the set threshold (:math:`\kappa`), the
S-score based on the values from the residual estimation window is being calculated.

The trading signals are generated from the S-scores using the following rules:

- Open a long position if :math:`s_{i} < - \bar{s_{bo}}`

- Close a long position if :math:`s_{i} < + \bar{s_{bc}}`

- Open a short position if :math:`s_{i} > + \bar{s_{so}}`

- Close a short position if :math:`s_{i} > - \bar{s_{sc}}`

Opening a long position means buying $1 of the corresponding stock (of the asset eigen portfolio)
and selling :math:`\beta_{i1}` dollars of assets from the first scaled eigenvector (:math:`Q^{(1)}_{i}`),
:math:`\beta_{i2}` from the second scaled eigenvector (:math:`Q^{(2)}_{i}`) and so on.

Opening a short position, on the other hand, means selling $1 of the corresponding stock and buying
respective beta values of stocks from scaled eigenvectors.

Authors of the paper, based on empirical analysis chose the following cutoffs. They were selected
based on simulating strategies from 2000 to 2004 in the case of ETF factors:

- :math:`\bar{s_{bo}} = \bar{s_{so}} = 1.25`

- :math:`\bar{s_{bc}} = 0.75`, :math:`\bar{s_{sc}} = 0.50`

The rationale behind this strategy is that we open trades when the eigen portfolio shows good mean
reversion speed and its S-score is far from the equilibrium, as we think that we detected an anomalous
excursion of the co-integration residual. We expect most of the assets in our portfolio to be near
equilibrium most of the time, so we are closing trades at values close to zero.

The signal generating function implemented in the ArbitrageLab package outputs target weights for each
asset in our portfolio for each observation time - target weights here are the sum of weights of all
eigen portfolios that show high mean reversion speed and have needed S-score value at a given time.

Implementation
**************

.. autoclass:: PCAStrategy
    :noindex:
    :members: __init__, get_sscores, get_signals

Examples
********

.. code-block::

   # Importing packages
   import pandas as pd
   import numpy as np
   from arbitragelab.other_approaches.pca_approach import PCAStrategy

   # Getting the dataframe with time series of asset returns
   data = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])

   # The PCA Strategy class that contains all needed methods
   pca_strategy = PCAStrategy()

   # Simply applying the PCAStrategy with standard parameters
   target_weights = pca_strategy.get_signals(data, k=8.4, corr_window=252,
                                             residual_window=60, sbo=1.25,
                                             sso=1.25, ssc=0.5, sbc=0.75,
                                             size=1)

   # Or we can do individual actions from the PCA approach
   # Standardizing the dataset
   data_standardized, data_std = pca_strategy.standardize_data(data)

   # Getting factor weights using the first 252 observations
   data_252days = data[:252]
   factorweights = pca_strategy.get_factorweights(data_252days)

   # Calculating factor returns for a 60-day window from our factor weights
   data_60days = data[(252-60):252]
   factorret = pd.DataFrame(np.dot(data_60days, factorweights.transpose()),
                            index=data_60days.index)

   # Calculating residuals for a set 60-day window
   residual, coefficient = pca_strategy.get_residuals(data_60days, factorret)

   # Calculating S-scores for each eigen portfolio for a set 60-day window
   s_scores = pca_strategy.get_sscores(residual, k=8)

Research Notebooks
##################

The following research notebook can be used to better understand the PCA approach described above.

* `PCA Approach`_

.. _`PCA Approach`: https://hudsonthames.org/notebooks/arblab/pca_approach.html

.. raw:: html

    <a href="https://hudthames.tech/3iKM250"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
    <a href="https://hudthames.tech/2S03R58"><button style="margin: 20px; margin-top: 0px">Download Sample Data</button></a>

Presentation Slides
###################

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vSkekx_tYH8hmHhMRPczS-IksfJEB5lzET81_qrZqzzVOgRxWBEazKSvcSj3AEeehZxt8Nwu7USNM4K/embed?start=false&loop=false&delayms=3000"
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

* `Avellaneda, M. and Lee, J.H., 2010. Statistical arbitrage in the US equities market. Quantitative Finance, 10(7), pp.761-782. <https://math.cims.nyu.edu/faculty/avellane/AvellanedaLeeStatArb071108.pdf>`__
* `Jolliffe, I. T., Principal Components Analysis, Springer Series in Statistics, Springer-Verlag, Heidelberg, 2002. <https://www.springer.com/gp/book/9780387954424>`__
