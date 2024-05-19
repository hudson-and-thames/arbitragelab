.. _cointegration_approach-multivariate_cointegration:

.. note::
    The following documentation closely follows the paper:

    - `Trading in the presence of cointegration. The Journal of Alternative Investments, 15(1):85–97. <http://www.ntuzov.com/Nik_Site/Niks_files/Research/papers/stat_arb/Galenko_2007.pdf>`__ by Galenko, A., Popova, E., and Popova, I. (2012).

====================================
Multivariate Cointegration Framework
====================================

Introduction
############

The cointegration relations between time series imply that the time series are bound together. Over time the time series
might drift apart for a short period of time, but they ought to re-converge. This could serve as the basis of a
profitable pairs trading strategy, as shown in the :ref:`Minimum Profit Optimization <cointegration_approach-minimum_profit>`
module. The current module extends the Minimum Profit Optimization framework to three or more cointegrated assets.
The corresponding trading strategy was illustrated with an empirical application to trading four European stock market
indices at a daily frequency.

Multivariate Cointegration
##########################

Cointegration is defined by the stochastic relationships among the asset log returns in the multivariate cointegration
framework.

Let :math:`P_i`, where :math:`i = 1, 2, \ldots, N` denote the price of :math:`N` assets. The continuously compounded asset
returns, i.e. log-returns at time :math:`t > 0` can be written as:

.. math::

    r_t^i = \ln{P_t^i} - \ln{P_{t-1}^i}

Now construct a process :math:`Y_t` as a linear combination of the :math:`N` asset prices:

.. math::

    Y_t = \sum_{i=1}^N b^i \ln{P_t^i}

where :math:`b^i` denotes the :math:`i`-th element for a finite vector :math:`\mathbf{b}`.
The corresponding asset returns series :math:`Z_t` can be defined as:

.. math::

    Z_t = Y_t - Y_{t-1} = \sum_{i=1}^N b^i r_t^i

Assume that the memory of the process :math:`Y_t` does not extend into the infinite past, which can be expressed as the
following expression in terms of the autocovariance of the process :math:`Y_t`:

.. math::

    \lim_{p \to \infty} \text{Cov} \lbrack Y_t, Y_{t-p} \rbrack = 0

Then the **log-price** process :math:`Y_t` is stationary, if and only if the following three conditions on
**log-returns** process :math:`Z_t` are satisfied:

.. math::
    :nowrap:

    \begin{gather*}
    E[Z_t] = 0 \\
    \text{Var }Z_t = -2 \sum_{p=1}^{\infty} \text{Cov} \lbrack Z_t, Z_{t-p} \rbrack \\
    \sum_{p=1}^{\infty} p \text{ Cov} \lbrack Z_t, Z_{t-p} \rbrack < \infty
    \end{gather*}

When :math:`Y_t` is stationary, the log-price series of the assets are cointegrated.

For equity markets, the log-returns time series can be assumed as stationary and thus satisfy the above conditions.
Therefore, when it comes to empirical applications, the Johansen test could be directly applied to the log price series
to derive the vector :math:`\mathbf{b}`.

Strategy Idea
#############

The core idea of the strategy is to bet on the spread formed by the cointegrated :math:`N` assets that have gone apart
but are expected to mean revert in the future. The trading strategy, using the notations in the previous section, can be
presented as:

    **For each time period, trade** :math:`-b^i C \sum_{p=1}^{\infty} Z_{t-p}` **value of asset** :math:`i, \: i=1, \ldots, N`

where :math:`C` is a positive scale factor. The profit of this strategy can be calculated:

.. math::

    \pi_t = \sum_{i=1}^N -b^i C \bigg[ \sum_{p=1}^{\infty} Z_{t-p} \bigg] r_t^i = -C \sum_{p=1}^{\infty} Z_{t-p} Z_t

The expectation of the profit is thus:

.. math::
    :nowrap:

    \begin{align*}
    E[\pi_t] & = E \bigg[ -C \sum_{p=1}^{\infty} Z_{t-p} Z_t \bigg] \\
             & = -C \sum_{p=1}^{\infty} (Z_{t-p} - E[Z_t])(Z_t - E[Z_t]) \\
             & = -C \sum_{p=1}^{\infty} \text{Cov} [Z_t, Z_{t-p}] \\
             & = 0.5 \: C \text{ Var} Z_t > 0
    \end{align*}

In the above derivation, the two conditions introduced in the previous section were applied:

1) :math:`E[Z_t] = 0`, and

2) :math:`\text{Var }Z_t = -2 \sum_{p=1}^{\infty} \text{Cov} \lbrack Z_t, Z_{t-p} \rbrack`.

By definition, both :math:`C` and the variance :math:`\text{Var } Z_t` are positive values, which means the
expected profit of this strategy is positive. However, the portfolio resulting from the strategy is not dollar neutral.

To construct a dollar neutral portfolio, the assets need to be partitioned based on the sign of the cointegration
coefficient of each asset, :math:`b^i`, into two disjoint sets, :math:`L` and :math:`S`.

.. math::

    i \in L \iff b^i \geq 0

    i \in S \iff b^i < 0

Then the notional of each asset to be traded can be calculated:

.. math::

    \frac{-b^i C \text{ sgn} \bigg( \sum_{p=1}^{\infty} Z_{t-p} \bigg)}{\sum_{j \in L} b^j}, \: i \in L

    \frac{b^i C \text{ sgn} \bigg( \sum_{p=1}^{\infty} Z_{t-p} \bigg)}{\sum_{j \in L} b^j}, \: i \in S

where :math:`\text{sgn(x)}` is the sign function that returns the sign of :math:`x`.

.. note::

    * The resulting portfolio will have :math:`C` dollars (or other currencies) invested in long positions and :math:`C` dollars (or other currencies) in short positions, and thus is dollar-neutral.
    * The expected profit of the strategy is defined by the log-returns, so altering the notional value of the positions will not change the returns.
    * The strategy will **NOT** always long the assets in the set :math:`L` (or always short the assets in the set :math:`S`).

In a real implementation, the price history of the assets is finite, which indicates that the true value of
:math:`\sum_{p=1}^\infty Z_{t-p}` cannot be obtained. The assumptions of the multivariate cointegration framework
suggest that returns of further history do not have predictability about the current returns
(:math:`\lim_{p \to \infty} \text{Cov} \lbrack Y_t, Y_{t-p} \rbrack = 0`). Therefore, a lag parameter :math:`P` will be
introduced and the infinite summation will be replaced by a finite sum :math:`\sum_{p=1}^P Z_{t-p}`.

Trading the Strategy
####################

The ``MultivariateCointegration`` class can be used to generate the cointegration vector, so that later the trading
signals (number of shares to long/short per each asset) can be generated using the Multivariate Cointegration
Trading Rule described in the :ref:`Spread Trading <trading-multi_coint>` section of the documentation.

The strategy is trading at daily frequency and always in the market.

Implementation
**************

.. automodule:: arbitragelab.cointegration_approach.multi_coint

    .. autoclass:: MultivariateCointegration
        :members:
        :inherited-members:

        .. automethod:: __init__

Example
*******

.. code-block::

    # Importing packages
    import pandas as pd
    from arbitragelab.cointegration_approach.multi_coint import MultivariateCointegration

    # Read price series data, set date as index
    data = pd.read_csv('X_FILE_PATH.csv', parse_dates=['Date'])
    data.set_index('Date', inplace=True)

    # Initialize the optimizer
    optimizer = MultivariateCointegration()

    # Set the training dataset
    optimizer = optimizer.set_train_dataset(data)

    # Fill NaN values
    optimizer.fillna_inplace(nan_method='ffill')

    # Generating the cointegration vector to later use in a trading strategy
    coint_vec = optimizer.get_coint_vec()

Research Notebooks
##################

The following research notebook can be used to better understand the Multivariate Cointegration Strategy described above.

* `Multivariate Cointegration Strategy`_

.. _`Multivariate Cointegration Strategy`: https://github.com/hudson-and-thames/arbitrage_research/blob/master/Cointegration%20Approach/multivariate_cointegration.ipynb

.. raw:: html

    <a href="https://hudsonthames.org/notebooks_zip/arblab/multivariate_cointegration.zip"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
    <a href="https://hudsonthames.org/notebooks_zip/arblab/Sample-Data.zip"><button style="margin: 20px; margin-top: 0px">Download Sample Data</button></a>

References
##########

* `Galenko, A., Popova, E. and Popova, I., 2012. Trading in the presence of cointegration. The Journal of Alternative Investments, 15(1), pp.85-97. <http://www.ntuzov.com/Nik_Site/Niks_files/Research/papers/stat_arb/Galenko_2007.pdf>`_
