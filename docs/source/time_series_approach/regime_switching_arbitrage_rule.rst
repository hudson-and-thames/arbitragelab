.. _time_series_approach-regime_switching:

.. note::
   The following implementations and documentation closely follow the publication by Bock, M. and Mestel, R:
   `A regime-switching relative value arbitrage rule. Operations Research Proceedings 2008, pages 9–14. Springer.
   <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.453.3576&rep=rep1&type=pdf>`_.

===============================
Regime-Switching Arbitrage Rule
===============================

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                margin-bottom: 5%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://www.youtube.com/embed/lE0FuOSDvVI"
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

The traditional pairs trading strategy usually fails when fundamental or economic reasons cause a
structural break on one of the stocks in the pair. This break will cause the temporary spread
deviations formed by the pair to become persistent spread deviations which will not revert.
Under these circumstances, betting on the spread to revert to its historical mean would imply a loss.

To overcome the problem of detecting whether the deviations are temporary or longer-lasting,
the paper by Bock and Mestel bridges the literature on Markov regime-switching and the
scientific work on statistical arbitrage to develop useful trading rules for pairs trading.

Assumptions
###########

Series Formed by the Trading Pair
*********************************

It models the series :math:`X_t` formed by the trading pair as, 

.. math::
    X_t = \mu_{s_t} + \epsilon_t,

where 

:math:`E[\epsilon_t] = 0`, :math:`\sigma^2_{\epsilon_t} = \sigma^2_{s_t}` and :math:`s_t` denotes the current regime.

Markov Regime-Switching Model
*****************************

A two-state, first-order Markov-switching process for :math:`s_t` is considered with the following transition probabilities:

.. math::
    \Bigg\{ \begin{matrix}
    prob[s_t = 1 | s_{t-1} = 1] = p \\
    prob[s_t = 2 | s_{t-1} = 2] = q \\
    \end{matrix} 

where

:math:`1` indicates a regime with a higher mean (:math:`\mu_{1}`) while
:math:`2` indicates a regime with a lower mean (:math:`\mu_{2}`).

Strategy
********

The trading signal :math:`z_t` is determined in the following way:

:math:`Case\ 1 \ \ current\ regime = 1`

.. math::
    z_t = \left\{\begin{array}{l}
    -1,\ if\ X_t \geq \mu_1 + \delta \cdot \sigma_1 \\ 
    +1,\ if\ X_t \leq \mu_1 - \delta \cdot \sigma_1 \wedge P(s_t = 1 | X_t) \geq \rho \\
    0,\ otherwise
    \end{array}\right.

:math:`Case\ 2 \ \ current\ regime = 2`

.. math::
    z_t = \left\{\begin{array}{l}
    -1,\ if\ X_t \geq \mu_2 + \delta \cdot \sigma_2 \wedge P(s_t = 2 | X_t) \geq \rho\\ 
    +1,\ if\ X_t \leq \mu_2 - \delta \cdot \sigma_2 \\
    0,\ otherwise
    \end{array}\right.

where

:math:`P(\cdot)` denotes the smoothed probabilities for each state, 

:math:`\delta` and :math:`\rho` denote the standard deviation sensitivity parameter and the probability threshold
of the trading strategy, respectively.

To be more specific, the trading signal can be described as,

:math:`Case\ 1 \ \ current\ regime = 1`

.. math::
  \left\{\begin{array}{l}
  Open\ a\ long\ trade,\ if\ X_t \leq \mu_1 - \delta \cdot \sigma_1 \wedge P(s_t = 1 | X_t) \geq \rho \\
  Close\ a\ long\ trade,\ if\ X_t \geq \mu_1 + \delta \cdot \sigma_1 \\ 
  Open\ a\ short\ trade,\ if\ X_t \geq \mu_1 + \delta \cdot \sigma_1 \\
  Close\ a\ short\ trade,\ if\ X_t \leq \mu_1 - \delta \cdot \sigma_1 \wedge P(s_t = 1 | X_t) \geq \rho \\
  Do\ nothing,\ otherwise
  \end{array}\right.

:math:`Case\ 2 \ \ current\ regime = 2`

.. math::
  \left\{\begin{array}{l}
  Open\ a\ long\ trade,\ if\ X_t \leq \mu_2 - \delta \cdot \sigma_2 \\
  Close\ a\ long\ trade,\ if\ X_t \geq \mu_2 + \delta \cdot \sigma_2 \wedge P(s_t = 2 | X_t) \geq \rho\\ 
  Open\ a\ short\ trade,\ if\ X_t \geq \mu_2 + \delta \cdot \sigma_2 \wedge P(s_t = 2 | X_t) \geq \rho\\ 
  Close\ a\ short\ trade,\ if\ X_t \leq \mu_2 - \delta \cdot \sigma_2 \\
  Do\ nothing,\ otherwise
  \end{array}\right.

Steps to Execute the Strategy
*****************************

Step 1: Select a Trading Pair
-----------------------------

In this paper, they used the DJ STOXX 600 component as the asset pool and applied the cointegration test
for the pairs selection. One can use the same method as the paper did or other pairs selection algorithms
like the distance approach for finding trading pairs.

Step 2: Construct the Spread Series
-----------------------------------

In this paper, they used :math:`\frac{P^A_t}{P^B_t}` as the spread series. One can use the same method
as the paper did or other formulae like :math:`(P^A_t/P^A_0) - \beta \cdot (P^B_t/P^B_0)` and
:math:`ln(P^A_t/P^A_0) - \beta \cdot ln(P^B_t/P^B_0)` for constructing the spread series.

Step 3: Estimate the Parameters of the Markov Regime-Switching Model
--------------------------------------------------------------------

Fit the Markov regime-switching model to the spread series with a rolling time window to estimate 
:math:`\mu_1`, :math:`\mu_2`, :math:`\sigma_1`, :math:`\sigma_2` and the current regime.

Step 4: Determine the Signal of the Strategy
--------------------------------------------

Determine the current signal based on the strategy and estimated parameters.

Step 5: Decide the Trade
------------------------

Decide the trade based on the signal at time :math:`t` and the position at :math:`t - 1`.
Possible combinations are listed below:

.. list-table::
   :header-rows: 1

   * - :math:`Position_{t - 1}`
     - :math:`Open\ a\ long\ trade`
     - :math:`Close\ a\ long\ trade`
     - :math:`Open\ a\ short\ trade`
     - :math:`Close\ a\ short\ trade`
     - :math:`Trade\ Action`
     - :math:`Position_{t}`
   * - 0
     - True
     - False
     - False
     - X
     - Open a long trade
     - +1 
   * - 0
     - False
     - X
     - True
     - False
     - Open a short trade
     - -1
   * - 0
     - Otherwise
     - 
     - 
     - 
     - Do nothing
     - 0 
   * - +1
     - False
     - True
     - False
     - X
     - Close a long trade
     - 0
   * - +1
     - False
     - X
     - True
     - False
     - Close a long trade and open a short trade
     - -1 
   * - +1
     - Otherwise
     - 
     - 
     - 
     - Do nothing
     - +1 
   * - -1
     - False
     - X
     - False
     - True
     - Close a short trade
     - 0 
   * - -1
     - True
     - False
     - False
     - X
     - Close a short trade and open a long trade
     - +1
   * - -1
     - Otherwise
     - 
     - 
     - 
     - Do nothing
     - -1 

where X denotes the don't-care term, the value of X could be either True or False.

Implementation
##############

.. py:currentmodule:: arbitragelab.time_series_approach.regime_switching_arbitrage_rule

.. autoclass:: RegimeSwitchingArbitrageRule
    :members: __init__

.. automethod:: RegimeSwitchingArbitrageRule.get_signal

.. automethod:: RegimeSwitchingArbitrageRule.get_signals

.. automethod:: RegimeSwitchingArbitrageRule.get_trades

.. automethod:: RegimeSwitchingArbitrageRule.plot_trades

.. automethod:: RegimeSwitchingArbitrageRule.change_strategy

.. tip::

    If the user is not satisfied with the default trading strategy described in the paper, one can use
    the :code:`change_strategy` method to modify it.

Examples
########

Code Example
************

.. doctest::

  >>> import matplotlib.pyplot as plt
  >>> import yfinance as yf
  >>> from arbitragelab.time_series_approach.regime_switching_arbitrage_rule import (
  ...     RegimeSwitchingArbitrageRule,
  ... )
  >>> data = yf.download("CL=F NG=F", start="2015-01-01", end="2020-01-01", progress=False)[
  ...     "Adj Close"
  ... ]
  >>> # Construct spread series
  >>> ratt = data["NG=F"] / data["CL=F"]
  >>> rsar = RegimeSwitchingArbitrageRule(delta=1.5, rho=0.6)
  >>> window_size = 60
  >>> # Get the current signal
  >>> signal = rsar.get_signal(
  ...     ratt[-window_size:], switching_variance=False, silence_warnings=True
  ... )
  >>> # [Open long, close long, open short, close short]
  >>> list(signal)  # doctest: +NORMALIZE_WHITESPACE
  [True, False, False, True]
  >>> signals = rsar.get_signals(
  ...     ratt, window_size, switching_variance=True, silence_warnings=True
  ... )
  >>> signals  # doctest: +ELLIPSIS
  array(...)
  >>> signals.shape
  (1256, 4)
  >>> # Decide on trades based on the signals
  >>> trades = rsar.get_trades(signals)
  >>> trades  # doctest: +ELLIPSIS
  array(...)
  >>> trades.shape
  (1256, 4)
  >>> # Plot trades
  >>> rsar.plot_trades(ratt, trades)  # doctest: +ELLIPSIS
  <Figure...>
  >>> # Changing rules
  >>> cl_rule = lambda Xt, mu, delta, sigma: Xt >= mu
  >>> cs_rule = lambda Xt, mu, delta, sigma: Xt <= mu
  >>> rsar.change_strategy("High", "Long", "Open", cl_rule)
  >>> rsar.change_strategy("High", "Short", "Close", cs_rule)
  >>> # Get signals on a rolling basis
  >>> signals = rsar.get_signals(
  ...     ratt, window_size, switching_variance=True, silence_warnings=True
  ... )
  >>> signals  # doctest: +ELLIPSIS
  array(...)
  >>> signals.shape
  (1256, 4)
  >>> # Deciding the trades based on the signals
  >>> trades = rsar.get_trades(signals)
  >>> trades  # doctest: +ELLIPSIS
  array(...)
  >>> trades.shape
  (1256, 4)
  >>> # Plotting trades
  >>> rsar.plot_trades(ratt, trades)  # doctest: +ELLIPSIS
  <Figure...>

Research Notebook
#################

The following research notebook can be used to better understand the strategy described above.

* `Statistical Arbitrage Strategy Based on the Markov Regime-Switching Model`_

.. _`Statistical Arbitrage Strategy Based on the Markov Regime-Switching Model`: https://github.com/hudson-and-thames/arbitrage_research/blob/master/Time%20Series%20Approach/regime_switching_arbitrage_rule.ipynb

.. raw:: html

    <a href="https://hudsonthames.org/notebooks_zip/arblab/regime_switching_arbitrage_rule.zip"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
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

    <button class="special" onclick="window.open('https://hudsonthames.org/pairs-trading-with-markov-regime-switching-model/','_blank')">
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

        <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQjGQvSa3Kcx96MOL5zbv3z7ZzlobzZjLHOVevYlO0207w0rPZJ-jkzrgXcbSS5gunzWOpHAsm1x9PH/embed?start=false&loop=false&delayms=3000"
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

1. `Bock, M. and Mestel, R., A regime-switching relative value arbitrage rule. Operations Research Proceedings 2008, pages 9–14. Springer <https://www.researchgate.net/publication/263339291_Pairs_trading_based_on_statistical_variability_of_the_spread_process>`_.
