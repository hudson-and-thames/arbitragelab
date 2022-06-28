.. _cointegration_approach-minimum_profit:

.. note::
    The following documentation closely follows two papers

    1. `Loss protection in pairs trading through minimum profit bounds: a cointegration approach <http://downloads.hindawi.com/archive/2006/073803.pdf>`__ by Lin, Y.-X., McCrae, M., and Gulati, C. (2006)
    2. `Finding the optimal pre-set boundaries for pairs trading strategy based on cointegration technique <https://ro.uow.edu.au/cgi/viewcontent.cgi?article=1040&context=cssmwp>`__ by Puspaningrum, H., Lin, Y.-X., and Gulati, C. M. (2010)

===========================
Minimum Profit Optimization
===========================

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                margin-bottom: 5%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://www.youtube.com/embed/1zz91G0nR14?start=928"
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

Introduction
############

A common pairs trading strategy is to "fade the spread", i.e. to open a trade when the spread is sufficiently far away
from its equilibrium in anticipation of the spread reverting to the mean. Within the context of cointegration, the
spread refers to cointegration error, and in the remainder of this documentation "spread" and "cointegration error" will
be used interchangeably.

In order to define a strategy, the concept of "sufficiently far away from the equilibrium of the spread", i.e. a pre-set
boundary chosen to open a trade, needs to be clearly defined. The boundary can affect the minimum total profit (MTP)
over a specific trading horizon. The higher the pre-set boundary for opening trades, the higher the profit per trade
but the lower the trade numbers. The opposite applies to lowering the boundary values. The number of trades over a
specified trading horizon is determined jointly by the average trade duration and the average inter-trade interval.

This module is designed to find the optimal pre-set boundary that would maximize the MTP for cointegration error
following an AR(1) process by numerically estimating the average trade duration, average inter-trade interval, and the
average number of trades based on the mean first-passage time.

In this strategy, the following assumptions are made:

- The price of two assets (:math:`S_1` and :math:`S_2`) are cointegrated over the relevant time period, which includes both in-sample and out-of-sample (trading) period.
- The cointegration error follows a stationary AR(1) process.
- The cointegration error is symmetrically distributed so that the optimal boundary could be applied on both sides of the mean.
- Short sales are permitted or possible through a broker and there is no interest charged for the short sales and no cost for trading.
- The cointegration coefficient :math:`\beta > 0`, where a cointegration relationship is defined as:

.. math::

    P_{S_1,t} - \beta P_{S_2,t} = \varepsilon_t

In the following sections, as originally shown in the paper, the derivation of the minimum profit per trade and the mean
first-passage time of a stationary AR(1) process is presented.

Minimum Profit per Trade
########################

Denote a trade opened when the cointegration error :math:`\varepsilon_t` overshoots the pre-set upper boundary :math:`U`
as a **U-trade**, and similarly a trade opened when :math:`\varepsilon_t` falls through the pre-set lower
boundary :math:`L` as an **L-trade**. Without loss of generality, it can be assumed that the mean
of :math:`\varepsilon_t` equals 0. Then the minimum profit per U-trade can be derived from the following trade setup.

- When :math:`\varepsilon_t \geq U` at :math:`t_o`, open a trade by selling :math:`N` of asset :math:`S_1` and buying :math:`\beta N` of asset :math:`S_2`.
- When :math:`\varepsilon_t \leq 0` at :math:`t_c`, close the trade.

The profit per trade would thus be:

.. math::

    P = N (P_{S_1, t_o} - P_{S_1, t_c}) + \beta N (P_{S_2, t_c} - P_{S_2, t_o})

Since the two assets are cointegrated during the trade period, the cointegration relationship can be substituted into
the above equation and derive the following:

.. math::
    :nowrap:

    \begin{align*}
    P & =  N (P_{S_1, t_o} - P_{S_1, t_c}) + \beta N (P_{S_2, t_c} - P_{S_2, t_o}) \\
      & =  N (\beta P_{S_2, t_c} - P_{S_1, t_c}) + N (P_{S_1, t_o} - \beta P_{S_2, t_o}) \\
      & =  -N \varepsilon_{t_c} + N \varepsilon_{t_o} \\
      & \geq N U
    \end{align*}

Thus, by trading the asset pair with the weight as a proportion of the cointegration coefficient, the profit per U-trade
is at least :math:`U` dollars when trading one unit of the pair. Should the required minimum profit be higher, then the
strategy can trade multiple units of the pair weighted by the cointegration coefficient.

According to the assumptions in the Introduction section, the lower boundary will be set at :math:`-U` due to the
symmetric distribution of the cointegration error. The profit of an L-trade can thus be derived from the following trade
setup.

- When :math:`\varepsilon_t \leq -U` at :math:`t_o`, open a trade by buying :math:`N` of asset :math:`S_1` and selling :math:`\beta N` of asset :math:`S_2`.
- When :math:`\varepsilon_t \geq 0` at :math:`t_c`, close the trade.

Using the same derivation above, it can be shown that the profit per L-trade is also at least :math:`U` dollars per unit.
Therefore, the boundary is exactly the minimum profit per trade, where the strategy only trade one unit of the
cointegrated pair weighted by the cointegration coefficient.

.. figure:: images/AME-DOV.png
    :width: 100 %
    :align: center

    An example of pair trading Ametek Inc. (AME) and Dover Corp. (DOV) from January 2nd, 2019 to date. The green line defines the boundary for U-trades and the red line defines the boundary for L-trades. They equally deviate from the cointegration error mean (the black line).

Mean First-passage Time of an AR(1) Process
###########################################

Consider a stationary AR(1) process:

.. math::

    Y_t = \phi Y_{t-1} + \xi_t

where :math:`-1 < \phi < 1`, and :math:`\xi_t \sim N(0, \sigma_{\xi}^2) \quad \mathrm{i.i.d}`. The mean first-passage
time over interval :math:`\lbrack a, b \rbrack` of :math:`Y_t`, starting at initial state
:math:`y_0 \in \lbrack a, b \rbrack`, which is denoted by :math:`E(\mathcal{T}_{a,b}(y_0))`, is given by

.. math::

    E(\mathcal{T}_{a,b}(y_0)) = \frac{1}{\sqrt{2 \pi}\sigma_{\xi}}\int_a^b E(\mathcal{T}_{a,b}(u)) \> \mathrm{exp} \Big( - \frac{(u-\phi y_0)^2}{2 \sigma_{\xi}^2} \Big) du + 1

This integral equation can be solved numerically using the Nyström method, i.e. by solving the following linear
equations:

.. math::

    \begin{pmatrix}
    1 - K(u_0, u_0) & -K(u_0, u_1) & \ldots & -K(u_0, u_n) \\
    -K(u_1, u_0) & 1 - K(u_1, u_1) & \ldots & -K(u_1, u_n) \\
    \vdots & \vdots & \vdots & \vdots \\
    -K(u_n, u_0) & -K(u_n, u_1) & \ldots & 1-K(u_n, u_n)
    \end{pmatrix}
    \begin{pmatrix}
    E_n(\mathcal{T}_{a,b}(u_0)) \\
    E_n(\mathcal{T}_{a,b}(u_1)) \\
    \vdots \\
    E_n(\mathcal{T}_{a,b}(u_n)) \\
    \end{pmatrix}
    =
    \begin{pmatrix}
    1 \\
    1 \\
    \vdots \\
    1 \\
    \end{pmatrix}

where :math:`E_n(\mathcal{T}_{a,b}(u_0))` is a discretized estimate of the integral, and the Gaussian kernel function
:math:`K(u_i, u_j)` is defined as:

.. math::

    K(u_i, u_j) = \frac{h}{2 \sqrt{2 \pi} \sigma_{\xi}} w_j  \> \mathrm{exp} \Big( - \frac{(u_j - \phi u_i)^2}{2 \sigma_{\xi}^2} \Big)

and the weight :math:`w_j` is defined by the trapezoid integration rule:

.. math::

    w_j = \begin{cases}
    1 & j = 0 \quad \mathrm{and} \quad j = n \\
    2 & 0 < j < n, j \in \mathbb{N}
    \end{cases}

The time complexity for solving the above linear equation system is :math:`O(n^3)` (see `here <https://www.netlib.org/lapack/lug/node71.html>`__
for an introduction of the time complexity of :code:`numpy.linalg.solve`), which is the most time-consuming part of this
procedure.

Minimum Total Profit (MTP)
##########################

The MTP of U-trades within a specific trading horizon :math:`\lbrack 0, T \rbrack` is defined by:

.. math::

    MTP(U) = \Big( \frac{T}{TD_U + I_U} - 1 \Big) U

where :math:`TD_U` is the trade duration and :math:`I_U` is the inter-trade interval.

From the definition, the MTP is simultaneously determined by :math:`TD_U` and :math:`I_U`, both of which can be derived
from the mean first-passage time. Also, it is already known that :math:`U` is the minimum profit per U-trade,
so :math:`\frac{T}{TD_U + I_U} - 1` can be used to estimate the number of U-trades. Following the assumption that the
de-meaned cointegration error follows an AR(1) process:

.. math::

    \varepsilon_t = \phi \varepsilon{t-1} + a_t \qquad a_t \sim N(0, \sigma_a^2) \> \mathrm{i.i.d}

Since the core idea of the approach is to "fade the spread" at :math:`U`, the trade duration can be defined
as the average time of the cointegration error to pass 0 for the first time given that its initial value
is :math:`U`. Thus using the definition of the mean first-passage time of the cointegration error process:

.. math::

    TD_U = E(\mathcal{T}_{0, \infty}(U)) = \lim_{b \to \infty} \frac{1}{\sqrt{2 \pi} \sigma_a} \int_0^b E(\mathcal{T}_{0, b}(s)) \> \mathrm{exp} \Big( - \frac{(s- \phi U)^2}{2 \sigma_a^2} \Big) ds + 1

The inter-trade interval is defined as the average time of the de-meaned cointegration error to pass :math:`U` the first
time given its initial value is 0.

.. math::

    I_U = E(\mathcal{T}_{- \infty, U}(0)) = \lim_{-b \to - \infty} \frac{1}{\sqrt{2 \pi} \sigma_a} \int_{-b}^U E(\mathcal{T}_{-b, U}(s)) \> \mathrm{exp} \Big( - \frac{s^2}{2 \sigma_a^2} \Big) ds + 1

Under the assumption that the cointegration error follows a stationary AR(1) process, the standard deviation of the
fitted residual :math:`\sigma_a` and the standard deviation of the cointegration error :math:`\sigma_{\varepsilon}` has
the following relationship:

.. math::

    \sigma_a = \sqrt{1 - \phi^2} \sigma_{\varepsilon}

The following stylized fact helped approximate the infinity limit for both integrals: for a stationary AR(1) process
:math:`\{ \varepsilon_t \}`, the probability that the absolute value of the process :math:`\vert \varepsilon_t \vert` is
greater than 5 times the standard deviation of the process :math:`5 \sigma_{\varepsilon}` is close to 0. Therefore,
:math:`5 \sigma_{\varepsilon}` will be used as an approximation of the infinity limit in the integrals.

Optimize the Pre-Set Boundaries that Maximizes MTP
##################################################

Based on the above definitions, the numerical algorithm to optimize the pre-set boundaries that maximize MTP could be
given as follows.

1. Perform Engle-Granger or Johansen test (see :ref:`here <cointegration_approach-cointegration_tests>`) to derive the cointegration coefficient :math:`\beta`.
2. Fit the cointegration error :math:`\varepsilon_t` to an AR(1) process and retrieve the AR(1) coefficient and the fitted residual.
3. Calculate the standard deviation of cointegration error (:math:`\sigma_{\varepsilon}`) and the fitted residual (:math:`\sigma_a`).
4. Generate a sequence of pre-set upper bounds :math:`U_i`, where :math:`U_i = i \times 0.01, \> i = 0, \ldots, b/0.01`, and :math:`b = 5 \sigma_{\varepsilon}`.
5. For each :math:`U_i`,

   a. Calculate :math:`{TD}_{U_i}`.
   b. Calculate :math:`I_{U_i}`. *Note: this is the main bottleneck of the optimization speed.*
   c. Calculate :math:`MTP(U_i)`.

6. Find :math:`U^{*}` such that :math:`MTP(U^{*})` is the maximum.
7. Set a desired minimum profit :math:`K \geq U^{*}` and calculate the number of assets to trade according to the following equations:

.. math::

    N_{S_2} = \Big \lceil \frac{K \beta}{U^{*}} \Big \rceil

    N_{S_1} = \Big \lceil \frac{N_{S_2}}{\beta} \Big \rceil

Trading the Strategy
####################

After applying the above-described optimization rules, the output is optimal levels to enter and exit trades
as well as number of shares to trade per leg of the cointegration pair. These outputs can be used in the Minimum
Profit Trading Rule described in the :ref:`Spread Trading <trading-minimum_profit>` section of the documentation.

Implementation
**************

.. automodule:: arbitragelab.cointegration_approach.minimum_profit

    .. autoclass:: MinimumProfit
        :members:
        :inherited-members:

        .. automethod:: __init__

Example
*******

.. code-block::

    # Importing packages
    import pandas as pd
    from arbitragelab.cointegration_approach.minimum_profit import MinimumProfit

    # Read price series data, set date as index
    data = pd.read_csv('X_FILE_PATH.csv', parse_dates=['Date'])
    data.set_index('Date', inplace=True)

    # Initialize the optimizer
    optimizer = MinimumProfit()

    # Set the training dataset
    optimizer = optimizer.set_train_dataset(data)

    # Run an Engle-Granger test to retrieve cointegration coefficient
    beta_eg, epsilon_t_eg, ar_coeff_eg, ar_resid_eg = optimizer.fit(use_johansen=False)

    # Optimize the pre-set boundaries, retrieve optimal upper bound, optimal minimum total profit,
    # and number of trades.
    optimal_ub, _, _, optimal_mtp, optimal_num_of_trades = optimizer.optimize(ar_coeff_eg,
                                                                              epsilon_t_eg,
                                                                              ar_resid_eg,
                                                                              len(train_df))

    # Generate optimal trading levels and number of shares to trade
    num_of_shares, optimal_levels = optimizer.get_optimal_levels(optimal_ub,
                                                                 minimum_profit,
                                                                 beta_eg,
                                                                 epsilon_t_eg)

Research Notebooks
##################

* `Minimum Profit Optimization`_

.. _`Minimum Profit Optimization`: https://hudsonthames.org/notebooks/arblab/minimum_profit_optimization.html

.. raw:: html

    <a href="https://hudsonthames.org/notebooks_zip/arblab/minimum_profit_optimization.zip"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
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

    <button class="special" onclick="window.open('https://hudsonthames.org/minimum-profit-optimization/','_blank')">
      <span>Read our article on the topic</span>
    </button>

|

Presentation Slides
###################

.. image:: images/minimum_profit_slides.png
   :scale: 40 %
   :align: center
   :target: https://drive.google.com/file/d/1ZeJD81OrKln8QDxm1sU63ivRgqXCcQbS/view

References
##########

* `Lin, Y.-X., McCrae, M., and Gulati, C., 2006. Loss protection in pairs trading through minimum profit bounds: a cointegration approach <http://downloads.hindawi.com/archive/2006/073803.pdf>`_
* `Puspaningrum, H., Lin, Y.-X., and Gulati, C. M. 2010. Finding the optimal pre-set boundaries for pairs trading strategy based on cointegration technique <https://ro.uow.edu.au/cgi/viewcontent.cgi?article=1040&context=cssmwp>`_
