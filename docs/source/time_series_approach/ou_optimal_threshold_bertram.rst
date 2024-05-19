.. _time_series_approach-ou_model:

.. note::
   The following implementations and documentation closely follow the publication by William K. Bertram:
   `Analytic solutions for optimal statistical arbitrage trading. Physica A: Statistical Mechanics and its Applications, 389(11): 2234–2243
   <http://www.stagirit.org/sites/default/files/articles/a_0340_ssrn-id1505073.pdf>`_.

===========================================
OU Model Optimal Trading Thresholds Bertram
===========================================

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                margin-bottom: 5%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://www.youtube.com/embed/C7iZLMXyIOQ"
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

For statistical arbitrage strategies, determining the trading thresholds is an essential issue,
and one of the solutions for this is to maximize performance per unit of time. To do so, the
investor should choose the proper entry and exit thresholds. If the thresholds are narrow, then
the time it needs to complete a trade is short, but the profit is small. In contrast, if thresholds
are wide, the profit in each trade is big, but the time it needs to complete a trade is long.
The interplay between the profit per trade and the trade length gives rise to an optimization problem.

In this paper, the author derives analytic formulae for statistical arbitrage trading where
the security price follows an exponential Ornstein-Uhlenbeck process. By framing the problem
in terms of the first-passage time of the process, he first derives the expressions for the
mean and the variance of the trade length. Then he derives the formulae for the expected
return and the variance of the return per unit of time. Finally, he resolves the problem
of choosing optimal trading thresholds by maximizing the expected return and the Sharpe ratio.

.. warning::

    Although the paper assumes that the long-term mean of the O-U process is zero, we still extend
    the results so that the O-U process whose mean is not zero can also use this method.
    We use :math:`\theta` for long-term mean, :math:`\mu` for mean-reversion speed and
    :math:`\sigma` for amplitude of randomness of the O-U process, which is different from
    the reference paper.

Assumptions
###########

Price of the Traded Security
****************************

It models the price of the traded security :math:`p_t` as,

.. math::
    {p_t = e^{X_t}};\quad{X_{t_0} = x_0}

where :math:`X_t` satisfies the following stochastic differential equation,

.. math::
    {dX_t = {\mu}({\theta} - X_t)dt + {\sigma}dW_t}

where :math:`\theta` is the long-term mean, :math:`\mu` is the speed at which the values will regroup around
the long-term mean and :math:`\sigma` is the amplitude of randomness of the O-U process.

Trading Strategy
****************

The trading strategy is defined by entering a trade when :math:`X_t = a`, exiting the trade at :math:`X_t = m`, where :math:`a < m`.

Trading Cycle
*************

The trading cycle is completed as :math:`X_t` change from :math:`a` to :math:`m`, then back to :math:`a`,
and the trade length :math:`T` is defined as the time needed to complete a trading cycle.

Analytic Formulae
#################

Mean and Variance of the Trade Length
*************************************

.. math::
    E[T] = \frac{\pi}{\mu} (Erfi(\frac{(m - \theta)\sqrt{\mu}}{\sigma}) - Erfi(\frac{(a - \theta)\sqrt{\mu}}{\sigma})),

where :math:`Erfi(x) = iErf(ix)` is the imaginary error function.

.. math::
    V[T] = \frac{{w_1(\frac{(m - \theta)\sqrt{2\mu}}{\sigma})} - {w_1(\frac{(a - \theta)\sqrt{2\mu}}{\sigma})} - {w_2(\frac{(m - \theta)\sqrt{2\mu}}{\sigma})} + {w_2(\frac{(a - \theta)\sqrt{2\mu}}{\sigma})}}{{\mu}^2},

where 

:math:`w_1(z) = (\frac{1}{2} \sum_{k=1}^{\infty} \Gamma(\frac{k}{2}) (\sqrt{2}z)^k / k! )^2 - (\frac{1}{2} \sum_{n=1}^{\infty} (-1)^k \Gamma(\frac{k}{2}) (\sqrt{2}z)^k / k! )^2,`

:math:`w_2(z) = \sum_{k=1}^{\infty} \Gamma(\frac{2k - 1}{2}) \Psi(\frac{2k - 1}{2}) (\sqrt{2}z)^{2k - 1} / (2k - 1)!,`

where :math:`\Psi(x) = \psi(x) − \psi(1)` and :math:`\psi(x)` is the digamma function.

Mean and Variance of the Return per Unit of Time
************************************************

.. math::
    \mu_s(a,\ m,\ c) = \frac{r(a,\ m,\ c)}{E [T]}

.. math::
    \sigma_s(a,\ m,\ c) = \frac{{r(a,\ m,\ c)}^2{V[T]}}{{E[T]}^3}

where :math:`r(a,\ m,\ c) = (m − a − c)` gives the continuously compound rate of return for a single trade
accounting for transaction cost.

Optimal Strategies
##################

To calculate an optimal trading strategy, we seek to choose optimal entry and exit thresholds that maximise
the expected return or the Sharpe ratio per unit of time for a given transaction cost/risk-free rate.

This paper shows that the maximum expected return/Sharpe ratio occurs when :math:`(m - \theta)^2 = (a - \theta)^2`.
Since we have assumed that :math:`a < m`, this implies that :math:`m = 2\theta − a`. Therefore, for a given
transaction cost/risk-free rate, the following equation can be maximized to find optimal :math:`a` and :math:`m`.

.. math::
    \mu^*_s(a, c) = \frac{r(a, 2\theta − a, c)}{E [T]}

.. math::
    S^*(a, c, r_f) = \frac{\mu_s(a, 2\theta − a, c) - r^*}{\sigma_s(a, 2\theta − a, c)}

where :math:`r^* = \frac{r_f}{E[T]}` and :math:`r_f` is the risk free rate.


Implementation
##############

Initializing OU-Process Parameters
**********************************

One can initialize the O-U process by directly setting its parameters or by fitting the process to the given data.
The fitting method can refer to pp. 12-13 in the following book:
`Tim Leung and Xin Li, Optimal Mean reversion Trading: Mathematical Analysis and Practical Applications
<https://www.amazon.com/Optimal-Mean-Reversion-Trading-Mathematical/dp/9814725919>`_.

.. py:currentmodule:: arbitragelab.time_series_approach.ou_optimal_threshold_bertram

.. autoclass:: OUModelOptimalThresholdBertram
    :members: __init__

.. automethod:: OUModelOptimalThresholdBertram.construct_ou_model_from_given_parameters

.. automethod:: OUModelOptimalThresholdBertram.fit_ou_model_to_data

Getting Optimal Thresholds
**************************

This paper examines the problem of choosing an optimal strategy under two different objective functions:
the expected return; and the Sharpe ratio. One can choose either to get the thresholds.
The following functions will return a tuple contains :math:`a` and :math:`m`, where :math:`a` is the optimal
entry thresholds, and :math:`m` is the optimal exit threshold.

.. note::
    :code:`initial_guess` is used to speed up the process and ensure the target equation can be solved by
    :code:`scipy.optimize`. If the value of :code:`initial_guess` is not given, the default value will be
    :math:`\theta - c - 10^{-2}`. From our experiment, the default value is suited for most of the cases.
    If you observe that the thresholds got by the functions is odd or the running time is larger than 5 second,
    please try a :code:`initial_guess` on different scales.

.. automethod:: OUModelOptimalThresholdBertram.get_threshold_by_maximize_expected_return

.. automethod:: OUModelOptimalThresholdBertram.get_threshold_by_maximize_sharpe_ratio

Calculating Metrics
*******************

One can calculate performance metrics for the trading strategy using the following functions.

.. automethod:: OUModelOptimalThresholdBertram.expected_trade_length

.. automethod:: OUModelOptimalThresholdBertram.trade_length_variance

.. automethod:: OUModelOptimalThresholdBertram.expected_return

.. automethod:: OUModelOptimalThresholdBertram.return_variance

.. automethod:: OUModelOptimalThresholdBertram.sharpe_ratio

Plotting Comparison
*******************

One can use the following functions to observe the impact of transaction costs and risk-free rates on the optimal
thresholds and performance metrics under the optimal thresholds.

.. automethod:: OUModelOptimalThresholdBertram.plot_target_vs_c

.. automethod:: OUModelOptimalThresholdBertram.plot_target_vs_rf

Examples
########

Code Example
************

.. doctest::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from arbitragelab.time_series_approach.ou_optimal_threshold_bertram import (
    ...     OUModelOptimalThresholdBertram,
    ... )
    >>> OUOTB = OUModelOptimalThresholdBertram()
    >>> # Init the OU-process parameter
    >>> OUOTB.construct_ou_model_from_given_parameters(theta=0, mu=180.9670, sigma=0.1538)
    >>> # Get optimal thresholds by maximizing the expected return
    >>> a, m = OUOTB.get_threshold_by_maximize_expected_return(c=0.001)
    >>> # Threshold when we enter a trade
    >>> a  # doctest: +ELLIPSIS
    -0.004...
    >>> # Threshold when we exit the trade
    >>> m  # doctest: +ELLIPSIS
    0.004...
    >>> # Get the expected return and the variance
    >>> expected_return = OUOTB.expected_return(a=a, m=m, c=0.001)
    >>> expected_return  # doctest: +ELLIPSIS
    0.492...
    >>> return_variance = OUOTB.return_variance(a=a, m=m, c=0.001)
    >>> return_variance  # doctest: +ELLIPSIS
    0.0021...
    >>> # Get optimal thresholds by maximizing the Sharpe ratio
    >>> a, m = OUOTB.get_threshold_by_maximize_sharpe_ratio(c=0.001, rf=0.01)
    >>> a  # doctest: +ELLIPSIS
    -0.01125...
    >>> m  # doctest: +ELLIPSIS
    0.01125...
    >>> # Get the Sharpe ratio
    >>> S = OUOTB.sharpe_ratio(a=a, m=m, c=0.001, rf=0.01)
    >>> S  # doctest: +ELLIPSIS
    3.862...
    >>> # Set an array of transaction costs
    >>> c_list = np.linspace(0, 0.01, 30)
    >>> # Plot the impact of transaction costs on the optimal entry threshold
    >>> OUOTB.plot_target_vs_c(
    ...     target="a", method="maximize_expected_return", c_list=c_list
    ... )  # doctest: +ELLIPSIS
    <Figure...>
    >>> # Set an array containing risk-free rates.
    >>> rf_list = np.linspace(0, 0.05, 30)
    >>> # Plot the impact of risk-free rates on the optimal entry threshold
    >>> OUOTB.plot_target_vs_rf(
    ...     target="a", method="maximize_sharpe_ratio", rf_list=rf_list, c=0.001
    ... )  # doctest: +ELLIPSIS
    <Figure...>

Research Notebooks
******************

The following research notebook can be used to better understand the method described above.

* `OU Model Optimal Trading Thresholds Bertram`_

.. _`OU Model Optimal Trading Thresholds Bertram`: https://github.com/hudson-and-thames/arbitrage_research/blob/master/Time%20Series%20Approach/ou_model_optimal_threshold_Bertram.ipynb

.. raw:: html

    <a href="https://hudsonthames.org/notebooks_zip/arblab/ou_model_optimal_threshold_Bertram.zip"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
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

    <button class="special" onclick="window.open('https://hudsonthames.org/optimal-trading-thresholds-for-the-o-u-process/','_blank')">
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

        <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vS1t-Tavwc5oEfPDShVCF3Q9s3aEEjcG4PqD40_izOHSb45J5JXxwwb0eBNgwrtkc-ZaGPd46BvTQKn/embed?start=false&loop=false&delayms=3000"
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

* `Bertram, W. K., Analytic solutions for optimal statistical arbitrage trading. Physica A: Statistical Mechanics and its Applications, 389(11): 2234–2243 <http://www.stagirit.org/sites/default/files/articles/a_0340_ssrn-id1505073.pdf>`_.
