.. _trading-mispricing_index_strategy:

========================================
Mispricing Index Copula Trading Strategy
========================================

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                margin-bottom: 5%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://www.youtube.com/embed/t99uCFQL5KI"
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

.. Note::
    The following strategy closely follows the implementations:

    * `Pairs trading with copulas. (2014) <https://efmaefm.org/0efmameetings/EFMA%20ANNUAL%20MEETINGS/2014-Rome/papers/EFMA2014_0222_FullPaper.pdf>`__ by Xie, W., Liew, R.Q., Wu, Y. and Zou, X.
    * `The profitability of pairs trading strategies: distance, cointegration and copula methods. (2016) <https://www.tandfonline.com/doi/pdf/10.1080/14697688.2016.1164337?casa_token=X0-FdUqDsv4AAAAA:ZFothfEHF-dO2-uDtFo2ZuFH0uzF6qijsweHD888yfPx3OZGXW6Szon1jvA2BB_AsgC5-kGYreA4fw>`__ by Rad, H., Low, R.K.Y. and Faff, R.

.. Note::
    The authors claimed a relatively robust 8-10% returns from this strategy in the formation period (6 mo).
    We are pretty positive that the rules proposed in the paper were implemented correctly in the :code:`MPICopulaTradingRule`
    module with thorough unit testing on every possible case, and thus it is very unlikely that we made logical mistakes.
    However the P&L is very sensitive to the opening and exiting logic and parameter values, input data and copula choice,
    and it cannot lead to the claimed returns, after trying all the possible interpretations of ambiguities.
    
    We found out that using an AND for opening and OR for exiting lead to a much less sensitive strategy and generally leads to a
    much better performance, and we provide such an option in the module.
    
    We still implement this module for people who intend to explore possibilities with copula, however the user should be
    aware of the nature of the proposed framework.
    Interested reader may read through the *Possible Issues* part and see where this strategy can be improved, and is encouraged
    to make changes in the source code, where we grouped the exit and open logic in one function for ease of alteration.
    
Introduction to the Strategy Concepts
#####################################
For convenience, the **mispricing index** implemented in the strategy will be referred to **MPI** when no ambiguity arises.


A Quick Review of the Basic Copula Strategy
*******************************************

Before we introduce the MPI strategy, let's recall the basic copula strategy and understand its pros and cons.

The basic copula strategy proposed by [Liew et al. 2013] works with price series (or log-price series, which 
has identical suggested trading signals under the copula framework) and looks at conditional probabilities.
For example, if :math:`P(X \le x_t | Y = y_t)` is small for stock pair :math:`(X, Y)` with prices :math:`(x_t, y_t)`,
then stock :math:`X` is considered undervalued given the current price of :math:`Y`.
Then we derive long/short positions regarding the spread based on this conditional probability.
This conditional probability is calculated from some copula fitted to the training data, and generally other 
models cannot produce this value.

Although this approach is in general reasonably sound, it has one critical drawback, that
**the price series is not in general stationary**.
For example, if one adopt the assumption that stocks move in a lognormal 
(One may also argue that such assumption can be quite situational. We won't get into the details here.) fashion,
then almost surely the price will reach any given level with enough time.

This implies that, if the basic copula framework was working on two stocks that have an upward(or downward) drift
in the trading period, it may go out of range of the training period, and the conditional probabilities calculated
from which will always be extreme values as :math:`0` and :math:`1`, bringing in nonsense trading signals.
One possible way to overcome this inconvenience is to keep the training set up to date so it is less likely to
have new prices out of range.
Another way is to work with a more likely stationary time series, for example, returns.


How is the MPI Strategy Constructed?
************************************

At first glance, the MPI strategy documented in [Xie et al. 2016] looks quite bizarre.
However, it is reasonably consistent when one goes through the logic of its construction:
In order to use returns to generate trading signals, one needs to be creative about utilizing the information.
It is one thing to know the dependence structure of a pair of stocks, it is another thing to trade based on it
because intrinsically stocks are traded on prices, not returns.

If one regards using conditional probabilities as a distance measure, then it is natural to think about how far
the returns have cumulatively driven the prices apart (or together), thereby introducing trading opportunities.

Hence we introduce the following concepts for the strategy framework:


Mispricing Index
****************

**MPI** is defined as the conditional probability of returns, i.e., 

.. math::
    MI_t^{X\mid Y} = P(R_t^X < r_t^X \mid R_t^Y = r_t^Y)

.. math::
    MI_t^{Y\mid X} = P(R_t^Y < r_t^Y \mid R_t^X = r_t^X)

for stocks :math:`(X, Y)` with returns random variable at day :math:`t`: :math:`(R_t^X, R_t^Y)` and returns value at
day :math:`t`: :math:`(r_t^X, r_t^Y)`.
Those two values determine how mispriced each stock is, based on that day's return.
Note that so far only one day's return information contributes, and we want to add it up to cumulatively use
returns to gauge how mispriced the stocks are.
Therefore we introduce the **flag** series:


Flag and Raw Flag
*****************

A more descriptive name than flag, in my opinion, would be **cumulative mispricing index**.
The **raw flag** series (with a star) is the cumulative sum of daily MPIs minus 0.5, i.e.,

.. math::
    FlagX^*(t) = FlagX^*(t-1) + (MI_t^{X\mid Y} - 0.5), \quad FlagX^*(0) = 0.

.. math::
    FlagY^*(t) = FlagY^*(t-1) + (MI_t^{Y\mid X} - 0.5), \quad FlagY^*(0) = 0.

Or equivalently

.. math::
    FlagX^*(t) = \sum_{s=0}^t (MI_s^{X\mid Y} - 0.5)

.. math::
    FlagY^*(t) = \sum_{s=0}^t (MI_s^{Y\mid X} - 0.5)

If one plots the raw flags series, they look quite similar to cumulative returns from their price series,
which is what they were designed to do:
Accumulate information from daily returns to reflect information on prices.
Therefore, you may consider it as a fancy way to represent the returns series.

However, the **real flag** series (without a star, :math:`FlagX(t)`, :math:`FlagY(t)`) **will be reset to 0**
whenever there is an exiting signal, which brings us to the trading logic.

Trading Logic
#############

Default Opening and Exiting Rules
*********************************

The authors propose a **dollar-neutral** trade scheme worded as follows:

Suppose stock :math:`X`, :math:`Y` are associated with :math:`FlagX`, :math:`FlagY` respectively.

Opening rules: (:math:`D = 0.6` in the paper)

- When :math:`FlagX` reaches :math:`D`,
  short :math:`X` and buy :math:`Y` in **equal amounts**. (:math:`-1` Position)

- When :math:`FlagX` reaches :math:`-D`,
  short :math:`Y` and buy :math:`X` in **equal amounts**. (:math:`1` Position)
  
- When :math:`FlagY` reaches :math:`D`,
  short :math:`Y` and buy :math:`X` in **equal amounts**. (:math:`1` Position)

- When :math:`FlagY` reaches :math:`-D`,
  short :math:`X` and buy :math:`Y` in **equal amounts**. (:math:`-1` Position)

Exiting rules: (:math:`S = 2` in the paper)

- If trades are opened based on :math:`FlagX`, then they are closed if :math:`FlagX` returns to zero or reaches stop-loss
  position :math:`S` or :math:`-S`.

- If trades are opened based on :math:`FlagY`, then they are closed if :math:`FlagY` returns to zero or reaches stop-loss
  position :math:`S` or :math:`-S`.

- After trades are closed, both :math:`FlagX` and :math:`FlagY` are reset to :math:`0`.

The rationale behind the dollar-neutral choice might be that (the authors did not mention this), because the signals are generated by returns, it makes sense to "reset" returns when entering into a long/short position.

Ambiguities
***********

The authors did not specify what will happen if the following occurs:

1. When :math:`FlagX`reaches :math:`D` (or :math:`-D`) and :math:`FlagY` reaches :math:`D` (or :math:`-D`) together.
2. When in a long(or short) position, receives a short(or long) trigger.
3. When receiving an opening and exiting signal together.
4. When the position was open based on :math:`FlagX` (or :math:`FlagY`), :math:`FlagY` (or :math:`FlagX`) reaches
   :math:`S` or :math:`-S`.

Here is our take on the above issues:

1. Do nothing.
2. Change to the trigger position. For example, a long position with a short trigger will go short.
3. Go for the exiting signal.
4. Do nothing.

Choices for Open and Exit Logic
*******************************
The above default logic is essentially an OR-OR logic for open and exit: When at least one of the 4 open conditions is satisfied, an
open signal (long or short) is triggered;
Similarly for the exit logic, to exit only one of them needs to be satisfied.
The opening trigger is in general too sensitive and leads to too many trades, and [Rad et al. 2016] suggested using AND-OR logic instead.
Thus, to achieve more flexibility, we allow the user to choose AND, OR for both open and exit logic and hence there are 4 possible combinations.
Based on our tests we found AND-OR to be the most reasonable choice in general, but in certain situations other choices may have an edge.

The default is OR-OR, as suggested in [Xie et al. 2014], and you can switch to other logic in the :code:`get_positions_and_flags` method
by setting :code:`open_rule` and :code:`exit_rule` to your own liking.
For instance :code:`open_rule='and'`, :code:`exit_rule='or'`.

.. Note::
    There are some nuiances on how the logic is carried.
    In the paper [Xie et al. 2014], they tracked which stock led to opening of a position, and it influences the exit.
    This tracking procedure makes no sense for other 3 trading logics.
    The variable :code:`open_based_on` is present in lower level functions that are (python) private, and they track which stock triggered the last
    opening.
    Thus this variable is not used (although still calculated, but it is likely incorrect) when using other logic.

.. figure:: images/returns_and_samples.png
    :scale: 40 %
    :align: center

    Sampling from the various fitted copulas, and plot the empirical density from training data
    from BKD and ESC.

.. figure:: images/mpi_normalized_prices.png
    :scale: 40 %
    :align: center
    
.. figure:: images/mpi_flags_positions.png
    :scale: 40 %
    :align: center
    
.. figure:: images/mpi_units.png
    :scale: 40 %
    :align: center

    A visualised output of flags, positions and units to hold using a Student-t copula. The stock pair considered 
    is BKD and ESC. 


Implementation
##############

.. automodule:: arbitragelab.trading.copula_strategy_mpi
        
    .. autoclass:: MPICopulaTradingRule
	:members: __init__, to_returns, set_copula, set_cdf, calc_mpi, get_condi_probs, positions_to_units_dollar_neutral, get_positions_and_flags

Example
*******

.. code-block::

   # Importing the module and other libraries
   from arbitragelab.trading.copula_strategy_mpi import MPICopulaTradingRule
   from arbitragelab.copula_approach import construct_ecdf_lin
   from arbitragelab.copula_approach.archimedean import N14
   import matplotlib.pyplot as plt
   import pandas as pd

   # Instantiating the module
   CSMPI = MPICopulaTradingRule(opening_triggers=(-0.6, 0.6), stop_loss_positions=(-2, 2))

   # Loading the data in prices of stock X and stock Y
   prices = pd.read_csv('FILE_PATH' + 'stock_X_Y_prices.csv').set_index('Date').dropna()
   
   # Convert prices to returns
   returns = CSMPI.to_returns(prices)

   # Split data into train and test sets
   training_len = int(len(prices) * 0.7)
   returns_train = returns.iloc[:training_len, :]
   returns_test = returns.iloc[training_len:, :]
   prices_train = prices.iloc[:training_len, :]
   prices_test = prices.iloc[training_len:, :]

   # Adding the N14 copula (it can be fitted with tools from the Copula Approach)
   cop = N14(theta=2)
   CSMPI.set_copula(cop)
													   
   # Constructing cdf for x and y
   cdf_x = construct_ecdf_lin(returns['BKD'])
   cdf_y = construct_ecdf_lin(returns['ESC'])
   CSMPI.set_cdf(cdf_x, cdf_y)

   # Forming positions and flags using trading data, assuming holding no position initially.
   # Default uses OR-OR logic for open-exit.
   positions, flags = CSMPI.get_positions_and_flags(returns=returns_test)

   # Use AND-OR logic.                       
   positions_and_or, flags_and_or = CSMPI.get_positions_and_flags(returns=returns_test,
                                                                  open_rule='and',
                                                                  exit_rule='or')
   
   # Changing the positions series to units to hold for
   # a dollar-neutral strategy for $10000 investment
   units = CSMPI.positions_to_units_dollar_neutral(prices_df=prices_test,
                                                   positions=positions,
                                                   multiplier=10000)


Possible Issues
###############
The following are critiques for the default strategy.
For a thorough comparison in large amounts of stocks across several decades, read For the AND-OR strategy, read more 
in [Rad et al. 2016] on comparisons with other common strategies, using the AND-OR logic.

1. The default strategy's outcome is quite sensitive to the values of opening and exiting triggers to the point that
   a well-fitted copula with a not-so-good set of parameters can actually lose money.

2. The trading signal is generated from the flags series, and the flags series will be calculated from the
   copula that we use to model.
   Therefore the explainability suffers.
   Also, it is based on the model in second order, and therefore the flag series and the suggested positions
   will be quite different across different copulas, making it not stable and not directly comparable mutually.

3. The way the flags series are defined does not handle well when both stocks are underpriced/overpriced concurrently.

4. Because flags will be reset to 0 once there is an exiting signal, it implicitly models the returns as
   martingales that do not depend on the current price level of the stock itself and the other stock.
   Such an assumption may be situational, and the user should be aware. (White noise returns do not imply that
   the prices are well cointegrated.)
   
5. The strategy is betting the flags series having dominating mean-reversion behaviors, for a pair of cointegrated stocks.
   It is not mathematically clear what justifies the rationale.
   
6. If accumulating mispricing index is basically using returns to reflect prices, and the raw flags look
   basically the same as normalized prices, why not just directly use normalized prices instead?

Research Notebooks
##################

The following research notebook can be used to better understand the copula strategy described above.

* `Mispricing Index Copula Strategy`_

.. _`Mispricing Index Copula Strategy`: https://hudsonthames.org/notebooks/arblab/Copula_Strategy_Mispricing_Index.html

.. raw:: html

    <a href="https://hudthames.tech/3xsu5ws"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
    <a href="https://hudthames.tech/2S03R58"><button style="margin: 20px; margin-top: 0px">Download Sample Data</button></a>

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

    <button class="special" onclick="window.open('https://hudsonthames.org/copula-for-pairs-trading-overview-of-common-strategies/','_blank')">
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

        <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQZMGmxe-3Z6gqxm66cKM6S-XVoflILDY1dhAmWo_LOsj-PQw2G9vuUAaI-QQ5nny_2uBNqd1COyJld/embed?start=false&loop=false&delayms=3000"
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

* `Xie, W., Liew, R.Q., Wu, Y. and Zou, X., 2016. Pairs trading with copulas. The Journal of Trading, 11(3), pp.41-52. <https://efmaefm.org/0efmameetings/EFMA%20ANNUAL%20MEETINGS/2014-Rome/papers/EFMA2014_0222_FullPaper.pdf>`__
* `Liew, R.Q. and Wu, Y., 2013. Pairs trading: A copula approach. Journal of Derivatives & Hedge Funds, 19(1), pp.12-30. <https://link.springer.com/article/10.1057/jdhf.2013.1>`__
* `Rad, H., Low, R.K.Y. and Faff, R., 2016. The profitability of pairs trading strategies: distance, cointegration and copula methods. Quantitative Finance, 16(10), pp.1541-1558. <https://www.tandfonline.com/doi/pdf/10.1080/14697688.2016.1164337?casa_token=X0-FdUqDsv4AAAAA:ZFothfEHF-dO2-uDtFo2ZuFH0uzF6qijsweHD888yfPx3OZGXW6Szon1jvA2BB_AsgC5-kGYreA4fw>`__
