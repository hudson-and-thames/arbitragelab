.. _time_series_approach-h_strategy:

.. note::
   The following implementations and documentation closely follow the publication by Bogomolov, T:
   `Pairs trading based on statistical variability of the spread process. Quantitative Finance, 13(9): 1411–1430
   <https://www.researchgate.net/publication/263339291_Pairs_trading_based_on_statistical_variability_of_the_spread_process>`_.

==========
H-Strategy
==========

In this paper, the author proposes a new nonparametric approach to pairs trading based on the idea of
Renko and Kagi charts. This approach exploits statistical information about the variability of the
tradable process. The approach does not aim to find a long-run mean of the process and trade towards it
like other methods of pairs trading. Instead, it manages the problem of how far the process should move
in one direction before trading in the opposite direction becomes potentially profitable by measure the
variability of the process.

H-construction
##############

Suppose :math:`P(t)` is a continuous time series on the time interval :math:`[0, T]`.

Renko construction
******************

Step 1: Generate the Renko Process
----------------------------------

The Renko process :math:`X(i)` is defined as,

.. math::
    X(i) : X(i) = P(\tau_i), i = 0, 1, ..., N,

where :math:`\tau_i`, :math:`i = 0, 1, ..., N` is an increasing sequence of time moments such that  

for some arbitrary :math:`H > 0`, :math:`\tau_0 = 0` and :math:`P(\tau_0) = P(0)`,

.. math::
    H \leq \max \limits_{t \in [0,T]} P(t) - \min \limits_{t \in [0,T]} P(t),

.. math::
    \tau_i = inf\{u \in [\tau_{i - 1}, T] : |P(u) − P(\tau_{i - 1})| = H\}.

Step 2: Determine Turning Points
--------------------------------

We create another sequence of time moments :math:`\{(\tau^a_n, \tau^b_n), n = 0, 1, ..., M\}` based on the
sequence :math:`{\tau_i}`. The sequence :math:`\{\tau^a_n\}` defines time moments when the renko process :math:`X(i)` has a local
maximum or minimum, that is the process :math:`X(i) = P(\tau_i)` changes its direction, and the sequence
:math:`\{\tau^b_n\}` defines the time moments when the local maximum or minimum is detected.

More precisely, when take :math:`\tau^a_0 = \tau_0` and :math:`\tau^b_0 = \tau_1` then

.. math::
    \tau^b_n = min\{\tau_i > \tau^b_{n-1}: (P(\tau_i) − P(\tau_{i-1}))(P(\tau_{i-1}) − P(\tau_{i-2})) < 0\}, 

.. math::
    \tau^a_n = \{\tau_{i - 1} : \tau^b_n = \tau_i\}.

Kagi construction
*****************

The Kagi construction is similar to the Renko construction with the only difference being that to create the sequence
of time moments :math:`\{(\tau^a_n, \tau^b_n), n = 0, 1, ..., M\}` for the Kagi construction we use local maximums
and minimums of the process :math:`P(t)` rather than the process :math:`X(i)` derived from it. The sequence
:math:`\{\tau^a_n\}` then defines the time moments when the price process :math:`P(t)` has a local maximum or
minimum and the sequence :math:`\{\tau^b_n\}` defines the time moments when that local maximum or minimum is recognized,
that is, the time when the process :math:`P(t)` moves away from its last local maximum or minimum by a
distance equal to :math:`H`.

More precisely, :math:`\tau^a_0`, :math:`\tau^b_0` and :math:`S_0` is defined as,

.. math::
    \tau^b_0 = inf\{u \in [0, T] : \max \limits_{t \in [0,u]} P(t) − \min \limits_{t \in [0,u]} P(t) = H\},

.. math::
    \tau^a_0 = inf\{u < \tau^b_0: |P(u) − P(\tau^b_0)| = H\},

.. math::
    S_0 = sign(P(\tau^a_0) − P(\tau^b_0)),

where :math:`S_0` can take two values: :math:`1` for a local maximum and :math:`−1` for a local minimum.

Then we define :math:`(\tau^a_n, \tau^b_n)`, :math:`n > 0` recursively. The construction of the full sequence
:math:`\{(\tau^a_n, \tau^b_n), n = 0, 1, ..., M\}` is done inductively by alternating the following cases.

:math:`Case\ 1: \ \ S_{n-1} = -1`

if :math:`S_{n-1} = -1`, then :math:`\tau^a_n, \tau^b_n` and :math:`S_n` is defined as,

.. math::
    \tau^b_n = inf\{u \in [\tau^a_{n-1}, T] : P(u) − \min \limits_{t \in [\tau^a_{n-1}\ \ ,\ u]} P(t) = H\},

.. math::
    \tau^a_n = inf\{u < \tau^b_n: P(u) = \min \limits_{t \in [\tau^a_{n-1}\ \ ,\ \tau^b_n]} P(t)\},

.. math::
    S_n = 1.


:math:`Case\ 2: \ \ S_{n-1} = 1`

if :math:`S_{n-1} = 1`, then :math:`\tau^a_n, \tau^b_n` and :math:`S_n` is defined as,

.. math::
    \tau^b_n = inf\{u \in [\tau^a_{n-1}, T] : \max \limits_{t \in [\tau^a_{n-1}\ \ ,\ u]} P(t) - P(u) = H\},

.. math::
    \tau^a_n = inf\{u < \tau^b_n: P(u) = \max \limits_{t \in [\tau^a_{n-1}\ \ ,\ \tau^b_n]} P(t)\},

.. math::
    S_n = -1.

H-statistics
############

H-inversion
***********

H-inversion counts the number of times the process :math:`P(t)` changes its direction for selected :math:`H`,
:math:`T` and :math:`P(t)`. It is given by

.. math::
    N_T (H, P) = \max \{n : \tau^{b}_{n} = T\} = N,

where :math:`H` denotes the threshold of the H-construction, and :math:`P` denotes the process :math:`P(t)`.

H-distances
***********

H-distances counts the sum of vertical distances between local maximums and minimums to the power :math:`p`. It is given by

.. math::
    V^p_T (H, P) = \sum_{n = 1}^{N}|P(\tau^a_n) − P(\tau^a_{n−1})|^p.

H-volatility
************

H-volatility of order p measures the variability of the process :math:`P(t)` for selected :math:`H` and :math:`T`. It is given by

.. math::
    \xi^p_T = {V^p_T (H, P)}/{N_T (H, P)}.

Strategies
##########

Momentum Strategy
*****************

The investor buys (sells) an asset at a stopping time :math:`\tau^b_n` when he or she recognizes that the process
passed its previous local minimum (maximum)and the investor expects a continuation of the movement.
The signal :math:`s_t` is given by

.. math::
    s_t = \left\{\begin{array}{l}
    +1,\ if\ t = \tau^b_n\ and\ P(\tau^b_n) - P(\tau^a_n) > 0\\
    -1,\ if\ t = \tau^b_n\ and\ P(\tau^b_n) - P(\tau^a_n) < 0\\
    0,\ otherwise
    \end{array}\right.

where :math:`+1` indicates opening a long trade or closing a short trade, :math:`-1` indicates opening a short trade
or closing a long trade and :math:`0` indicates holding the previous position.

The profit from one trade according to the momentum H-strategy over time from :math:`\tau^b_{n−1}` to :math:`\tau^b_{n}` is

.. math::
    Y_{\tau^b_n} = (P(\tau^b_n) − P(\tau^b_{n−1})) · sign(P(\tau^a_n) − P(\tau^a_{n−1}))

and the total profit from time :math:`0` till time :math:`T` is

.. math::
    Y_T(H, P) = (\xi^1_T (H, P) − 2H) \cdot N_T (H, P)

Contrarian Strategy
*******************

The investor sells (buys) an asset at a stopping time :math:`\tau^b_n` when he or she decides that the process
has passed far enough from its previous local minimum (maximum), and the investor expects a movement reversion.
The signal :math:`s_t` is given by

.. math::
    s_t = \left\{\begin{array}{l}
    +1,\ if\ t = \tau^b_n\ and\ P(\tau^b_n) - P(\tau^a_n) < 0\\
    -1,\ if\ t = \tau^b_n\ and\ P(\tau^b_n) - P(\tau^a_n) > 0\\
    0,\ otherwise
    \end{array}\right.

where :math:`+1` indicates opening a long trade or closing a short trade, :math:`-1` indicates opening a short
trade or closing a long trade and :math:`0` indicates holding the previous position.

The profit from one trade according to the momentum H-strategy over time from :math:`\tau^b_{n−1}` to :math:`\tau^b_{n}` is

.. math::
    Y_{\tau^b_n} = (P(\tau^b_n) − P(\tau^b_{n−1})) · sign(P(\tau^a_{n−1}) - P(\tau^a_n)),

and the total profit from time $0$ till time $T$ is

.. math::
    Y_T(H, P) = (2H - \xi^1_T (H, P)) \cdot N_T (H, P).

Properties
**********

It is clear that the choice of H-strategy depends on the value of H-volatility. If :math:`\xi^1_T > 2H`,
then to achieve a positive profit the investor should
employ a momentum H -strategy; If :math:`\xi^1_T < 2H` then the investor should use a contrarian H-strategy.

Suppose :math:`P(t)` follows the Wiener process, the H-volatility :math:`\xi^1_T = 2H`. As a result,
it is impossible to profit from the trading on the process :math:`P(t)`. We can also see that H-volatility
:math:`\xi^1_T = 2H` is a property of a martingale. Likewise :math:`\xi^1_T > 2H` could be a property of a
sub-martingale or a super-martingale or a process regularly switching over time from a sub-martingale to a
super-martingale and back.

In this paper, the author proposes that for any mean-reverting process, regardless of its distribution,
the H-volatility is less than :math:`2H`. Hence, theoretically, trading the mean-reverting process by the
contrarian H-strategy is profitable for any choice of :math:`H`.

Pairs Selection
###############

* Purpose: Select trading pairs from the assets pool by using the properties of the H-construction.
* Algorithm:

    1. Determine the assets pool and the length of historical data.

    2. Take log-prices of all assets based on the history, combine them in all possible pairs and build a spread process for each pair.

        * :math:`spread_{ij} = log(P_i) - log(P_j)`

    3. For each spread process, calculate its standard deviation, and set it as the threshold of the H-construction.

    4. Determine the construction type of the H-construction.

        * It could be either Renko or Kagi.

    5. Build the H-construction on the spread series formed by each possible pair.

    6. The top N pairs with the highest/lowest H-inversion are used for pairs trading.

        * Mean-reverting process tends to have higher H-inversion.

Implementation
##############

HConstruction
*************

.. py:currentmodule:: arbitragelab.time_series_approach.h_strategy

.. autoclass:: HConstruction
    :members: __init__

.. automethod:: HConstruction.h_inversion

.. automethod:: HConstruction.h_distances

.. automethod:: HConstruction.h_volatility

.. automethod:: HConstruction.get_signals

.. automethod:: HConstruction.extend_series

HSelection
**********

.. autoclass:: HSelection
    :members: __init__

.. automethod:: HSelection.select

.. automethod:: HSelection.get_pairs

Examples
########

HConstruction
*************

.. code-block::

    # Importing packages
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import yfinance as yf
    from arbitragelab.time_series_approach.h_strategy import HConstruction

    # Loading data
    data =  yf.download("KO PEP", start="2019-01-01", end="2020-12-31")["Adj Close"]

    # Constructing spread series
    series = np.log(data["KO"]) - np.log(data["PEP"])

    # Creating a class object
    threshold = series["2019"].std()
    hc = HConstruction(series["2020"], threshold, "Kagi")

    #Getting H-statistics
    print("H-inversion:", hc.h_inversion())
    print("H-distances:", hc.h_distances())
    print("H-volatility:", hc.h_volatility())

    #Getting signals
    signals = hc.get_signals("contrarian")

    # A quick backtesting
    positions = signals.replace(0, np.nan).ffill()
    returns = data["KO"]["2020"].pct_change() - data["PEP"]["2020"].pct_change()
    total_returns = ((positions.shift(1)*returns).dropna() + 1).cumprod()
    total_returns.plot()
    plt.show()

HSelection
**********

.. code-block::

    # Importing packages
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import yfinance as yf
    from arbitragelab.time_series_approach.h_strategy import HSelection

    # Loading data
    tickers = "AAPL MFST AMZN FB GOOGL GOOG TSLA NVDA JPM"
    data = yf.download(tickers, start="2019-01-01", end="2020-12-31")["Adj Close"]

    # Creating a class object
    hs = HSelection(data)
    hs.select()

    # Getting pairs
    pairs = hs.get_pairs(5, "highest", False)
    for p in pairs:
        print("H-inversion:", p[0], "Threshold for H-construction:", p[1], "Pairs:", p[2])

Research Notebooks
******************

The following research notebook can be used to better understand the method described above.

* `H-Strategy`_

.. _`H-Strategy`: https://github.com/hudson-and-thames/arbitrage_research/blob/H_strategy/Time%20Series%20Approach/H_strategy.ipynb

References
##########

* `Bogomolov, T., Pairs trading based on statistical variability of the spread process. Quantitative Finance, 13(9): 1411–1430 <https://www.researchgate.net/publication/263339291_Pairs_trading_based_on_statistical_variability_of_the_spread_process>`_.