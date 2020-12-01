.. _heat_potentials-heat_potentials:


.. note::
   The following documentation closely follows the article by Alexandr Lipton and Marcos Lopez de Prado:
   `"A closed-form solution for optimal mean-reverting trading strategies"<https://ssrn.com/abstract=3534445>`__

   As well as the book by Marcos Lopez de Prado:
   `"Advances in Financial Machine Learning"<https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089>`__

====================================================================
A closed-form solution for optimal mean-reverting trading strategies
====================================================================

As stated in the the paper by Marcos Lopez de Prado: "When prices reflect all available information,
they oscillate around an equilibrium level. This oscillation is the result of the temporary market impact caused by
waves of buyers and sellers." Which means that if we can consider the asset prices to be mean-reverted, then we have
a clear toolkit for it's approximation, namely, the Ornstein-Uhlenbeck process.

Attempting to monetize this oscillations market makers provide liquidity by entering the long position if the asset is
underpriced - hence, the price is lower than the equilibrium, and short position if the situation is reversed. The
position is then held until one of three outcomes occurs:

* they achieve a targeted profit
* they experience a maximum tolerated loss
* the position is held longer then the maximum tolerated horizon

The main problem that arises now is how to define the optimal profit-taking and stop-loss levels. In this module to
obtain the solution we utilize the method introduced by Alexandr Lipton and Marcos Lopez de Prado that utilizes
the method of heat potentials widely applied in physics to Sharpe ratio (later SR) maximization problem with respect to the border
values of our exit corridor.

Problem definition
##################

We suppose an investment strategy S invests in i = 1,...I opportunities or bets. At each opportunity i, S takes
a position of :math:`m_i` units of security X, where :math:`m_i \in (\infty; -\infty). The transaction that
entered such opportunity was priced at a value :math:`m_i P_{1,0}`, where :math:`P_{i,0}` is the average price per unit
at which the mi securities were transacted.

.. note::

    In this approach we use the volume clock metric instead of the time-based metric. More in that in paper
    `"The Volume Clock: Insights into the High Frequency Paradigm"<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2034858>`__
    by David Easley, Marcos Lopez de Prado and Maureen O'Hara

As other market participants continue to transact security X, we can mark-to-market (MtM) the value of
that opportunity i after t observed transactions as :math:`m_i P_{1,t}`. Where :math:`m_i P_{1,t}` represents the
value of opportunity i if it were liquidated at the price observed in the market after t transactions. Accordingly,
we can compute the MtM p/l of opportunity i after t transactions as :math:`\pi_{i,T_i}=m(P_{1,t}-P_{i,0})`
The exiting opportunity arizes in the following scenarios:

* :math:`\pi_{i,T_i}\geq \bar{\pi}, where \bar{\pi}>0 is a profit-taking threshold
* :math:`\pi_{i,T_i}\leq \underbar{\pi}, where \underbar{\pi}<0 is a stop-loss level

Consider the OU process representing the :math:`{P_i}` series of prices

.. math::

    P_i{i,t} - \mathbb{E}_0[P_i{i,t}] = \mu(\mathbb{E}_0 - P_i{i,t-1}) + \sigma\epsilon_{i,t}

In order to calculate the Sharpe ratio we need to reformulate our problem in terms of heat potentials.

Suppose *S* - long investment strategy with p/l driven by the OU process:

.. math::

    dx' = \mu'(\theta'-x')dt'+sigma'dW_{t'}, x'(0) = 0

and a trading rule :math:`R = {\bar{\pi}',\underbar{\pi}',T'}`. Now we transform it to use its steady-state
by performing scaling to remove superfluous parameters.

.. math::

    t = \mu't',\ , T = \mu'T', x = \frac{\sqrt{\mu'}}{\sigma'} x',

    \theta = \frac{\sqrt{\mu'}}{\sigma'} \theta',\ \bar{\pi} = \frac{\sqrt{\mu'}}{\sigma'} \bar{\pi}',
    \ \underbar{\pi} = \frac{\sqrt{\mu'}}{\sigma'} \underbar{\pi}'

and get

.. math::

    dx = (\theta-x)dt + dW_t, \ \bar{\pi}' \leq x \leq \underbar{\pi},\ 0 \leq t \leq T

where :math:`\theta` is an expected value and its standart deviation is given by :math:`frac{1}{\sqrt{2}}`

According to the trading rule we exit the trade in one of the three scenarios:

* price hits a targeted profit \bar{\pi}
* price hits \underbar{\pi} stop-loss level
* the trade expires at t=T

.. tip::

    For a short strategy reverses the roles of :math:`{\bar{\pi}',\underbar{\pi}}',
    :math:`-\underbar{\pi}` equals the profit taken when the price hits :math:`\underbar{\pi}` and
    :math:`-\bar{\pi}` losses are incurred while price hits :math:`-\bar{\pi}`

Hence, we can restrict ourself to case with :math:`\theta \geq 0'

