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

