.. _optimal_mean_reversion-heat_potentials:


.. note::
   The following documentation closely follows the article by Alexandr Lipton and Marcos Lopez de Prado:
   `"A closed-form solution for optimal mean-reverting trading strategies" <https://ssrn.com/abstract=3534445>`__

   As well as the book by Marcos Lopez de Prado:
   `"Advances in Financial Machine Learning" <https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089>`__

====================================================================
A closed-form solution for optimal mean-reverting trading strategies
====================================================================

As stated in the paper by Marcos Lopez de Prado: "When prices reflect all available information,
they oscillate around an equilibrium level. This oscillation is the result of the temporary market impact caused by
waves of buyers and sellers." Which means that if we can consider the asset prices to be mean-reverted, then there is a
clear toolkit for it's approximation, namely, the Ornstein-Uhlenbeck process.

Attempting to monetize this oscillations market makers provide liquidity by entering the long position if the asset is
underpriced - hence, the price is lower than the equilibrium, and short position if the situation is reversed. The
position is then held until one of three outcomes occurs:

* they achieve a targeted profit
* they experience a maximum tolerated loss
* the position is held longer then the maximum tolerated horizon

The main problem that arises now is how to define the optimal profit-taking and stop-loss levels. In this module to
obtain the solution we utilize the method introduced by Alexandr Lipton and Marcos Lopez de Prado that applies the widely
applied in physics method of heat potentials to Sharpe ratio maximization problem. The maximization is performed
with respect to the border values of our exit corridor.

Problem definition
##################

We suppose an investment strategy S invests in i = 1,...I opportunities or bets. At each opportunity i, S takes
a position of :math:`m_i` units of security X, where :math:`m_i \in (\infty; -\infty)`. The transaction that
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
The exiting opportunity arises in the following scenarios:

* :math:`\pi_{i,T_i}\geq \bar{\pi}`, where :math:`\bar{\pi}>0` is a profit-taking threshold
* :math:`\pi_{i,T_i}\leq \underline{\pi}`, where :math:`\underline{\pi}<0` is a stop-loss level

Consider the OU process representing the :math:`{P_i}` series of prices

.. math::

    P_{i,t} - \mathbb{E}_0[P_{i,t}] = \mu(\mathbb{E}_0 - P_{i,t-1}) + \sigma\epsilon_{i,t}

In order to calculate the Sharpe ratio we need to reformulate our problem in terms of heat potentials.

Suppose *S* - long investment strategy with p/l driven by the OU process:

.. math::

    dx' = \mu'(\theta'-x')dt'+\sigma'dW_{t'}, x'(0) = 0

and a trading rule :math:`R = {\bar{\pi}',\underline{\pi}',T'}`. Now we transform it to use its steady-state
by performing scaling to remove superfluous parameters.

.. math::

    t = \mu't',\ T = \mu'T',\ x = \frac{\sqrt{\mu'}}{\sigma'} x',\

    \theta = \frac{\sqrt{\mu'}}{\sigma'} \theta',\ \bar{\pi} = \frac{\sqrt{\mu'}}{\sigma'} \bar{\pi}',
    \ \underline{\pi} = \frac{\sqrt{\mu'}}{\sigma'} \underline{\pi}'

and get

.. math::

    dx = (\theta-x)dt + dW_t, \ \bar{\pi}' \leq x \leq \underline{\pi},\ 0 \leq t \leq T

where :math:`\theta` is an expected value and its standard deviation is given by :math:`\Omega=\frac{1}{\sqrt{2}}`

According to the trading rule we exit the trade in one of the three scenarios:

* price hits a targeted profit :math:`\bar{\pi}`
* price hits :math:`\underline{\pi}` stop-loss level
* the trade expires at t=T

.. tip::

    Short strategy reverses the roles of :math:`{\bar{\pi}',\underline{\pi}'}`:

    :math:`-\underline{\pi}` equals the profit taken when the price hits :math:`\underline{\pi}` and

    :math:`-\bar{\pi}` losses are incurred while price hits :math:`-\bar{\pi}`

Hence, we can restrict ourself to case with :math:`\theta \geq 0`

Sharpe ratio calculation
########################

To compute the approximate SR we need to perform the four-step numerical evaluation.

**Step 1: Defining a calculation grid**

First of all we define the grid :math:`\upsilon` based on which we will perform our numerical calculation:

.. math::

    0=\upsilon_0<\upsilon_1<...<\upsilon_n=\Upsilon,\  \upsilon(t) = \frac{1 - e^{-2(T-t)}}{2}

**Step 2: Numerically calculate helper functions** :math:`\bar{\epsilon}, \underline{\epsilon}, \bar{\phi}, \underline{\phi}`

We are going to use the classical method of heat potentials to calculate the SR.
As a preparation, in this step we solve the two sets of Volterra equations by using the trapezoidal rule of integral calculation.

**Step 3: Calculate the values of** :math:`\hat{E}(\Upsilon,\bar{\omega})` **and** :math:`\hat{F}(\Upsilon,\bar{\omega})`

We need to compute these functions at one point :math:`\Upsilon,\bar{\omega}`, which can be done by approximation of the integrals using the
trapezoidal rule:

.. math::

    \hat{E}(\Upsilon,\bar{\omega}) = \frac{1}{2} \sum_{i=1}^k(\underline{w}_{n,i}\underline{\epsilon}_i + \underline{w}_{n,i-1}\underline{\epsilon}_{i-1} + \bar{w}_{n,i}\bar{\epsilon}_i + \bar{w}_{n,i-1}\bar{\epsilon}_{i-1})(\upsilon_i - \upsilon_{i-1})

    \hat{F}(\Upsilon,\bar{\omega}) = \frac{1}{2} \sum_{i=1}^k(\underline{w}_{n,i}\underline{\phi}_i + \underline{w}_{n,i-1}\underline{\phi}_{i-1} + \bar{w}_{n,i}\bar{\phi}_i + \bar{w}_{n,i-1}\bar{\phi}_{i-1})(\upsilon_i - \upsilon_{i-1})

Where *w* are the weights.

**Step 4: calculate the SR using the obtained values**

The previously computed functions :math:`\hat{E}(\Upsilon,\bar{\omega})` and :math:`\hat{F}(\Upsilon,\bar{\omega})`
are substituted into the following formula to calculate the Sharpe ratio.

.. math::
    SR = \frac{\hat{E}(\Upsilon,\bar{\omega}) - \frac{\bar{\omega}-\theta}{ln(1-2\Upsilon)}}{\sqrt{\hat{F}(\Upsilon,\bar{\omega})} - (\hat{E}(\Upsilon,\bar{\omega}))^2 + \frac{4(\Upsilon + ln(1-2\Upsilon)(\bar{\omega})\hat{E}(\Upsilon,\bar{\omega})}{(ln(1-2\Upsilon))^2}}

To find the optimal thresholds for the data provided by user we maximize the calculated SR with respect to
:math:`\bar{\pi}\geq0,\underline{\pi}\leq0`

.. math::

    {\bar{\pi}*,\underline{\pi}*}=\underset{\bar{\pi}\geq0,\underline{\pi}\leq0}{\arg\max}\ SR


The ``HeatPotentials`` module of the ArbitrageLab package allows the user to calculate the threshold levels that
establish the trading rule for their data, the provided parameters are transformed to a steady-state solutions
internally and the reverse transformation is performed for the optimal threshold values.

Implementation
##############

First of all the user has to use ``fit`` for the parameters to be scaled to remove superfluous parameters,
and set up the delta for the grid calculation and maximum duration of trade.


.. py:currentmodule:: arbitragelab.optimal_mean_reversion.heat_potential.HeatPotentials

.. autofunction:: fit

To separately perform the optimization process we use the ``optimal_levels`` function.

.. autofunction:: optimal_levels

There is also a possibility to calculate the Sharpe ratio for chosen optimal levels and the maximum duration of the
trade of choice.

.. autofunction:: sharpe_calculation

To view the optimal levels scaled back to initial parameters the ``description`` function is used.

.. autofunction:: description

Example
#######

.. code-block::

    from arbitragelab.optimal_mean_reversion import HeatPotentials
    from arbitragelab.optimal_mean_reversion.ou_model import OrnsteinUhlenbeck

    # Create the class object
    example = HeatPotentials()

    # Generating the sample OU data
    ou_data = OrnsteinUhlenbeck()

    data = ou_data.ou_model_simulation(n=1000, theta_given=0.03711, mu_given=65.3333,
                                sigma_given=0.3, delta_t_given=1/255)

    # To get the model parameters we need to fit the OU model to the data.

    # Assign the delta value
    ou_data.delta_t = 1/252

    # Model fitting
    ou_data.fit_to_portfolio(data)

    # Now we obtained the parameters to use for our optimization procedure
    theta,mu,sigma = ou_data.theta,ou_data.mu,np.sqrt(ou_data.sigma_square)

    # Establish the instance of the class
    example = HeatPotentials()

    # Fit the model and establish the maximum duration of the trade
    example.fit(ou_params=(theta, mu, sigma), delta_grid=0.1, max_trade_duration=0.03)


    # Calculate the initial optimal levels
    levels = example.optimal_levels()

    print(levels)

    # We can also calculate the Sharpe ratio for given scaled parameters
    sr = example.sharpe_calculation(max_trade_duration=1.9599, optimal_profit=5.07525, optimal_stop_loss=-3.41002)

    print(sr)

    # To get the results scaled back to our initial model we call the description function
    example.description()

