.. _stochastic_control_approach-ou_model_mudchanatongsuk:

.. note::
    The following implementations and documentation closely following work:

    `Mudchanatongsuk, S., Primbs, J.A. and Wong, W., 2008, June. Optimal pairs trading: A stochastic control approach.
    <http://folk.ntnu.no/skoge/prost/proceedings/acc08/data/papers/0479.pdf>`__


========================
OU Model Mudchanatongsuk
========================

Introduction
############

In the paper corresponding to this module, the authors implement a stochastic control-based approach to the problem of pairs trading.
The paper models the log-relationship between a pair of stock prices as an Ornstein-Uhlenbeck process
and use this to formulate a portfolio optimization based stochastic control problem.
This problem is constructed in such a way that one may either
trade based on the spread (by buying and selling equal amounts of the stocks in the pair) or
place money in a risk-free asset. Then the optimal solution to this control problem
is obtained in closed form via the corresponding Hamilton-Jacobi-Bellman equation under a power utility on terminal wealth.

Modelling
#########

.. note::
    In this module and the corresponding paper,

    :math:`k` denotes the rate of mean reversion of the spread,
    :math:`\theta` denotes the long run mean and,
    :math:`\eta` denotes the standard deviation of the spread.


Let :math:`A(t)` and :math:`B(t)` denote respectively the prices of the
pair of stocks :math:`A` and :math:`B` at time :math:`t`. The authors assume that stock :math:`B`
follows a geometric Brownian motion,

.. math::

    d B(t)=\mu B(t) d t+\sigma B(t) d Z(t)

where :math:`\mu` is the drift, :math:`\sigma` is the volatility, and :math:`Z(t)` is a standard
Brownian motion.

Let :math:`X(t)` denote the spread of the two stocks at time :math:`t`,
defined as

.. math::

    X(t) = \ln(A(t)) − \ln(B(t))

The authors assume that the spread follows an Ornstein-Uhlenbeck process

.. math::

    d X(t)=k(\theta-X(t)) d t+\eta d W(t)

where :math:`k` is the rate of reversion, :math:`\eta` is the standard deviation and
:math:`\theta` is the long-term equilibrium level to which the spread reverts.

:math:`\rho` denotes the instantaneous correlation coefficient between :math:`Z(t)` and :math:`W(t)`.

Let :math:`V(t)` be the value of a self-financing pairs-trading portfolio and
let :math:`h(t)` and :math:`-h(t)` denote respectively the
portfolio weights for stocks :math:`A` and :math:`B` at time :math:`t`.


The wealth dynamics of the portfolio value is given by,

.. math::
    d V(t)= V(t)\left\{\left[h(t)\left(k(\theta-X(t))+\frac{1}{2} \eta^{2}+\rho \sigma \eta\right)+
    r\right] d t+\eta d W(t)\right\}



Given below is the formulation of the portfolio optimization pair-trading problem
as a stochastic optimal control problem. The authors assume that an investor’s preference
can be represented by the utility function :math:`U(x) = \frac{1}{\gamma} x^\gamma`
with :math:`x ≥ 0` and :math:`\gamma < 1`. In this formulation, our objective is to maximize expected utility at
the final time :math:`T`. Thus, the authors seek to solve


.. math::
    \begin{aligned}
    \sup _{h(t)} \quad & E\left[\frac{1}{\gamma}(V(T))^{\gamma}\right] \\[0.8em]
    \text { subject to: } \quad & V(0)=v_{0}, \quad X(0)=x_{0} \\[0.5em]
    d X(t)=& k(\theta-X(t)) d t+\eta d W(t) \\
    d V(t)=& V(t)((h(t)(k(\theta-X(t))+\frac{1}{2} \eta^{2}\\
    &+\rho \sigma \eta)+r) d t+\eta d W(t))
    \end{aligned}

Finally, the optimal weights are given by,

.. math::
    h^{*}(t, x)=\frac{1}{1-\gamma}\left[\beta(t)+2 x \alpha(t)-\frac{k(x-\theta)}{\eta^{2}}+
    \frac{\rho \sigma}{\eta}+\frac{1}{2}\right]



How to use this submodule
#########################

This submodule contains two public methods. One for estimating the parameters of the model using training data,
and the second method is for calculating the final optimal portfolio weights using evaluation data.

Step 1: Model fitting
*********************

We input the training data to the fit method which calculates the spread
and the estimators of the parameters of the model.

Implementation
==============


.. automodule:: arbitragelab.stochastic_control_approach.ou_model_mudchanatongsuk


.. autoclass:: OUModelMudchanatongsuk
   :members: __init__


.. automethod:: OUModelMudchanatongsuk.fit

.. note::
    Although the paper provides closed form solutions for parameter estimation,
    this module uses log-likelihood maximization to estimate the parameters as we found the closed form solutions provided to be unstable.

.. tip::
    To view the estimated model parameters from training data, call the ``describe`` function.

    .. automethod:: OUModelMudchanatongsuk.describe

    .. figure:: images/mudchana_describe.png
        :scale: 100 %
        :align: center
        :figclass: align-center


Step 2: Getting the Optimal Portfolio Weights
*********************************************

In this step we input the evaluation data and specify the utility function parameter :math:`\gamma`.

.. warning::
    As noted in the paper, please make sure the value of gamma is less than 1.


Implementation
==============

.. automethod:: OUModelMudchanatongsuk.optimal_portfolio_weights


Example
#######

We use GLD and GDX tickers from Yahoo Finance as the dataset for this example.

.. code-block::

    import yfinance as yf

    data1 =  yf.download("GLD GDX", start="2012-03-25", end="2016-01-09")
    data2 =  yf.download("GLD GDX", start="2016-02-21", end="2020-08-15")

    data_train_dataframe = data1["Adj Close"][["GLD", "GDX"]]
    data_test_dataframe = data2["Adj Close"][["GLD", "GDX"]]


In the following code block, we are initializing the class and firstly,
we use the fit method to generate the parameters of the model.
Then, we call ``describe`` to view the estimated parameters.
Finally, we use the out-of-sample test data to calculate the optimal portfolio weights using the fitted model.

.. code-block::

    from arbitragelab.stochastic_control_approach.ou_model_mudchanatongsuk import OUModelMudchanatongsuk

    sc = OUModelMudchanatongsuk()

    sc.fit(data_train_dataframe)

    print(sc.describe())

    plt.plot(sc.optimal_portfolio_weights(data_test_dataframe))
    plt.show()


Research Notebook
#################

The following research notebook can be used to better understand the approach described above.

* `Optimal Pairs Trading A Stochastic Control Approach`_

.. _`Optimal Pairs Trading A Stochastic Control Approach`:


References
##########

*     `Mudchanatongsuk, S., Primbs, J.A. and Wong, W., 2008, June. Optimal pairs trading: A stochastic control approach. <http://folk.ntnu.no/skoge/prost/proceedings/acc08/data/papers/0479.pdf>`__
