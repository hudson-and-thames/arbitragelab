.. _ml_approach-threshold_ar:

.. note::
   The following documentation follows the work of `Dunis et al. (2006) <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.568.7460&rep=rep1&type=pdf>`_ that is based on `Enders and Granger (1998)  <https://doi.org/10.2307/1392506>`_

.. warning::
   In order to use this module, you should additionally install *TensorFlow v2.2.1.* and *Keras v2.3.1.*
   For more details, please visit our :ref:`ArbitrageLab installation guide <getting_started-installation>`.

=========================
Threshold Auto Regression
=========================

Introduction
############

The gasoline crack spread can be interpreted as the profit margin gained by
processing crude oil into unleaded gasoline. It is simply the monetary
difference between West Texas Intermediate crude oil and Unleaded Gasoline,
both of which are traded on the New York Mercantile Exchange (NYMEX).

.. math:: 
    
    S_{t} = GAS_t - WTI_t

:math:`S_{t}` is the price of the spread at time :math:`t` (in \$ per
barrel), :math:`GAS_t` is the price of unleaded gasoline at time :math:`t`
(in \$ per barrel), and :math:`WTI_t` is the price of West Texas Intermediate
crude oil at time :math:`t` (in \$ per barrel).

In `Dunis et al. (2006) <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.568.7460&rep=rep1&type=pdf>`_ 
the case is made that the crack spread exhibits asymmetry at the \$5
dollar mark, with seemingly larger moves occurring on the upside of the
long-term 'fair value' than on the downside.

Cointegration was first introduced by `(Engle and Granger 1987) <https://doi.org/10.2307/1913236>`_. The technique
is to test the null hypothesis that any combination of two series contains
a unit root. If the null hypothesis is refuted and the conclusion is that a
unit root does not exist, the combination of the two series is cointegrated.

The phenomena of the spread exhibiting larger moves in one direction than in
the other, is known as asymmetry. Since the traditional unit root test has only
one parameter for the autoregressive estimate, it assumes upside and downside
moves to be identical or symmetric. Non-linear cointegration was first introduced
by `(Enders and Granger 1998) <https://doi.org/10.2307/1392506>`_, who extended the unit root test by considering upside
and downside moves separately, thus allowing for the possibility of asymmetric adjustment. 


TAR Model
#########

Enders and Granger extend the Dickey-Fuller test to allow for the unit root
hypothesis to be tested against an alternative of asymmetric adjustment. Here,
this is developed from its simplest form; consider the standard Dickey–Fuller test

.. math::
    
    \Delta \mu_{t} = p \mu_{t-1} + \epsilon_t 

where :math:`\epsilon_t` is a white noise process. The null hypothesis of
:math:`p=0` is tested against the alternative of :math:`p \neq 0`. :math:`p=0`
indicates that there is no unit root, and therefore :math:`\mu_i` is a stationary
series. If the series :math:`\mu_i` are the residuals of a long-run cointegration
relationship as indicated by Johansen, this simply results in a test of the validity
of the cointegrating vector (the residuals of the cointegration equation should 
form a stationary series).

The extension provided by `(Enders and Granger 1998) <https://doi.org/10.2307/1392506>`_ is to consider the upside and 
downside moves separately, thus allowing for the possibility of asymmetric 
adjustment. Following this approach;

.. math::

    \Delta \mu_{t} = I_t p_1 \mu_{i-1} + (1 - I_t) p_2 \mu_{i-1} + \epsilon_t

where :math:`I_t` is the zero-one ‘heaviside’ indicator function. 
This paper uses the following specification;

.. math::

    I_t = \left \{ {{1, if \mu_{t-1} \geq 0} \over {0, if \mu_{t-1} < 0}} \right.

Enders and Granger refer to the model defined above as threshold autoregressive
(TAR). The null hypothesis of symmetric adjustment is :math:`(H_0: p_1 = p_2)`,
which can be tested using the standard F-test (in this case the Wald test), with
an additional requirement that both :math:`p_1` and :math:`p_2` do not equal zero.
If :math:`p_1 \neq p_2`, cointegration between the underlying assets is non-linear.

Implementation
**************

.. py:currentmodule:: arbitragelab.ml_approach.tar

.. autoclass:: TAR
    :noindex:
    :members: __init__, fit, summary

Example
*******

.. code-block::

    # Importing packages
    import pandas as pd
    from arbitragelab.ml_approach.tar import TAR
    
    # Getting the dataframe with time series of asset returns
    data = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])

    # Calculating spread returns and std dev.
    spread_series = data['spread']
    
    # The TAR model expects a Zero mean series.
    demeaned_spread = (spread_series - spread_series.mean())

    # Initializing and fit TAR model.
    model = TAR(demeaned_spread)
    tar_results = model.fit()
    tar_results.summary()
    
    tar_results.fittedvalues.plot()

    # Show metrics on model fit.
    model.summary()
    

Research Notebooks
##################

The following research notebooks can be used to better understand the components of the model described above.

* `Fair Value Modeling`_ - showcases the use of the TAR model on the crack spread.

.. _`Fair Value Modeling`: https://hudsonthames.org/notebooks/arblab/fair_value_modeling.html

.. raw:: html

    <a href="https://hudthames.tech/3gFGwy8"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
    <a href="https://hudthames.tech/2S03R58"><button style="margin: 20px; margin-top: 0px">Download Sample Data</button></a>

References
##########

* `Dunis, C.L., Laws, J. and Evans, B., 2006. Modelling and trading the gasoline crack spread: A non-linear story. Derivatives Use, Trading & Regulation, 12(1-2), pp.126-145. <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.568.7460&rep=rep1&type=pdf>`__

* `Engle, R.F. and Granger, C.W., 1987. Co-integration and error correction: representation, estimation, and testing. Econometrica: journal of the Econometric Society, pp.251-276. <https://doi.org/10.2307/1913236>`_

* `Enders, W. and Granger, C.W.J., 1998. Unit-root tests and asymmetric adjustment with an example using the term structure of interest rates. Journal of Business & Economic Statistics, 16(3), pp.304-311. <https://doi.org/10.2307/1392506>`_
