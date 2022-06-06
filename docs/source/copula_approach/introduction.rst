.. _copula_approach-introduction:

.. Note::
    These descriptions closely follow the following two papers:

    `Pairs trading: a copula approach. (2013) <https://link.springer.com/article/10.1057/jdhf.2013.1>`__ by Liew, Rong Qi, and Yuan Wu.

    `Trading strategies with copulas. (2013) <https://www.researchgate.net/publication/318054326>`__ by Stander, Yolanda, Daniël Marais, and Ilse Botha.

============
Introduction
============

Copula is a relatively new analysis tool for pairs trading, compared to more traditional approaches such
as distance and cointegration. Since pairs trading can be considered one of the long/short equity strategies,
copula enables a more nuanced and detailed understanding of the traded pair when compared to, say, Euclidean distance
approaches, thereby generating more reasonable trading opportunities for capturing relative mispricing.

Consider having a pair of cointegrated stocks. By analyzing their time series, one can calculate their standardized
price gap as part of a distance approach, or project their long-run mean as in a cointegrated system as part of a
cointegration approach. However, none of the two methods are built with the distributions from their time series.
The copula model naturally incorporates their marginal distributions, together with other interesting properties from
each copula, e.g., tail dependency for capturing rare and/or extreme moments like large, cointegrated swings in the
market.

Briefly speaking, copula is a tool to capture details of how two random variables are "correlated". By having a more
detailed modeling framework, we expect the pairs trading strategy followed to be more realistic and robust and possibly 
to bring more trading opportunities.

.. figure:: images/copula_marginal_dist_demo.png
    :scale: 30 %
    :align: center

    An illustration of the conditional distribution function of V for a given value of U and the conditional
    distribution function of U for a given value of V using the N14 copula dependence structure.
    An example from
    "Trading strategies with copulas."
    by Stander, Yolanda, Daniël Marais, and Ilse Botha.

Tools presented in this module enable the user to:

* Transform and fit pair's price data to a given type of copula;

* Sample and plot from a given copula;

* Generate trading positions given the pair's data using a copula:

    - Feed in training lists (i.e., data from 2016-2019) and thus generate a position list.

    - Feed in a single pair's data point (i.e., EOD data from just today) and thus generate a single position.

There are 8 commonly used pure copulas that are now available: :code:`Gumbel`, :code:`Frank`, :code:`Clayton`, :code:`Joe`,
:code:`N13`, :code:`N14`, :code:`Gaussian` and :code:`Student` (Student-t) under :code:`Copula`.
Also there are 2 mixed copulas :code:`CTGMixCop` (Clayton-Student-Gumbel) and :code:`CFGMixCop` (Clayton-Frank-Gumbel) under
:code:`MixedCopula`.
They share some common repertoire of methods and attributes.
However, most of the time for trading purposes, the user is not expected to directly use the copulas.
All trading related functionalities are included in the :code:`BasicCopulaTradingRule` and :code:`MispricingIndexCopulaTradingRule` class.

The user may choose to fit the pair's data to all provided copulas, then compare the information criterion scores (AIC,
SIC, HQIC, Log-likelihood) to decide the best copula. One can further use the fitted copula to generate trading positions
by giving thresholds from data.
