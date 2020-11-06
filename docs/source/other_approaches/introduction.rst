.. _other_approaches-introduction:

.. Note::

    These descriptions closely follow a book by Ernest P. Chan:
    `Algorithmic Trading: Winning Strategies and Their Rationale <https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146>`__.

============
Introduction
============

PCA Approach
############

This module shows how the Principal Component Analysis can be used to generate trading signals.
The main idea of the method is to consider residuals or idiosyncratic components of returns and
to model them as a mean-reverting process.

.. figure:: images/pca_approach_portfolio.png
    :scale: 60 %
    :align: center

    Performance of a portfolio composed using the PCA approach in comparison to the market cap portfolio.
    An example from `Statistical Arbitrage in the U.S. Equities Market <https://math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb20090616.pdf>`__.
    by Marco Avellaneda and Jeong-Hyun Lee.

