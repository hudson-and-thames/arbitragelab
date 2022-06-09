.. _spread_selection-cointegration_spread_selection:

====================================
Cointegration Rules Spread Selection
====================================

The rules selection flow diagram from `A Machine Learning based Pairs Trading Investment Strategy <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`__.
by Simão Moraes Sarmento and Nuno Horta.

The rules that each spread needs to pass are:

- The spread's constituents are cointegrated. Literature suggests cointegration performs better than minimum distance and correlation approaches
- The spread's spread Hurst exponent reveals a mean-reverting character. Extra layer of validation.
- The spread's spread diverges and converges within convenient periods.
- The spread's spread reverts to the mean with enough frequency.

.. figure:: images/pairs_selection_rules_diagram.png
    :align: center

To test for cointegration, the framework proposes the application of the Engle-Granger test, due
to its simplicity. One critic `Armstrong (2001) <http://doi.org/10.1007/978-0-306-47630-3>`__ points
at the Engle-Granger test sensitivity to the ordering of variables. It is a possibility that one of
the relationships will be cointegrated, while the other will not. This is troublesome because we would
expect that if the variables are truly cointegrated the two equations will yield the same conclusion.

To mitigate this issue, the original paper proposes that the Engle-Granger test is run for the two
possible selections of the dependent variable and that the combination that generated the lowest t-statistic
is selected. Further work in `Hoel (2013) <https://openaccess.nhh.no/nhh-xmlui/bitstream/handle/11250/169897/hoel2013.pdf?sequence=1>`__
adds on, "the unsymmetrical coefficients imply that a hedge of long / short is not the opposite of long / short ,
i.e. the hedge ratios are inconsistent".

A better solution is proposed and implemented, based on `Gregory et al. (2011) <http://dx.doi.org/10.2139/ssrn.1663703>`__
to use orthogonal regression – also referred to as Total Least Squares (TLS) – in which the residuals
of both dependent and independent variables are taken into account. That way, we incorporate the volatility
of both legs of the spread when estimating the relationship so that hedge ratios are consistent, and thus
the cointegration estimates will be unaffected by the ordering of variables.

Hudson & Thames research team has also found out that optimal (in terms of cointegration tests statistics) hedge ratios
are obtained by minimizng spread's half-life of mean-reversion. Alongside this hedge ration calculation method,
there is a wide variety of algorithms to choose from: TLS, OLS, Johansen Test Eigenvector, Box-Tiao Canonical Decomposition,
Minimum Half-Life, Minimum ADF Test T-statistic Value.

.. note::
    More information about the hedge ratio methods and their use can be found in the
    :ref:`Hedge Ratio Calculations <hedge_ratios-hedge_ratios>` section of the documentation.

Secondly, an additional validation step is also implemented to provide more confidence in the mean-reversion
character of the pairs’ spread. The condition imposed is that the Hurst exponent associated with the spread
of a given pair is enforced to be smaller than 0.5, assuring the process leans towards mean-reversion.

In third place, the pair's spread movement is constrained using the half-life of the mean-reverting process.
In the framework paper the strategy built on top of the selection framework is based on the medium term price
movements, so for this reason the spreads that either have very short (< 1 day) or very long mean-reversion (> 365 days)
periods were not suitable.

Lastly, we enforce that every spread crosses its mean at least once per month, to provide enough liquidity and
thus providing enough opportunities to exit a position.

.. note::
    In practice to calculate the spread of the pairs supplied by this module, it is important to also consider
    the hedge ratio as follows:

    :math:`S = leg1 - (hedgeratio_2) * leg2 - (hedgeratio_3) * leg3 - .....`

.. warning::
    The pairs selection function is very computationally heavy, so execution is going to be long and might slow down your system.

.. note::
    The user may specify thresholds for each pair selector rule from the framework described above. For example, Engle-Granger test 99%
    threshold may seem too strict in pair selection which can be decreased to either 95% or 90%. On the other hand,
    the user may impose more strict thresholds on half life of mean reversion.

.. note::
    H&T teams has extended pair selection rules to higher dimensions such that filtering rules can be applied to any spread, not only
    pairs. As a result, the module can be applied in statistical arbitrage applications.

Implementation
**************


.. automodule:: arbitragelab.spread_selection.cointegration

.. autoclass:: CointegrationSpreadSelector
   :members: __init__


.. automethod:: CointegrationSpreadSelector.select_spreads
.. automethod:: CointegrationSpreadSelector.generate_spread_statistics
.. automethod:: CointegrationSpreadSelector.construct_spreads
.. automethod:: CointegrationSpreadSelector.apply_filtering_rules

Examples
########

.. code-block::

    # Importing packages
    import pandas as pd
    import numpy as np
    from arbitragelab.spread_selection import CointegrationSpreadSelector

    data = pd.read_csv('sp100_prices.csv', index_col=0, parse_dates=[0])
    input_spreads = [('ABMD', 'AZO'), ('AES', 'BBY'), ('BKR', 'CE'), ('BKR', 'CE', 'AMZN')]
    pairs_selector = CointegrationSpreadSelector(prices_df=data, baskets_to_filter=input_spreads)
    filtered_spreads = pairs_selector.select_spreads(hedge_ratio_calculation='TLS',
                                                     adf_cutoff_threshold=0.9,
                                                     hurst_exp_threshold=0.55,
                                                     min_crossover_threshold=0,
                                                     min_half_life=20)

    # Statistics logs data frame.
    logs = pairs_selector.selection_logs.copy()

    # A user may also specify own constructed spread to be tested.
    spread = pd.read_csv('spread.csv', index_col=0, parse_dates=[0])
    pairs_selector = CointegrationSpreadSelector(prices_df=None, baskets_to_filter=None)
    # Using log_info=True to save stats.
    stats = pairs_selector.generate_spread_statistics(spread, log_info=True)
    print(pairs_selector.selection_logs)
    print(stats)
    filtered_spreads = pairs_selector.apply_filtering_rules(adf_cutoff_threshold=0.99,
                                                            hurst_exp_threshold=0.5)


Research Notebooks
##################

The following research notebook can be used to better understand the Cointegration Rules Spread Selection described above.

* `ML based Pairs Selection`_

.. _`ML based Pairs Selection`: https://hudsonthames.org/notebooks/arblab/ml_based_pairs_selection.html

.. raw:: html

    <a href="https://hudthames.tech/3gFGwy8"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
    <a href="https://hudthames.tech/2S03R58"><button style="margin: 20px; margin-top: 0px">Download Sample Data</button></a>

References
##########

* `Sarmento, S.M. and Horta, N., A Machine Learning based Pairs Trading Investment Strategy. <https://www.springer.com/gp/book/9783030472504>`__

* `Armstrong, J.S. ed., 2001. Principles of forecasting: a handbook for researchers and practitioners (Vol. 30). Springer Science & Business Media. <http://doi.org/10.1007/978-0-306-47630-3>`__

* `Hoel, C.H., 2013. Statistical arbitrage pairs: can cointegration capture market neutral profits? (Master's thesis). <https://openaccess.nhh.no/nhh-xmlui/bitstream/handle/11250/169897/hoel2013.pdf?sequence=1>`__

* `Gregory, I., Ewald, C.O. and Knox, P., 2010, November. Analytical pairs trading under different assumptions on the spread and ratio dynamics. In 23rd Australasian Finance and Banking Conference. <http://dx.doi.org/10.2139/ssrn.1663703>`__
