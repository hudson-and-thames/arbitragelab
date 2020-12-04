.. _time_series_approach-distance_approach:

.. note::
   The following documentation closely follows a book by Simão Moraes Sarmento, and Nuno Horta
    `"A Machine Learning based Pairs Trading Investment Strategy" <https://www.springer.com/gp/book/9783030472504>`__.
=============================
Quantile Time Series Strategy
=============================

...

Auto ARIMA
##########

...

Implementation
**************

.. py:currentmodule:: arbitragelab.time_series_approach.arima_predict

.. autoclass:: AutoARIMAForecast

Quantile Time Series Strategy
#############################

...

Implementation
**************

.. py:currentmodule:: arbitragelab.time_series_approach.quantile_time_series

.. autoclass:: QuantileTimeSeriesTradingStrategy


Examples
########

Code Example
************

.. code-block::

   # Importing packages
   import pandas as pd
   from arbitragelab.distance_approach.quantile_time_series import QuantileTimeSeriesTradingStrategy

Research Notebooks
******************

The following research notebook can be used to better understand the time series approach described above.

* `Quantile Time Series Strategy`_

.. _`Quantile Time Series Strategy`: https://github.com/Hudson-and-Thames-Clients/arbitrage_research/blob/master/Distance%20Approach/basic_distance_approach.ipynb


References
##########

* ...
