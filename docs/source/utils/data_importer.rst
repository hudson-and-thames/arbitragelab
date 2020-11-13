.. _utils-data_importer:

=============
Data Importer
=============

This module features helpers to fetch pricing data commonly used by the quant community to benchmark algorithms on data that comes from the ‘real world’.

Asset Universes
###############

Get ticker collections of a specific asset universe.

.. py:currentmodule:: arbitragelab.util.DataImporter

.. autofunction:: get_sp500_tickers
.. autofunction:: get_dow_tickers

Price/Fundamental Data Fetcher
##############################

Pull data about a specific symbol/symbol list using the yfinance library.

.. autofunction:: get_price_data
.. autofunction:: get_ticker_sector_info

Pre/Post Processing Pricing Data
################################

After pulling/loading the pricing data, it has to be processed before being used in models.

.. autofunction:: get_returns_data
.. autofunction:: remove_nuns