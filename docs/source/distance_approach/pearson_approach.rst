.. _distance_approach-pearson_approach:

.. note::
   The following documentation closely follows a paper by Chen et al. (2012):
   `Empirical investigation of an equity pairs trading strategy <http://www.pbcsf.tsinghua.edu.cn/research/chenzhuo/paper/Empirical%20Investigation%20of%20an%20Equity%20Pairs%20Trading%20Strategy.pdf>`_.

   As well as a paper by Perlin, M. S. (2009):
   `Evaluation of pairs-trading strategy at the Brazilian financial market <https://link.springer.com/article/10.1057/jdhf.2009.4>`_.

   And also a paper by Christopher Krauss (2015):
   `Statistical arbitrage pairs trading strategies: Review and outlook <https://www.econstor.eu/bitstream/10419/116783/1/833997289.pdf>`_.

================
Pearson Approach
================

After the distance approach was introduced in the paper by Gatev et al. (2006), a lot of research has been
conducted to further develop the original distance approach. One of the adjustments is the Pearson correlation
approach proposed by Chen et al.(2012). In this paper, the authors use the same data set and time frame as
in the work by Gatev et al.(2006) but they used Pearson correlation on return level for forming pairs.
In the formation period(5 years in the paper), pairwise return correlations for all pairs in the universe
are calculated based on monthly return data. Then the authors construct a new variable, Return Divergence
(:math:`D_{i j t}`), to capture the return divergence between a single stock’s return and its pairs-portfolio returns:

.. math::

    D_{i j t}=\beta\left(R_{i t}-R_{f}\right)-\left(R_{j t}-R_{f}\right)


where :math:`\beta` denotes the regression coefficient of stock's monthly return :math:`R_{i t}` on its
pairs-portfolio return :math:`R_{j t}` and :math:`R_{f}` is the risk-free rate

The hypothesis in this approach is that if a stock’s return deviates more from its pairs portfolio returns
than usual, this divergence should be reversed in the next month and expecting abnormally higher returns than
other stocks.

Therefore, after calculating the return divergence of all the stocks, decile portfolios are constructed where
stocks with high return divergence have higher subsequent returns. Therefore after all stocks are sorted in
descending order based on their previous month’s return divergence, decile 10 is “long stocks” and decile 1
is “short stocks”.

Pairs Portfolio Formation
#########################

This stage of PearsonStrategy consists of the following steps:

1. **Data preprocessing**

As the method has to compute all of the pairs’ correlation in the following steps, for :math:`m` stocks,
there are :math:`\frac{m*(m-1)}{2}` correlations to be computed in the formation period. As the number of
observations for the correlations grows exponentially with the number of stocks, the estimation is
computationally intensive.

Therefore, to reduce the computation burden, this method uses monthly stock returns data in the formation
period (ex. 60 monthly stock returns if the formation period is 5 years). For the given daily price data,
the method calculates the monthly returns before moving into the next steps.

2. **Finding pairs**

Using monthly stock returns data in the formation period, for each stock, the method finds :math:`n` stocks
with the highest correlations to the stock as its pairs. For each stock, to calculate the pairs portfolio’s
returns for creating :math:`beta` in the following step, the method uses two different weighting metrics in
calculating the returns.

The first is an equal-weighted portfolio. The method by default computes the pairs portfolio returns as the
equal-weighted average returns of the top n pairs of stocks. The second is a correlation-weighted portfolio.
If this metric is chosen, the method uses the stock’s correlation values to each of the pairs and forms a
portfolio weighted by these values and the weights are calculated by the formula:

.. math::

    w_{k}=\frac{\rho_{k}}{\sum_{i=1}^{n} \rho_{i}}

where :math:`w_{k}` is the weight of stock k in the portfolio and :math:`\rho_{i}` is a correlation of the stock
and one of its pairs.

3. **Calculating beta**

After pairs portfolio returns are calculated, the method derives beta from the monthly return of the stock and
its pairs portfolio. By using linear regression, setting stock return as independent variable and pairs portfolio
return as the dependent variable, the methods set beta as a regression coefficient. Then the beta is stored in a
dictionary for future uses in trading. Below is a figure showing two stocks with high and low beta.

.. figure:: images/distance_approach_pair.png
    :scale: 100 %
    :align: center

Implementation
**************

.. automodule:: arbitragelab.pearson_distance_approach

.. autoclass:: PearsonStrategy
   :members: __init__

.. automethod:: PearsonStrategy.form_portfolio

Trading Signal Generation
#########################

As basic pairs formation confirms declining profitability in pairs trading, some other refined pair
selection criteria have emerged. Here, we describe three different methods from the basic approach in
selecting pairs for trading.

First is only allowing for matching securities within the same industry group . The second is sorting
selected pairs based on the number of zero-crossings in the formation period and the third is sorting
selected pairs based on the historical standard deviation where pairs with high standard deviation are selected.
These selection methods are inspired by the work by Do and Faff (2010, 2012).

1. **Pairs within the same industry group**

In the pairs formation step above, one can add this method when finding pairs in order to match securities
within the same industry group.

With a dictionary containing the name/ticker of the securities and each corresponding industry group,
the securities are first separated into different industry groups. Then, by calculating the Euclidean
square distance for each of the pair within the same group, the :math:`n` closest pairs are selected(in default,
our function also allows skipping a number of first pairs, so one can choose pairs 10-15 to study). This pair
selection criterion can be used as default before adding other methods such as zero-crossings or variance if one
gives a dictionary of industry group as an input.

2. **Pairs with a higher number of zero-crossings**

The number of zero crossings in the formation period has a positive relation to the future spread
convergence according to the work by Do and Faff (2010).

After pairs were matched either within the same industry group or every industry, the top :math:`n` pairs
that had the highest number of zero crossings during the formation period are admitted to the
portfolio we select. This method incorporates the time-series dimension of the historical data in the
form of the number of zero crossings.

3. **Pairs with a higher historical standard deviation**

The historical standard deviation calculated in the formation period can also be a criterion to sort
selected pairs. According to the work of Do and Faff(2010), as having a small SSD decreases the variance
of the spread, this approach could increase the expected profitability of the method.

After pairs were matched, we can sort them based on their historical standard deviation in the formation period
to select top :math:`n` pairs with the highest variance of the spread.


Implementation
**************

.. automethod:: DistanceStrategy.trade_pairs

###

Results output and plotting
###########################

The DistanceStrategy class contains multiple methods to get results in the desired form.

Functions that can be used to get data:

- **get_signals()** outputs generated trading signals for each pair.

- **get_portfolios()** outputs values series of each pair portfolios.

- **get_scaling_parameters()** outputs scaling parameters from the training dataset used to normalize data.

- **get_pairs()** outputs a list of tuples, containing chosen top pairs in the pairs formation step.

- **get_num_crossing()** outputs a list of tuples, containing chosen top pairs with its number of zero-crossings.

Functions that can be used to plot data:

- **plot_pair()** plots normalized price series for elements in a given pair and the corresponding
  trading signals for portfolio of these elements.

- **plot_portfolio()** plots portfolio value for a given pair and the corresponding trading signals.

Implementation
**************

.. automethod:: PearsonStrategy.form_portfolio

.. automethod:: PearsonStrategy.trade_portfolio

.. automethod:: PearsonStrategy.get_trading_signal

.. automethod:: PearsonStrategy.get_short_stocks

.. automethod:: PearsonStrategy.get_long_stocks


Examples
########

Code Example
************

.. code-block::

   # Importing packages
   import pandas as pd
   from arbitragelab.distance_approach.basic_distance_approach import DistanceStrategy

   # Getting the dataframe with price time series for a set of assets
   data = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])

   # Dividing the dataset into two parts - the first one for pairs formation
   data_pairs_formation = data.loc[:'2019-01-01']

   # And the second one for signals generation
   data_signals_generation = data.loc['2019-01-01':]

   # Performing the pairs formation stage of the DistanceStrategy
   # Choosing pairs 5-25 from top pairs to construct portfolios
   strategy = DistanceStrategy()
   strategy.form_pairs(data_pairs_formation, num_top=20, skip_top=5)

   # Adding an industry-based selection criterion to The DistanceStrategy
   strategy_industry = DistanceStrategy()
   strategy_industry.form_pairs(data_pairs_formation, industry_dict=industry_dict,
                                num_top=20, skip_top=5)

   # Using the number of zero-crossing for pair selection after industry-based selection
   strategy_zero_crossing = DistanceStrategy()
   strategy_zero_crossing.form_pairs(data_pairs_formation, method='zero_crossing',
                                     industry_dict=industry_dict, num_top=20, skip_top=5)

   # Checking a list of pairs that were created
   pairs = strategy.get_pairs()

   # Checking a list of pairs with the number of zero crossings
   num_crossing = strategy.get_num_crossing()

   # Now generating signals for formed pairs, using (2 * st. variation) as a threshold
   # to enter a position
   strategy.trade_pairs(data_signals_generation, divergence=2)

   # Checking portfolio values for pairs and generated trading signals
   portfolios = strategy.get_portfolios()
   signals = strategy.get_signals()

   # Plotting price series for elements in the second pair (counting from zero)
   # and corresponding trading signals for the pair portfolio
   figure = strategy.plot_pair(1)

Research Notebooks
******************

The following research notebook can be used to better understand the distance approach described above.

* `Basic Distance Approach`_

* `Basic Distance Approach Comparison`_

.. _`Basic Distance Approach`: https://github.com/Hudson-and-Thames-Clients/arbitrage_research/blob/master/Distance%20Approach/basic_distance_approach.ipynb

.. _`Basic Distance Approach Comparison`: https://github.com/Hudson-and-Thames-Clients/arbitrage_research/blob/master/Distance%20Approach/basic_distance_approach_comparison.ipynb

References
##########

* `Do, B. and Faff, R., 2010. Does simple pairs trading still work?. Financial Analysts Journal, 66(4), pp.83-95. <https://www.jstor.org/stable/pdf/25741293.pdf?casa_token=nIfIcPq13NAAAAAA:Nfrg__C0Q1lcvoBi6Z8DwC_-6pA_cHDdLxxINYg7BPvuq-R5nNzbhVWra2PBL7t2hntj_WBxGH_vCezpp-ZN7NKYhKuZMoX97A7im7PREt7oh2mAew>`_
* `Do, B., and Faff, R. (2012). Are pairs trading profits robust to trading costs? Journal of Financial Research, 35(2):261–287. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1707125>`_
* `Gatev, E., Goetzmann, W.N. and Rouwenhorst, K.G., 2006. Pairs trading: Performance of a relative-value arbitrage rule. The Review of Financial Studies, 19(3), pp.797-827. <https://www.nber.org/system/files/working_papers/w7032/w7032.pdf>`_
* `Krauss, C., 2017. Statistical arbitrage pairs trading strategies: Review and outlook. Journal of Economic Surveys, 31(2), pp.513-545. <https://www.econstor.eu/bitstream/10419/116783/1/833997289.pdf>`_
