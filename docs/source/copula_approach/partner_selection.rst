.. _copula_approach-partner_selection:

.. Note::
    These descriptions closely follow the following papers:

    `Statistical Arbitrage with Vine Copulas. (2016) <https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf>`__ by Stübinger,  Mangold, and Krauss.

    `Multivariate extensions of Spearman’s rho and related statistics. (2007) <https://wisostat.uni-koeln.de/fileadmin/sites/statistik/pdf_publikationen/SchmidSchmidtSpearmansRho.pdf>`__ by Schmid, F., Schmidt, R.

    `A multivariate linear rank test of independence based on a multiparametric copula with cubic sections. (2015) <https://www.statistik.rw.fau.de/files/2016/03/IWQW-10-2015.pdf>`__ by Mangold, B.

========================================
Vine Copula Partner Selection Approaches
========================================

This module contains implementation of the four partner selection approaches mentioned in Section 3.1.1 of Statistical Arbitrage with Vine Copulas.

In this paper[1], Stubinger, Mangold and Krauss developed a multivariate statistical arbitrage strategy based on vine copulas - a highly flexible instrument for linear and nonlinear multivariate dependence modeling. Pairs trading is a relative-value arbitrage strategy, where an investor seeks to profit from mean-reversion properties of the price spread between two co-moving securities. Existing literature focused on using bivariate copulas to model the dependence structure between two stock return time series, and to identify mispricings that can potentially be exploited in a pairs trading application.

This paper proposes a multivariate copula-based statistical arbitrage framework, where specifically, for each stock in the S&P 500 data base, the three most suitable partners are selected by leveraging different selection criteria. Then, the multivariate copula models are benchmarked to capture the dependence structure of the selected quadruples. Later on, the paper focusses on the generation of trading signals and backtesting.

Introduction
############

This module will focus on the various Partner Selection procedures and their implementations, as described in the paper. For every stock in the S&P 500, a partner triple is identified based on adequate measures of association. The following four partner selection approaches are implemented :

- Traditional Approach - baseline approach where the high dimensional relation between the four stocks is approximated by their pairwise bivariate correlations via Spearman’s :math:`\rho`;

- Extended Approach - calculating the multivariate version of Spearman’s :math:`\rho` based on Schmid and Schmidt (2007)[2];

- Geometric Approach - involves calculating the sum of euclidean distances from the 4-dimensional hyper-diagonal;

- Extremal Approach - involves calculating a non-parametric :math:`\chi^2` test statistic based on Mangold (2015)[3] to measure the degree of deviation from independence.

Firstly, all measures of association are calculated using the ranks of the daily discrete returns of our samples. Ranked transformation provides robustness against outliers. Secondly, only the top 50 most highly correlated stocks are taken into consideration for each target stock, to reduce the computational burden.

Traditional Approach
********************

Extended Approach
********************

Geometric Approach
********************

Extremal Approach
********************


Implementation
##############




Research Notebooks
##################

The following research notebook can be used to better understand the partner selection approaches described above.

* `Vine Copula Partner Selection Approaches`_

.. _`Vine Copula Partner Selection Approaches`:

References
##########

* `Stübinger, Johannes; Mangold, Benedikt; Krauss, Christopher; 2016. Statistical Arbitrage with Vine Copulas.  <https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf>`__
* `Schmid, F., Schmidt, R., 2007. Multivariate extensions of Spearman’s rho and related statis-tics. Statistics & Probability Letters 77 (4), 407–416.  <https://wisostat.uni-koeln.de/fileadmin/sites/statistik/pdf_publikationen/SchmidSchmidtSpearmansRho.pdf>`__
* `Mangold, B., 2015. A multivariate linear rank test of independence based on a multipara-metric copula with cubic sections. IWQW Discussion Paper Series, University of Erlangen-N ̈urnberg.  <https://www.statistik.rw.fau.de/files/2016/03/IWQW-10-2015.pdf>`__


