.. _copula_approach-partner_selection:

.. Note::
    These descriptions closely follow the following papers:

    `Statistical Arbitrage with Vine Copulas. (2016) <https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf>`__ by Stübinger,  Mangold, and Krauss.

    `Multivariate extensions of Spearman’s rho and related statistics. (2007) <https://wisostat.uni-koeln.de/fileadmin/sites/statistik/pdf_publikationen/SchmidSchmidtSpearmansRho.pdf>`__ by Schmid, F., Schmidt, R.

    `A multivariate linear rank test of independence based on a multiparametric copula with cubic sections. (2015) <https://www.statistik.rw.fau.de/files/2016/03/IWQW-10-2015.pdf>`__ by Mangold, B.

=============================
Vine Copula Partner Selection
=============================

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                margin-bottom: 5%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://www.youtube.com/embed/qg-idKbPH24"
                frameborder="0"
                allowfullscreen
                style="position: absolute;
                       top: 0;
                       left: 0;
                       width: 100%;
                       height: 100%;">
        </iframe>
        <br/>
    </div>

|

This module contains implementation of the four partner selection approaches
mentioned in Section 3.1.1 of Statistical Arbitrage with Vine Copulas.

In this paper, Stubinger, Mangold and Krauss developed a multivariate statistical arbitrage strategy based on vine copulas -
a highly flexible instrument for linear and nonlinear multivariate dependence modeling. Pairs trading is a relative-value arbitrage strategy,
where an investor seeks to profit from mean-reversion properties of the price spread between two co-moving securities.
Existing literature focused on using bivariate copulas to model the dependence structure between two stock return time series,
and to identify mispricings that can potentially be exploited in a pairs trading application.

This paper proposes a multivariate copula-based statistical arbitrage framework, where specifically,
for each stock in the S&P 500 data base, the three most suitable partners are selected by leveraging different selection criteria.
Then, the multivariate copula models are benchmarked to capture the dependence structure of the selected quadruples.
Later on, the paper focusses on the generation of trading signals and backtesting.

Introduction
############

This module will focus on the various Partner Selection procedures and their implementations, as described in the paper.
For every stock in the S&P 500, a partner triple is identified based on adequate measures of association.
The following four partner selection approaches are implemented:

- Traditional Approach - baseline approach where the high dimensional relation between the four stocks is approximated by their pairwise bivariate correlations via Spearman’s :math:`\rho`;

- Extended Approach - calculating the multivariate version of Spearman’s :math:`\rho` based on Schmid and Schmidt (2007);

- Geometric Approach - involves calculating the sum of euclidean distances from the 4-dimensional hyper-diagonal;

- Extremal Approach - involves calculating a non-parametric :math:`\chi^2` test statistic based on Mangold (2015) to measure the degree of deviation from independence.

Firstly, all measures of association are calculated using the ranks of the daily discrete returns of our samples.
Ranked transformation provides robustness against outliers.

Secondly, only the top 50 most highly correlated stocks are taken into consideration for each target stock, to reduce the computational burden.

The traditional, the extended, and the geometric approach share a common feature - they measure the deviation from linearity in ranks.
All three aim at finding the quadruple that behaves as linearly as possible to ensure that there is an actual relation between its components to model.
While it is true that this aspiration for linearity excludes quadruples with components that are not connected (say, independent),
it also rules out nonlinear dependencies in ranks.
On the other hand, the extremal approach tries to maximize the distance to independence with focus on the joint extreme observations.

.. Note::
    Out of the four approaches, only extremal approach takes into consideration both linear and non-linear dependencies.
    This results in a better preselection and thus better results compared to the other routines.

    So, extremal approach is generally preferred and it should be considered as default for partner selection.

Traditional Approach
####################

As a baseline approach, the high dimensional relation between the four stocks is approximated
by their pairwise bi-variate correlations via Spearman’s :math:`\rho`.
We used ranked returns data for this approach. In addition to the robustness obtained by rank transformation,
it allows to capture non-linearities in the data to a certain degree.

The procedure is as follows:

- Calculate the sum of all pairwise correlations for all possible quadruples, consisting of a fixed target stock.

- Quadruple with the largest sum of pairwise correlations is considered the final quadruple and saved to the output matrix.

Implementation
**************

.. Note::
    This approach takes around 25 ms to run for each target stock.

.. automodule:: arbitragelab.copula_approach

.. autoclass:: PartnerSelection
   :members: __init__

.. automethod:: PartnerSelection.traditional

Extended Approach
#################

Schmid and Schmidt (2007) introduce multivariate rank based measures of association.
This paper generalizes Spearman’s :math:`\rho` to arbitrary dimensions - a natural extension of the traditional approach.

In contrast to the strictly bi-variate case, this extended approach – and the two following approaches –
directly reflect multivariate dependence instead of approximating it by pairwise measures only.
This approach provides a more precise modeling of high dimensional association and thus a better performance in trading strategies.

The procedure is as follows:

- Calculate the multivariate version of Spearman’s :math:`\rho` for all possible quadruples, consisting of a fixed target stock.

- Quadruple with the largest value is considered the final quadruple and saved to the output matrix.


:math:`d` denotes the number of stocks daily returns observed from day :math:`1` to day :math:`n`. :math:`X_i` denotes the :math:`i`-th stock's return.

1. We calculate the empirical cumulative density function (ECDF) :math:`\hat{F}_i` for stock :math:`i`.

2. Calculate quantile data for each :math:`X_{i}`, by

.. math::

    \hat{U}_i = \frac{1}{n} (\text{rank of} \ X_i) = \hat{F}_i(X_i)

The formula for the three estimators are given below, as in the paper.

.. math::

   \hat{\rho}_1 = h(d) \times \Bigg\{-1 + \frac{2^d}{n} \sum_{j=1}^n \prod_{i=1}^d (1 - \hat{U}_{ij}) \Bigg\}

   \hat{\rho}_2 = h(d) \times \Bigg\{-1 + \frac{2^d}{n} \sum_{j=1}^n \prod_{i=1}^d \hat{U}_{ij} \Bigg\}

   \hat{\rho}_3 = -3 + \frac{12}{n {d \choose 2}} \times \sum_{k<l} \sum_{j=1}^n (1-\hat{U}_{kj})(1-\hat{U}_{lj})

Where:

.. math::

   h(d) = \frac{d+1}{2^d - d -1}

We use the mean of the above three estimators as the final measure used to return the final quadruple.

Implementation
**************

.. Note::
    This approach takes around 500 ms to run for each target stock.

.. automethod:: PartnerSelection.extended

Geometric Approach
##################

This approach tries to measure the geometric relation between the stocks in the quadruple.

Consider the relative ranks of a bi-variate random sample, where every observation takes on values in the :math:`[0,1] \times [0,1]` square.
If there exists a perfect linear relation among both the ranks of the components of the sample,
a plot of the relative ranks would result in a perfect line of dots between the points (0,0) and (1,1) – the diagonal line.
However, if this relation is not perfectly linear, at least one point differs from the diagonal.
The Euclidean distance of all ranks from the diagonal can be used as a measure of deviation from linearity, the diagonal measure.

Hence, we try to find the quadruple :math:`Q` that leads to the minimal value of the sum of these Euclidean distances.

The procedure is as follows:

- Calculate the four dimensional diagonal measure for all possible quadruples, consisting of a fixed target stock.

- Quadruple with the smallest diagonal measure is considered the final quadruple and saved to the output matrix.


The diagonal measure in four dimensional space is calculated using the following equation,

.. math::
    \sum_{i=1}^{n} | (P - P_{1}) - \frac{(P - P_{1}) \cdot (P_{2} - P_{1})}{| P_{2} -P_{1} |^{2}} (P_{2} - P_{1}) |

where,

.. math::
    P_{1} = (0,0,0,0)

.. math::
    P_{2} = (1,1,1,1)

are points on the hyper-diagonal, and

.. math::
    P = (u_{1},u_{2},u_{3},u_{4})

where :math:`u_i` represents the ranked returns of a stock :math:`i` in quadruple.



Implementation
**************

.. Note::
    This approach takes around 180 ms to run for each target stock.

.. automethod:: PartnerSelection.geometric

Extremal Approach
#################

Mangold (2015) proposes a nonparametric test for multivariate independence.
The resulting :math:`\chi^2` test statistic can be used to measure the degree of deviation from independence, so dependence.
The value of the measure increases on the occurence of an abnormal number of joint extreme events.


The procedure is as follows:

- Calculate the :math:`\chi^2` test statistic for all possible quadruples, consisting of a fixed target stock.

- Quadruple with the largest test statistic is considered the final quadruple and saved to the output matrix.

Given below are the steps to calculate the :math:`\chi^2` test statistic described in Proposition 3.3 of Mangold (2015):

.. Note::
    These steps assume a 4-dimensional input.

1) Analytically calculate the 4-dimensional Nelsen copula from Definition 2.4 in Mangold (2015):

.. math::
    C_{\theta}(u_{1}, u_{2}, u_{3}, u_{4}) = u_1u_2u_3u_4 \times (1 + ((1- u_{1})(1- u_{2})(1- u_{3})(1- u_{4})) *
.. math::
    (\theta_{1} ((1- u_{1})(1- u_{2})(1- u_{3})(1- u_{4})) +
    \theta_{2} ((1- u_{1})(1- u_{2})(1- u_{3})u_{4}) +
.. math::
    \theta_{3} ((1- u_{1})(1- u_{2})u_{3}(1- u_{4})) +
    \theta_{4} ((1- u_{1})(1- u_{2})u_{3}u_{4}) +
.. math::
    \theta_{5} ((1- u_{1})u_{2}(1- u_{3})(1- u_{4})) +
    \theta_{6} ((1- u_{1})u_{2}(1- u_{3})u_{4}) +
.. math::
    \theta_{7} ((1- u_{1})u_{2}u_{3}(1- u_{4})) +
    \theta_{8} ((1- u_{1})u_{2}u_{3}u_{4}) +
.. math::
    \theta_{9} (u_{1}(1- u_{2})(1- u_{3})(1- u_{4})) +
    \theta_{10} (u_{1}(1- u_{2})(1- u_{3})u_{4}) +
.. math::
    \theta_{11} (u_{1}(1- u_{2})u_{3}(1- u_{4})) +
    \theta_{12} (u_{1}(1- u_{2})u_{3}u_{4}) +
.. math::
    \theta_{13} (u_{1}u_{2}(1- u_{3})(1- u_{4})) +
    \theta_{14} (u_{1}u_{2}(1- u_{3})u_{4}) +
.. math::
    \theta_{15} (u_{1}(1- u_{2})u_{3}(1- u_{4})) +
    \theta_{16} (u_{1}u_{2}u_{3}u_{4})
    )


2) Analytically calculate the corresponding density function of 4-dimensional copula:

.. math::
    c_{\theta}(u_{1}, u_{2}, u_{3}, u_{4}) = \frac{\partial^{4}}{\partial u_{1} \partial u_{2} \partial u_{3}\partial u_{4}}C_{\theta}(u_{1}, u_{2}, u_{3}, u_{4})

.. Note::
    To calculate the density function, we can notice a pattern in the copula equation.
    The form of each :math:`u_i` beside :math:`\theta_i` is either

    .. math::
        u_{i}(1- u_{i})^2 \quad  or \quad  u_{i}^2(1 - u_{i})

    and the corresponding partial derivatives of these two forms are,

    .. math::
        (u_{i} - 1)(3u_{i} - 1) \quad or \quad  u_{i}(2 - 3u_{i})

    This observation simplifies the analytical calculation of the density function.


3) Calculate the Partial Derivative of above density function :math:`w.r.t \ \theta`.

.. math::
    \dot{c_{\theta}} = \frac{\partial c_{\theta}(u_{1}, u_{2}, u_{3}, u_{4})}{\partial \theta}


4) Calculate the Test Statistic for p-dimensional rank test:

.. math::
    T=n \boldsymbol{T}_{p, n}^{\prime} \Sigma\left(\dot{c}_{\theta_{0}}\right)^{-1} \boldsymbol{T}_{p, n} \stackrel{a}{\sim} \chi^{2}(q)

where,

.. math::
    \boldsymbol{T}_{p, n}=\mathbb{E}\left[\left.\frac{\partial}{\partial \theta} \log c_{\theta}(B)\right|_{\theta=\theta_{0}}\right]

.. math::
    \Sigma\left(\dot{c}_{0}\right)_{i, j}=\int_{[0,1]^{p}}\left(\left.\frac{\partial c_{\theta}(\boldsymbol{u})}
    {\partial \theta_{i}}\right| _{\boldsymbol{\theta}=\mathbf{0}}\right) \times\left(\left.\frac{\partial c_{\theta}(\boldsymbol{u})}
    {\partial \theta_{j}}\right |_{\theta=0}\right) \mathrm{d} \boldsymbol{u}





Implementation
**************

.. Note::
    This approach is extremely heavy compared to other approaches and takes around 15 sec to run for each target stock.

    Please be aware that there is a big overhead at the start of this method which involves calculating the covariance matrix.
    This should take around 1 to 2 minutes when d = 4 which is the default. Increasing the value of d will increase the processing time significantly.

.. automethod:: PartnerSelection.extremal

Code Example
############

.. code-block::

    # Importing the module and other libraries
    from arbitragelab.copula_approach.vine_copula_partner_selection import PartnerSelection
    import pandas as pd

    # Importing DataFrame of daily pricing data for all stocks in S&P 500.(at least 12 months data)
    df = pd.read_csv(DATA_PATH, parse_dates=True, index_col='Date').dropna()

    # Instantiating the partner selection module.
    ps = PartnerSelection(df)

    # Calculating final quadruples using traditional approach for first 20 target stocks.
    Q = ps.traditional(20)
    print(Q)
    # Plotting the final quadruples.
    ps.plot_selected_pairs(Q)

    # Calculating final quadruples using extended approach for first 20 target stocks.
    Q = ps.extended(20)
    print(Q)
    # Plotting the final quadruples.
    ps.plot_selected_pairs(Q)

    # Calculating final quadruples using geometric approach for first 20 target stocks.
    Q = ps.geometric(20)
    print(Q)
    # Plotting the final quadruples.
    ps.plot_selected_pairs(Q)

    # Calculating final quadruples using extremal approach for first 20 target stocks.
    Q = ps.extremal(20)
    print(Q)
    # Plotting the final quadruples.
    ps.plot_selected_pairs(Q)


Research Notebooks
##################

The following research notebook can be used to better understand the partner selection approaches described above.

* `Vine Copula Partner Selection Approaches`_

.. _`Vine Copula Partner Selection Approaches`: https://hudsonthames.org/notebooks/arblab/Vine_copula_partner_selection.html

.. raw:: html

    <a href="https://hudthames.tech/3xsu5ws"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
    <a href="https://hudthames.tech/2S03R58"><button style="margin: 20px; margin-top: 0px">Download Sample Data</button></a>

Research Article
################

.. raw:: html

    <style>
      .special {
        display: inline-block;
        background-color: #0399AB;
        color: #eeeeee;
        text-align: center;
        font-size: 180%;
        padding: 15px;
        width: 100%;
        transition: all 0.5s;
        cursor: pointer;
        font-family: 'Josefin Sans';
      }
      .special span {
        cursor: pointer;
        display: inline-block;
        position: relative;
        transition: 0.5s;
      }
      .special span:after {
        content: '\00bb';
        position: absolute;
        opacity: 0;
        top: 0;
        right: -20px;
        transition: 0.5s;
      }
      .special:hover {
        background-color: #e7f2fa;
        color: #000000;
      }
      .special:hover span {
        padding-right: 25px;
      }
      .special:hover span:after {
        opacity: 1;
        right: 0;
      }
    </style>

    <button class="special" onclick="window.open('https://hudsonthames.org/copula-for-statistical-arbitrage-stocks-selection-methods/','_blank')">
      <span>Read our article on the topic</span>
    </button>

|

Presentation Slides
###################

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQPOwlFOZ19H24WITCqQLTM3Pl9BQ1-wS0PZvSOOKdCTEW2tzBan_ca30yZlhsf-96Su80-aVkcGEAZ/embed?start=false&loop=false&delayms=3000"
                frameborder="0"
                allowfullscreen
                style="position: absolute;
                       top: 0;
                       left: 0;
                       width: 100%;
                       height: 100%;">
        </iframe>
    </div>

|

References
##########

* `Stübinger, Johannes; Mangold, Benedikt; Krauss, Christopher; 2016. Statistical Arbitrage with Vine Copulas.  <https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf>`__
* `Schmid, F., Schmidt, R., 2007. Multivariate extensions of Spearman’s rho and related statis-tics. Statistics & Probability Letters 77 (4), 407–416.  <https://wisostat.uni-koeln.de/fileadmin/sites/statistik/pdf_publikationen/SchmidSchmidtSpearmansRho.pdf>`__
* `Mangold, B., 2015. A multivariate linear rank test of independence based on a multipara-metric copula with cubic sections. IWQW Discussion Paper Series, University of Erlangen-N ̈urnberg.  <https://www.statistik.rw.fau.de/files/2016/03/IWQW-10-2015.pdf>`__
