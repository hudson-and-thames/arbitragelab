.. _copula_approach-copula_brief_intro:


.. note::
   This document was greatly inspired by:

    1. Nelsen, Roger B. `An introduction to copulas <https://www.springer.com/gp/book/9780387286594>`__.
       Springer Science & Business Media, 2007.
    2. Nelsen, Roger B. `"Properties and applications of copulas: A brief survey." <http://w4.stern.nyu.edu/ioms/docs/sg/seminars/nelsen.pdf>`__
       Proceedings of the first brazilian conference on statistical modeling in insurance and finance.
       University Press USP Sao Paulo, 2003.
    3. Chang, Bo. `Copula: A Very Short Introduction <https://bochang.me/blog/posts/copula/>`__.
    4. Wiecki, Thomas. `An intuitive, visual guide to copulas <https://twiecki.io/blog/2018/05/03/copulas/>`__.

==================================
A Practical Introduction to Copula
==================================

Instead of starting with a formal definition of copula with mathematical rigor, let's start by doing the following
calculations to try to temporarily bypass it. The mathematical definition may not help much unless we have an idea
of where the copula comes from.

Suppose we have two random variables :math:`S_1`, :math:`S_2 \in [-\infty, \infty]` in a probability space
:math:`(\Omega, \mathbb{P}, \mathcal{F})`.
:math:`S_1` and :math:`S_2` have their own fixed, continuous distributions. Now suppose we know those two distributions and
their CDFs (cumulative density functions) :math:`F_1, F_2`. For each value pair :math:`(S_1(\omega), S_2(\omega))`,
:math:`\omega \in \Omega`, we can
calculate each :math:`S_i`'s quantile :math:`u_1(\omega), u_2(\omega)` in :math:`[0, 1]` from their CDFs.
Then we plot the random variable pair's quantile data :math:`(u_1, u_2)` on a :math:`[0, 1] \times [0, 1]` canvas.
Now, we just made a density (sample) plot for a copula, uniquely defined by the two random variables.
(For interested reader, the guaranteed justification of existence and uniqueness is Sklar's Theorem.)

Moreover, one should realize that the copula although depends on the CDFs, but more fundamentally it was characterized
by how the two random variables are "related". One key convenience of using copula is to separate the discussion of
the pair's own arbitrary marginal distributions from quantifying how they are "related". Since the quantile pair
:math:`u_1`, :math:`u_2` has to be uniform in :math:`[0, 1]`, we are able to study a lot more combinations of random
variable pairs under the copula framework, instead of spending efforts in idiosyncratic features of their marginal
distributions.

The process of sampling:
    1. For two random variables with fixed distributions, find their own marginal CDFs. (The CDFs need to be continuous.)
    2. Pick a specific realization :math:`\omega \in \Omega`, and from it sample the two random variables.
    3. Sample multiple times and map the random samples into their quantile domain by their CDFs.
    4. Plot the pair's quantile data.

Definition of Bivariate Copula
##############################

(**Definition using Sklar's Theorem**) For two random variables :math:`S_1`, :math:`S_2 \in [-\infty, \infty]`.
:math:`S_1` and :math:`S_2` have their own fixed, continuous CDFs :math:`F_1, F_2`.
Consider their (cumulative) joint distribution :math:`H(s_1, s_2) := P(S_1 \le s_1, S_2 \le s_2)`.
Now take the uniformly distributed quantile random variable :math:`U_1(S_1)`, :math:`U_2(S_2)`, for every pair
:math:`(u_1, u_2)` drawn from the pair's quantile we define the **bivariate copula**
:math:`C: [0, 1] \times [0, 1] \rightarrow [0, 1]` as:

.. math::

    \begin{align}
    C(u_1, u_2) &= P(U_1 \le u_1, U_2 \le u_2) \\
    &= P(S_1 \le F_1^{-1}(u_1), S_2 \le F_2^{-1}(u_2)) \\
    &= H(F_1^{-1}(u_1), F_2^{-1}(u_2))
    \end{align}

where :math:`F_1^{-1}` and :math:`F_2^{-1}` are quasi-inverse of the marginal CDFs :math:`F_1` and :math:`F_2`.

Moreover, we define the **cumulative conditional probabilities**:

.. math::
    \begin{align}
    P(U_1\le u_1 | U_2 = u_2) &:= \frac{\partial C(u_1, u_2)}{\partial u_2}, \\
    P(U_2\le u_2 | U_1 = u_1) &:= \frac{\partial C(u_1, u_2)}{\partial u_1},
    \end{align}

and the **copula density** :math:`c(u_1, u_2)`:

.. math::
    c(u_1 , u_2) := \frac{\partial^2 C(u_1, u_2)}{\partial u_1 \partial u_2},

which by definition is the probability density.

.. Note::

    :math:`C(u_1, u_2)` is the copula's definition, and almost all mathematical analysis stems from here,
    however it is rarely used, if at all, for trading purposes. In sampling and maximum likelihood estimation, we use
    the copula density :math:`c(u_1, u_2)`, and in trading signal formation by (Liew et al., 2013), they use conditional
    cumulative probabilities :math:`P(U_1\le u_1 | U_2 = u_2)` and :math:`P(U_2 \le u_2 | U_1 = u_1)`.

Copula Types and Generators
###########################

The most commonly used bivariate types are
**Gumbel**, **Frank**, **Clayton**, **Joe**, **N13**, **N14**, **Gaussian**, and **Student-t**.
All of those except for Gaussian and Student-t copulas fall into the category of **Archimedean copulas**.
A bivariate copula :math:`C` is called Archimedean if it can be represented as:

.. math::

    C(u_1, u_2; \theta) = \phi^{[-1]}(\phi(u_1; \theta), \phi(u_2; \theta))

where :math:`\phi: [0,1] \times \Theta \rightarrow [0, + \infty)` is called the generator for the copula,
:math:`\phi^{[-1]}` is its pseudo-inverse. The generators' formulae are available from standard literature.
Loosely speaking, :math:`\theta` is the parameter that measures how "closely" the two random variables
are related, and its exact range and interpretation are different across different Archimedean copulas.

The takeaway here is that, in general, arbitrary copulae are quite difficult to examine, whereas Archimedean
copulas enable further analysis by having the nice structure above.
Two of the most important features of the Archimedean copula are its symmetry and scalability to multiple dimensions,
although a closed-form solution may not be available in higher dimensions.
As a result, one uses a generator to define an Archimedean copula.

For the Gaussian and Student-t copula, the concepts are much easier to follow:
Suppose for a correlation matrix :math:`R \in [-1, 1]^{d \times d}`, the multi-variate Gaussian copula with
parameter matrix :math:`R` is defined as:

.. math::
    C_R(\mathbf{u}) := \Phi_R(\Phi^{-1}(u_1),\dots, \Phi^{-1}(u_d))

where :math:`\Phi_R` is the joint Gaussian CDF with :math:`R` being its covariance matrix,
:math:`\Phi^{-1}` is the inverse of the CDF of a standard normal.

The Student-t copula can be defined in a similar way, with :math:`\nu` being the degrees of freedom:

.. math::
    C_{R,\nu}(\mathbf{u}) := \Phi_{R,\nu}(\Phi_{\nu}^{-1}(u_1),\dots, \Phi_{\nu}^{-1}(u_d))

Gaussian and Student-t copulas belong to a family called **Elliptical copula**.

Generators for the Archimedean copulas included in the package
    - Gumbel: :math:`\phi(t; \theta) = (- \ln t)^\theta`, :math:`\theta \in [1, +\infty)`
    - Frank: :math:`\phi(t; \theta) = - \ln \left(\frac{e^{-\theta t}-1}{e^{-\theta}-1} \right)`, :math:`\theta \in [-\infty, \infty)\backslash\{0\}`
    - Clayton: :math:`\phi(t; \theta) = \frac{t^{-\theta}-1}{\theta}`, :math:`\theta \in [-1, +\infty)\backslash\{0\}`
    - Joe: :math:`\phi(t; \theta) = -\ln(1-(1-t)^{\theta})`, :math:`\theta \in [1, +\infty)`
    - N13: :math:`\phi(t; \theta) = (1- \ln t)^\theta - 1`, :math:`\theta \in [0, +\infty)`
    - N14: :math:`\phi(t; \theta) = (t^{-1/\theta}- 1)^\theta`, :math:`\theta \in [1, +\infty)`

Densities and Marginal Probabilities
************************************

It is often impractical to calculate a copula's density and marginal probabilities by definition if
a closed-form solution for :math:`C(u_1,u_2)` is not available.
Luckily, for Archimedean copulas, one can use the definition and often find closed-form solutions for
:math:`c(u_1, u_2)` and conditional probabilities.

For elliptical copulas, all of those quantities can be derived from their definitions. Although one may not have a
closed-form solution, it is indeed very quick and accurate to calculate numerically.

Below are densities and conditional probabilities for the bivariate Gaussian and Student-t copula:

    - Gaussian:

    .. math::
        P(U_1 \le u_1 \mid U_2 = u_2) =
        \Phi\left(\frac{\Phi^{-1}(u_1) - \rho \Phi^{-1}(u_2)}{\sqrt{1 - \rho^2}} \right)

    .. math::
        c(u_1, u_2) = \frac{1}{\sqrt{1-\rho^2}}
        \exp \left[ \frac{
        \rho(-2\Phi^{-1}(u_1) \Phi^{-1}(u_2) + (\Phi^{-1}(u_1))^2 \rho + (\Phi^{-1}(u_2))^2 \rho)}
        {2(\rho^2 - 1)} \right]

    - Student-t:

    .. math::
        P(U_1 \le u_1 \mid U_2 = u_2) =
        \Phi_{\nu + 1}\left(
        (\Phi_{\nu}^{-1}(u_1) - \rho \Phi_{\nu}^{-1}(u_2))
        \sqrt{\frac{\nu + 1}{(\nu + \Phi_{\nu}^{-1}(u_2))(1-\rho^2)}}
        \right)

    .. math::
        c(u_1, u_2) = 
        \frac{f_{R,\nu}(\Phi_{\nu}^{-1}(u_1), \Phi_{\nu}^{-1}(u_2))}
        {f_{\nu}(\Phi_{\nu}^{-1}(u_1)) f_{\nu}(\Phi_{\nu}^{-1}(u_2))}

    where :math:`f_{R, \nu}` is the PDF for bivariate Student-t distribution with degrees of
    freedom :math:`\nu` and covariance matrix being the correlation matrix :math:`R`, and
    :math:`f_{\nu}` is the univariate Student-t PDF. :math:`\rho \in [-1, 1]` is the correlation parameter.
	
Notice that all bivariate Archimedean copulas and Gaussian copula have only one parameter :math:`\theta`
or :math:`\rho` to be uniquely determined (and thus to be estimated from data),
whereas Student-t copula has two parameters :math:`\rho` and :math:`\nu` to be determined.
Estimation of :math:`\nu` from stock's time series is still an open topic, and this module uses maximum likelihood
to choose :math:`\nu`.

.. Note::
    Using :math:`\nu = (` sample size :math:`- 1 )` for correlated time series data is strongly discouraged, since each
    data point is not independent from others. Also one should keep :math:`\nu` to be reasonably small
    so that it makes sense to use Student-t to model.
    In general, for :math:`\nu > 12`, especially when there is obviously no tail dependency from data, one should use the
    Gaussian copula instead.


Sample Generation from a Copula
*******************************

We sample from a given copula according to its density :math:`c(u_1, u_2)`. The sample can be used, for example,
to visually justify the fit with actual data.
Further, one can draw a sample from a given copula, and use the inverse of marginal CDFs to simulate future data.

For Archimedean copulas, the general methodology for sampling or simulation comes from (Nelsen, 2006):

	1. Generate two uniform in :math:`[0, 1]` i.i.d.'s :math:`(v_1, v_2)`.
	2. Calculate :math:`w = K_c^{-1}(v_2)`, :math:`K_c(t) = t - \frac{\phi(t)}{\phi'(t)}`.
	3. Calculate :math:`u_1 = \phi^{-1}[v_1 \phi(w)]` and :math:`u_2 = \phi^{-1}[(1-v_1) \phi(w)]`.
	4. Return :math:`(u_1, u_2)`.

For some copulas, the above method can greatly be simplified due to having closed-form solutions for step :math:`2`.
Otherwise, one will have to use appropriate numerical methods to find :math:`w`.
Interested readers can check `Procedure to Generate Uniform Random Variates from Each Copula
<https://www.caee.utexas.edu/prof/bhat/ABSTRACTS/Supp_material.pdf>`_
for all the simplified forms.

For Gaussian and Student-t copulas, one can follow the procedures below:

	1. Generate two a pair :math:`(v_1, v_2)` using a bivariate Gaussian/Student-t distribution with desired 
	   correlation (and degrees of freedom).

	2. Transform those into quantiles using CDF :math:`\Phi` from standard Gaussian or Student-t distribution (with
	   desired degrees of freedom). i.e., :math:`u_1 = \Phi(v_1)`, :math:`u_2 = \Phi(v_2)`.

	3. Return :math:`(u_1, u_2)`.


Pseudo-Maximum Likelihood Fit to Data
#####################################

Suppose we have a pair of stocks' price time series data, and they *are known to be correlated to start with*.
To be able to use the copula method, to its root there are three fundamental questions to answer:

	1. What data do we use to fit.
	2. Which copula to use.
	3. What is(are) the parameter(s) for this copula.

Data transform
**************

One may use the implied **cumulative log return** (Liew et al., 2013) or **log return** (Stander et al., 2013) instead
of the raw prices but the fitted copula will be identical.
Because copula is invariant under any strictly monotone mappings for its marginal random variables.

.. Note::
    One key concern is that, the type of processed data fed in needs to be **approximately stationary**.
    i.e., :math:`\mathbb{E}[X(t_1)] \approx \mathbb{E}[X(t_2)]` for time series :math:`X`, for all :math:`t_1, t_2` in
    the scope of interest.
    For example, if we model each stock's price to have a log-Normal distribution, then the price itself cannot be stationary
    after some time.
    One can consider just using the daily return or its logarithm instead, given that the stock's price has a log-Normal 
    distribution. i.e., :math:`\frac{X(t+1)}{X(t)}` or :math:`\ln \left( \frac{X(t+1)}{X(t)} \right)`.

Choice of Copula
****************

There is no rule of thumb in regard to choosing a certain copula. However, there are some empirical guidelines to follow.
One may likely consider the **tail dependency** significant, as large correlated moves in prices need to be accounted for.
In such case, Gumbel is a good choice.

Realistically when using the module, one can fit the data to every copula and compare the score (in SIC, AIC, HQIC,
log-likelihood) to find the appropriate copula since the calculations are quick.
However, such approach should always be proceeded with caution, as certain important characteristics of the stocks pair
might have been neglected.

Determine Parameter(s)
**********************

For all Archimedean copulas in this module, we follow a two-step pseudo-MLE approach as below:

	1. Use Empirical CDF (ECDF) to map each marginal data to its quantile.
	2. Calculate Kendall's :math:`\hat\tau` for the quantile data, and use Kendall's :math:`\hat\tau` to calculate :math:`\hat\theta`.

.. Tip::
    The :code:`construct_ecdf_lin` function we provide in the :code:`copula_calculation` module is a wrapper around :code:`ECDF`
    from :code:`statsmodels.distributions.empirical_distribution`
    `[Link] <https://www.statsmodels.org/stable/generated/statsmodels.distributions.empirical_distribution.ECDF.html>`__
    that allows linear interpolations instead of using a step function.
    Also it will not hit :math:`0` or :math:`1` but stays sufficiently close to avoid numerical issues in calculations.

.. Note::
	For Archimedean copula, :math:`\tau` and :math:`\theta` are implicitly related via
	
	.. math::
		\tau(\theta) = 1 + 4 \int_0^1 \frac{\phi(t;\theta)}{\phi'(t;\theta)} dt
	
	Then one inversely solves :math:`\hat\theta(\hat\tau)`. For some copulas, the inversion has a closed-form solution. For
	others, one has to use numerical methods.

For elliptical copulas, we calculate the Kendall's :math:`\hat{\tau}` and then find :math:`\hat{\rho}` via

.. math::
		\hat{\rho} = \sin \left( \frac{\hat{\tau} \pi}{2} \right)

for the covariance matrix :math:`\mathbf{\sigma}_{2 \times 2}` (though technically speaking, for bivariate
copulas, only correlation :math:`\rho` is needed, and thus it is uniquely determined) from the quantile data,
then use :math:`\mathbf{\sigma}_{2 \times 2}` for a Gaussian or Student-t copula.
Fitting by Spearman's :math:`\rho` for the variance-covariance matrix from data for elliptic copulas is also practiced
by some.
But Spearman's :math:`\rho` is in general less stable than Kendall's :math:`\tau` (though with faster calculation speed).
And using var-covar implicitly assumes a multi-variate Gaussian model, and it is sensitive to outliers because it is a
parametric fit.
See `An Introduction to Copulas <http://www.columbia.edu/~mh2078/QRM/Copulas.pdf>`__ for more detail.

Also note that, theoretically speaking, for Student-t copula, Determining :math:`\nu` (degrees of freedom) analytically from
an arbitrary time series is still an open problem.
Therefore we opted to use a maximum likelihood fit for :math:`\nu` for the family of Student-t copulas initiated by
:math:`\mathbf{\sigma}_{2 \times 2}`.
This calculation is relatively slow.

Fitting mixed copula is a process that is a bit more complicated and is discussed in the
separate documentation: :ref:`A Deeper Intro to Copulas <copula_approach-copula_deeper_intro>`.
Here are a few takeaways:

- Generic max likelihood fit is not stable, and does not drive small weights to 0.

- We opt for an expectation-maximization(EM) algorithm, which greatly increases the stability, and generally converges to
  a much better result than a generic max likelihood algorithm.
  
- Any mixture with Student-t copula will greatly decrease the speed for fitting.

- Mixed copulas generally give the best result in terms of max likelihood across all copulas we provide.

Research Notebooks
##################

The following research notebook can be used to better understand the basic copula strategy.

* `Basic Copula Strategy`_

.. _`Basic Copula Strategy`: https://github.com/Hudson-and-Thames-Clients/arbitrage_research/blob/master/Copula%20Approach/Copula_Strategy_Basic.ipynb

References
##########

* `Liew, R.Q. and Wu, Y., 2013. Pairs trading: A copula approach. Journal of Derivatives & Hedge Funds, 19(1), pp.12-30. <https://link.springer.com/article/10.1057/jdhf.2013.1>`__
* `Stander, Y., Marais, D. and Botha, I., 2013. Trading strategies with copulas. Journal of Economic and Financial Sciences, 6(1), pp.83-107. <https://www.researchgate.net/publication/318054326_Trading_strategies_with_copulas>`__
* `Schmid, F., Schmidt, R., Blumentritt, T., Gaißer, S. and Ruppert, M., 2010. Copula-based measures of multivariate association. In Copula theory and its applications (pp. 209-236). Springer, Berlin, Heidelberg. <https://www.researchgate.net/publication/225898324_Copula-Based_Measures_of_Multivariate_Association>`__
