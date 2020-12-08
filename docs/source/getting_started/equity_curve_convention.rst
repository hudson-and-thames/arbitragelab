
=======================
Equity Curve Convention
=======================

Why We Write This
#################

For every strategy, in the end we wish to see its performance on portfolios comprised of real world data.
However, there are very little well-organized resourses that are openly and readily available.
Often it is not clear from the litertuares on their methodologies and conventions about the calculation.
For example, if a pair's trading strategy suggests a position like "long the spread", what is the hedging ratio? How many
units to purchase? Is the spread in terms of price or log price? What is the return at the end?
Those are very important questions for real world applications.

Therefore, we looked carefully through the logic behind calculating the equity curve based on a portfolio with
different common assumptions.
Moreover, in order to stay logically sound, the convention described in this document is used in research notebooks
for :code:`arbitragelab`, in stead of evaluating every strategy's equity curve to returns.

There are some commonly sound approaches with regard to, loosely speaking, "calculating performance of the strategy on a
portfolio".
Obviously what one calculates depends on what the definition of "performance" is.
Listed below are some generally used conventions:

1. The return. Usually used for stocks.
2. Profit for unit position. Usually used for long-short strategies, e.g. pairs trading.
3. Returns on committed capital (ROCC).

Forming A Portfolio
###################
Before diving into calculating equity curves, we need to discuss how to form a reasonable portfolio.
A rule of thumb is that, the portfolio denpends on the strategy that one aims to implement, and a good portfolio should
hedge away unnecessary exposures to various kinds of risks, while emphasizing the real strength of the strategy.
Thus, there is no universally best way to form portfolios, as a terrible choice for one strategy may be almost optimal for
another.

To keep the discussion manageable and to the point, we limit our cases to a long-short pairs trading framework with
stocks underlying.
The ideas can be generalized easily to a multi-asset portfolio, which we may cover in detail in the future.

Suppose we have two stocks price series :math:`S_A`, :math:`S_B`.
In the portfolio we have :math:`X_A` units of :math:`A`, :math:`X_B` units of :math:`B`.
:math:`X_A, X_B` may or may not change with time.

.. Note::
    Often :math:`X_A` normalized to 1, but this is not a hard-and-fast rule.

Thus the portfolio price is

.. math::
    \Pi(t) = X_A S_A(t) + X_B S_B(t)

with revenue :math:`R` (**NOT** the return :math:`r(t) = \frac{S(t)}{S(0)} - 1`)

.. math::
    \begin{align}
    R(t) &= \Pi(t) - \Pi(0) \\ 
    &= X_A (S_A(t)-S_A(0)) + X_B (S_B(t)-S_B(0)) \\
    &= X_A (R_A(t)) + X_B (R_B(t))
    \end{align}

Below we discuss typical methods and their potential drawbacks from different typical approaches.

Benchmark Methods
*****************
Those methods are widely used, and are generally considered benchmarks for other methods to compare with.
A good example would be *Ordinary Least Square (OLS)*.
One can simply run an OLS on test data for two stocks' price time series :math:`S_A` and :math:`S_B` to get the hedge ratio.

The detail and justification is written below:

OLS is under the assumption that :math:`R_A` and :math:`R_B` are jointly normally distributed.
i.e., :math:`S_A` and :math:`S_B` follow simple (correlated) random walks.
We aim to find :math:`h` that minimizes the variance of the portfolio

.. math::
    \Pi_h = S_A - h S_B.

In this case,

.. math::
    \begin{align}
    Var(R_h) &= \mathbb{E}[R_A^2 - 2h R_A R_B + h^2 R_B^2] - 0\\
    &= \sigma_A^2 - 2h \sigma_{AB} + \sigma_{B}^2 h^2
    \end{align}

achieving min when

.. math::
    h = \frac{\sigma_{AB}}{\sigma_B^2}

In terms of implementation, one can calculate using the above formula, or note that

.. math::
    R_A = h R_B + R_h

where :math:`R_h` is normally distributed, independent from :math:`R_A`.
Thus it is the same as running an OLS on :math:`R_B` against :math:`R_A`, or equivalently running an OLS on
:math:`S_B` against :math:`S_A`:

.. math::
    S_A = h S_B + (R_h + S_A(0) - S_B(0))

.. Note::
    OLS may face the following possible issues:
    
    1. Modeling stock prices as random walks may be situational;
    2. The interdependencies between stock prices may not be modeled as a bivariate Gaussian, 
       especially when they are known to bear strong tail dependencies;
    3. OLS does not consider :math:`S_A` and :math:`S_B` as time series, but random variables.
       One can re-shuffle the order of :math:`(S_A(t), S_B(t))` in the training data and still get the same result, which 
       makes OLS subject to criticism for a potential loss of information.

Cointegration Methods
*********************

Dollar Neutral Portfolio
************************

Calculating Equity Curve
########################

Stocks and Index Funds
**********************
If the traded portfolio's value is **strictly positive**, for example, common stocks and index funds, then one can take
advantage of it to be able to calculate **the return** as follows:

1. Construct the portfolio (price or value series) :math:`\Pi(t)` with some hedge ratio.

    .. math::
        \Pi(t) = S_A(t) - h S_B(t)

2. Get portfolio's daily returns series.
   This step breaks down if the series is not strictly positive.

    .. math::
        r(t) = \frac{\Pi(t)}{\Pi(t-1)} - 1

    .. Warning::
        Returns of the portfolio is **NOT** the linear combination of returns from each component:
    
        .. math::
            r(t) = \frac{S_A(t) - h S_B(t)}{S_A(t-1) - h S_B(t-1)} - 1 \neq r_A(t) - h r_B(t) - 1

3. Get the positions :math:`P(t)` from some strategy.
4. Calculatethe daily returns :math:`r_s(t)` from our *strategy*. It is the pointwise multiplication

    .. math::
        r_s(t) = r(t)P(t), \ \text{for each} \ t

5. Then we use daily returns :math:`r_s(t)` to reconstruct our portfolio's **equity curve in return**:

    .. math::
        \mathcal{E}(t) = \left( \prod_{\tau=0}^t [r_s(\tau) + 1] \right) - 1

Notice the result series :math:`\mathcal{E}(t)` is constructed purely from the returns, and it holds no information about 
the portfolio's value.
Therefore the result is the return series from the strategy: :math:`0.2` means :math:`20` dollar profit for :math:`100`
dollar initial investment.

Spread for a Stock Pair
***********************
In this case we have to back up and derive everything by definition from daily P&L.
It makes no sense to even use traditional measures like return on capital.
For example, if a strategy tells you to long the spread when the spread is :math:`0`, and you have :math:`100,000` dollar
capital to invest, how many units can you buy?
(Hint: Infinity is not the answer.)
Moreover, because the spread can be positive or negative, it makes no sense to use returns for calculation as well.
Therefore, we calculate **the cumulative P&L for 1 unit of spread**.

1. Construct the portfolio (price series) :math:`\Pi(t)` with some hedge ratio.
 
    .. math::
        \Pi(t) = S_A(t) - h S_B(t)

2. Get portfolio's daily revenue (price difference, daily P&L for one unit) series.

    .. math::
        R(t) = \Pi(t) - \Pi(t-1)

3. Get the positions :math:`P(t)` from some strategy.
4. Calculatethe daily returns :math:`R_s(t)` from our *strategy*. It is the pointwise multiplication

    .. math::
        R_s(t) = R(t)P(t), \ \text{for each} \ t

5. Then we use daily P&L :math:`R_s(t)` to reconstruct our portfolio's **equity curve in cumulative P&L for 1 unit**
   as a cumulative sum of P&L:

    .. math::
        \mathcal{E}(t) = \sum_{\tau=0}^t R_s(\tau)

This approach also makes little sense for returns calculation,
because we are trading one unit of the portfolio, and its value is arbitrary.
For example, suppose one enter a long position for a year when the spread is 0, what does "10 % annual return" mean?