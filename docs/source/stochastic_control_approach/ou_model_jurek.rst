.. _stochastic_control_approach_strategies-ou_model_jurek:

.. note::
   The following implementations and documentation closely following work:

    `Jurek, J.W. and Yang, H., 2007, April. Dynamic portfolio selection in arbitrage. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=882536>`__


==============
OU Model Jurek
==============

Introduction
############

In this module, we derive the optimal dynamic strategy for arbitrageurs with a finite horizon and
non-myopic preferences facing a mean-reverting arbitrage opportunity (e.g. an equity pairs
trade).

The law of one price asserts that - in a frictionless market - securities with identical payoffs
must trade at identical prices. If this were not the case, arbitrageurs could generate a riskless
profit by supplying (demanding) the expensive (cheap) asset until the mispricing was eliminated.
Of course, real world markets are not frictionless, and the prices of securities with identical payoffs
may significantly diverge for extended periods of time. Arbitrageurs can earn potentially attractive
profits by taking offsetting positions in these relatively mispriced securities, but a worsening of the
mispricing can result in sizable capital losses.

Unlike textbook arbitrages, which generate riskless profits and require no capital commitments,
exploiting real-world mispricings requires the assumption of some combination of horizon and divergence risk.
These two risks represent the uncertainty about whether the mispricing will converge before the positions
must be closed (or profits reported) and the possibility of a worsening in the mispricing prior to its elimination.

While a complete treatment of optimal arbitrage and price formation requires a general equilibrium framework,
this paper takes on the more modest goal of examining the arbitrageur's strategy in
partial equilibrium.

Modelling
#########

To capture the presence of horizon and divergence risk, we model the dynamics of the mispricing
using a mean-reverting stochastic process. Under this process, although the mispricing is guaranteed
to be eliminated at some future date, the timing of convergence, as well as the maximum magnitude
of the mispricing prior to convergence, are uncertain. With this assumption, we are able to derive
the arbitrageur's optimal dynamic portfolio policy for a set of general, non-myopic preference
specifications, including CRRA utility defined over wealth at a finite horizon and Epstein-Zin
utility defined over intermediate cash flows (e.g. fees). This allows us to analytically examine the
role of intertemporal hedging demands in arbitrage activities and represents a novel contribution
relative to the commonly assumed log utility specification, under which hedging demands are absent.

We find that, in the presence of horizon and divergence risk, there is a critical level of the mispricing
beyond which further divergence in the mispricing precipitates a reduction in the allocation.
Although a divergence in the mispricing results in an improvement in the instantaneous investment
opportunity set and should induce added participation by rational arbitrageurs, this effect can be
more than offset by the combination of the loss in wealth and the nearing of the evaluation date,
which reduce the arbitrageur's effective risk-bearing capacity. The complex tradeoff between these
two effects leads to the creation of a time-varying boundary, outside of which continued divergence
of the mispricing induces rational arbitrageurs to cut their losses and decrease their allocations to
the mispricing.

The central assumption of our model is that the arbitrage opportunity is described by an
Ornstein-Uhlenbeck process (henceforth OU). The OU process captures the two key features of a
real-world mispricing: the convergence times are uncertain and the mispricing can diverge arbitrarily far
from its mean prior to convergence.

The optimal trading strategies we derive are self-financing and can be interpreted as the optimal
trading rules for a fund which is not subject to withdrawals but also cannot raise additional assets
(i.e. a closed-end fund). The dynamics of the optimal allocation to the arbitrage opportunity are
driven by two factors: the necessity of maintaining wealth (equity) above zero and the proximity
of the arbitrageur's terminal evaluation date, which affects his appetite for risk.


Investor Preferences
********************

We consider two alternative preferences structures for the arbitrageur in our continuous-time
model. In the first, we assume that the agent has constant relative risk aversion and maximizes
the discounted utility of terminal wealth. The arbitrageur's value function at time :math:`t` - denoted by
:math:`V_t` - takes the form:

.. math::

    V_{t}=\sup E_{t}\left[e^{-\beta(T-t)} \frac{W_{T}^{1-\gamma}}{1-\gamma}\right]

The second preference structure we consider is the recursive utility of Epstein and Zin (1989, 1991),
which allows the elasticity of intertemporal substitution and the coefficient of relative risk aversion
to vary independently. Under this preference specification, the value function of the arbitrageur is
given by:

.. math::

    V_{t}=\sup E_{t}\left[\int_{t}^{T} f\left(C_{s}, J_{s}\right) d s\right]

where :math:`f\left(C_{s}, J_{s}\right)` is the normalized aggregator for the continuous-time Epstein-Zin utility function:

.. math::

    f\left(C_{t}, J_{t}\right)=\beta(1-\gamma) \cdot J_{t} \cdot\left[\log C_{t}-\frac{1}{1-\gamma} \log \left((1-\gamma) J_{t}\right)\right]


Here we consider the special case of a unit elasticity of intertemporal substitution (:math:`\psi = 1`).

.. note::

    In this paper, the choice of preference structures is driven by economic intuition regarding the incentives
    of real-life arbitrageurs. In particular, it can be assumed that the arbitrageur
    is a proprietary trading desk or delegated money manager with a fixed investment horizon. It
    seems likely that such investors would only be interested in the distribution of wealth at a finite
    horizon, e.g. at the end of the fiscal year, rather than the value of a long-dated consumption
    stream.

    However, the decision to model arbitrageurs as finite-horizon CRRA investors neglects
    the role of management fees, which are often collected by arbitrageurs. To capture this feature,
    we also consider the Epstein-Zin model specialized to the case of a unit elasticity of inter-temporal
    substitution. In this case the agent's consumption to wealth ratio is constant,
    which we exploit as a model of a flat management fee, collected (and consumed) as a continuous
    stream rather than as a lump-sum payment.


Spread Construction
*******************

To construct the spread for the portfolio, firstly we calculate the total return index for each asset :math:`i` in the spread.

.. math::
    P_{i, t}=\left(\frac{1}{P_{i, 1}}\right) \cdot\left(P_{i, 1} \cdot \prod_{j=1}^{t-1}\left(1+R_{i, j+1}\right)\right)

The price spread is then constructed by taking a linear combination of the total return indices.
These weights are estimated by using a co-integrating regression technique such as Engle Granger.


Optimal Portfolio Strategy
**************************

The portfolio consists of a riskless asset and the mean reverting spread. We denote
the prices of the two assets by :math:`B_t` and :math:`S_t`, respectively. Their dynamics are given by,

.. math::

    \begin{aligned}
    d B_{t} &=r B_{t} d t \\
    d S_{t} &=\kappa\left(\bar{S}-S_{t}\right) d t+\sigma d Z
    \end{aligned}

.. note::

    We assume that there are no margin constraints, no transaction costs and a frictionless, continuous-time setting.

The evolution of wealth which determines the budget constraints is written as,

.. math::

    d W_{t}=N_{t} d S_{t}+M_{t} d B_{t}-C_{t} 1\left[C_{t}>0\right] d t

where :math:`N_t` denotes the number of units of spread and :math:`M_t` denotes the number of riskless assets.

For the terminal wealth problem, the optimal portfolio allocation is given by:

.. math::

    N(W, S, \tau)=\left\{\begin{array}{cc}
    \left(\frac{\kappa(\bar{S}-S)-r S}{\sigma^{2}}\right) W & \gamma=1 \\
    \left(\frac{\kappa(\bar{S}-S)-r S}{\gamma \sigma^{2}}+\frac{2 A(\tau) S+B(\tau)}{\gamma}\right) W & \gamma \neq 1
    \end{array}\right.

The functions :math:`A(\tau)` and :math:`B(\tau)` depend on the time remaining to the horizon and the parameters of the underlying model.

For the intermediate consumption problem, the optimal portfolio allocation has the same form as the corresponding equation for terminal wealth problem.

Obviously, the functional form of the coefficient functions :math:`A(\tau)` and :math:`B(\tau)` are different.


Stabilization Region
********************

In this section, We are interested in determining the direction in which an arbitrageur trades in response
to a shock to the value of the spread asset. If an arbitrageur increases his position in the spread asset
in response to an adverse shock, his trading is likely to have a stabilizing effect on the mispricing,
contributing to its elimination in equilibrium. Conversely, if the arbitrageur decreases his position
in response to the adverse shock, his trading will tend to exacerbate the mispricing.

Sometimes arbitrageurs do not arbitrage. For instance, if the mispricing is sufficiently wide, a divergence
in the mispricing can result in the decline of the total allocation, as the wealth effect dominates
the improvement in the investment opportunity set. To characterize the conditions under which arbitrageurs cease
to trade against the mispricing, we derive precise, analytical conditions for the time-varying envelope within which
arbitrageurs trade against the mispricing.

In the general case when :math:`\bar{S} \neq 0` the range of values of :math:`S`
for which the arbitrageur's response to an adverse shock is stabilizing - i.e. the agent trades against
the spread, increasing his position as the spread widens - is determined by a time-varying envelope
determined by both :math:`A(\tau)` and :math:`B(\tau)`. The boundary of the stabilization region is determined by the
following inequality:

.. math::

    \left| \phi(\tau) S+\frac{\kappa \bar{S}+\sigma^{2} B(\tau)}{\gamma \sigma^{2}}\right |<\sqrt{-\phi(\tau)}

where,

.. math::

    \phi(\tau) = \left(\frac{2 A(\tau)}{\gamma}-\frac{\kappa+r}{\gamma \sigma^{2}}\right)

As long as the spread is within the stabilization region, the improvement in investment opportunities
from a divergence of the spread away from its long-run mean outweighs the negative
wealth effect and the arbitrageur increases his position, :math:`N`, in the mean-reverting asset. When
the spread is outside of the stabilization region, the wealth effect dominates, leading the agent to
curb his position despite an improvement in investment opportunities.


Fund Flows
**********

This section deals with the inclusion of fund flows. Delegated managers are not only exposed to the
financial fluctuations of asset prices but also to their client's desires to contribute or withdraw funds.
Paradoxically, clients are most likely to withdraw funds after performance has been poor
(i.e. spreads have been widening) and investment opportunities are the best.

In the presence of fund flows the evolution of wealth under management will depend not only
on performance, denoted by :math:`\Pi_t`, but also on fund flows, :math:`F_t`.We therefore have:

.. math::

    \begin{aligned}
    d \Pi &=\tilde{N} d S+(W-\tilde{N} S) r d t \\
    d F &=f d \Pi+\sigma_{f} W d Z_{f} \\
    d W &=d \Pi+d F=(1+f) d \Pi+\sigma_{f} W d Z_{f}
    \end{aligned}

where :math:`\tilde{N}` is the optimal policy rule chosen by a fund manager facing fund flows of the type described
above, and :math:`E[d Z_{f} dZ] = 0`. The fund flow magnifies the effect of performance on wealth
under management, with each dollar in performance generating a fund flow of :math:`f` dollars.


The optimal portfolio allocation of an agent with constant relative risk aversion with utility
defined over terminal wealth, in the presence of fund flows is given by:

.. math::

    \tilde{N}(S, \tau)=\left(\frac{1}{1+f}\right) \cdot N(S, \tau)

where :math:`N(S, \tau)` is the optimal policy function in the problem without fund flows
and :math:`f` denotes the proportionality coefficient.

The intuition behind this elegant solution is simple. The performance-chasing component of
fund flows increases the volatility of wealth by a factor of :math:`(1 + f)`, causing a manager who anticipates
this flow to commensurately decrease the amount of risk taken on by the underlying strategy.


How to use this submodule
#########################


Step 1: Model fitting
*********************


Implementation
==============

.. automodule:: arbitragelab.stochastic_control_approach.ou_model_jurek

.. autoclass:: StochasticControlJurek
   :members: __init__

.. automethod:: StochasticControlJurek.fit


Step 2: Getting the Optimal Portfolio Weights
*********************************************


Implementation
==============

.. automethod:: StochasticControlJurek.optimal_portfolio_weights


Example
#######


Research Notebook
#################



References
##########

