.. _stochastic_control_approach_strategies-ou_model_jurek:

.. note::
   The following implementations and documentation closely following work:

    `Jurek, J.W. and Yang, H., 2007, April. Dynamic portfolio selection in arbitrage. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=882536>`__.


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


