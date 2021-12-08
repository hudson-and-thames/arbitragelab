.. _stochastic_control_approach-introduction:

.. note::
    The following documentation appears in Section 5 of the following work:

    `Krauss (2015), Statistical arbitrage pairs trading strategies: Review and outlook <https://www.econstor.eu/bitstream/10419/116783/1/833997289.pdf>`__


============
Introduction
============

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                margin-bottom: 5%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://www.youtube.com/embed/P0r-Vqzpk5k"
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

Modeling asset pricing dynamics with the Ornstein-Uhlenbeck process
###################################################################

OU Model Jurek
**************

`Jurek and Yang (2007) <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=882536>`__ provide the paper with the
highest impact in this domain. In their setup, they allow non-myopic arbitrageurs to allocate their capital
to a mean-reverting spread or to a risk-free asset. The former evolves according to an Ornstein-Uhlenbeck process,
and the latter is compounded continuously with the risk-free rate. Two scenarios for investor preferences are considered
over a finite time horizon: constant relative risk aversion and the recursive Epstein-Zinutility function.
Utilizing the asset price dynamics, Jurek and Yang develop the budget constraints
and the wealth dynamics of the arbitrageurs’ assets.


Applying stochastic control theory, the authors are able to derive the Hamilton-Jacobi-Bellmann (HJB) equation and
subsequently find closed-form solutions for the value and policy functions for both scenarios. Jurek and Yang provide
the most comprehensive discussion of the stochastic control approach applied to an Ornstein-Uhlenbeck framework.

OU Model Mudchanatongsuk
************************

`Mudchanatongsuk  et  al.(2008) <http://folk.ntnu.no/skoge/prost/proceedings/acc08/data/papers/0479.pdf>`__ also solve
the stochastic control problem for pairs trading under power utility for terminal wealth.
Their ansatz mostly differs in the assumed asset pricing dynamics, but the spread also relies on an OU-process.


References
##########

*   `Krauss, Christopher (2015) : Statistical arbitrage pairs trading strategies:Review and outlook, IWQW Discussion Papers, No. 09/2015, <https://www.econstor.eu/bitstream/10419/116783/1/833997289.pdf>`__
