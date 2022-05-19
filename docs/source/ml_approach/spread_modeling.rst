.. _ml_approach-spread_modeling:

===============
Spread Modeling
===============

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                margin-bottom: 5%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://www.youtube.com/embed/z6x7pLDwBVM"
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

Introduction
############

In this module, we are following the works of Dr Christian Dunis and co-authors relating to
the efficient modeling of a few major commodity spreads. The main attributes of trading spreads
are:

Advantages 
**********

- **Less likely to suffer from information shocks**

It is important to note that spreads are less likely to suffer from information shocks, 
as the movements of the two legs  will offset each other.

- **Less likely to be subject to speculative bubbles**

`(Sweeney 1988) <https://www.jstor.org/stable/2331068>`_ notes that speculative bubbles are a big source of market 
inefficiency. This effect is less likely to happen in spread markets because any
bubble effect will be replicated in the opposing leg of the spread (assuming
the two legs are sufficiently correlated). Therefore, the effect of the bubble is largely offset.

Disadvantages
*************

- **Limited Return Potential**

The spreads are hard to trade since they offer a limited potential return because of the muted
effect of market inefficiencies. This is also exacerbated by the fact that two sets of transaction costs
have to be covered in order to trade a spread. 

- **Transaction costs for multiple legs**

A point made by `(Butterworth and Holmes 2002) <https://www.tandfonline.com/doi/abs/10.1080/09603100110044236>`_ that ‘the overall profitability
of the strategy is seriously impaired by the difficulty, which traders face, in liquidating
their positions’ indicates a definite need for more discerning trade selection which is 
solved using an assortment of filters.

In the literature, the initial motivation of the works was to model and forecast the spread
as accurately as possible. As time progressed, the literature started focusing more on
seeking models/methods that discriminated between large/small moves, so that transaction
costs would be minimized.

Papers used in this module
##########################

Modelling and trading the gasoline crack spread: A non-linear story (Dunis et al. 2005)
***************************************************************************************
- Spread being modelled : Crack Spread
- Benchmark Model : Fair Value Non Linear Cointegration Model
- Novel Models : MLP, RNN, HONN
- Filters : Threshold Filter, Asymmetric Threshold Filter, Correlation Filter
- Best Results Out of Sample : HONN with standard threshold filter

Volatility filters for asset management: An application to managed futures (Dunis et al. 2005)
**********************************************************************************************
- Filters : Time Varying RiskMetrics volatility model
- Strategies : No Trade Strategy, Reverse Strategy

Modelling and Trading The Soybean-Oil Crush Spread with Recurrent and Higher Order Networks: A Comparative Analysis (Dunis et al. 2006)
****************************************************************************************************************************************
- Spread being modelled : Soy Crush Spread
- Benchmark Model : Fair Value Cointegration Model
- Novel Models : MLP, RNN, HONN
- Benchmark Filter : Traditional Threshold Filter
- Filters : Correlation Filter
- Best Results In sample : MLP with correlation filter
- Best Results Out of sample : MLP with correlation filter 
- Comparison Method : Risk Adjusted Return

Modelling and Trading the EUR/USD Exchange Rate at the ECB Fixing (Dunis et al. 2008)
*************************************************************************************
- Spread being modelled : EUR USD Exchange Rate
- Benchmark Models : ARMA MACD, Naive
- Novel Models : MLP, HONN, Pi Sigma, RNN
- Filters : Threshold Filter 

Trading and hedging the corn/ethanol crush spread using time-varying leverage and nonlinear models (Dunis et al. 2013)
**********************************************************************************************************************
- Spread being modelled : Corn Crush Spread
- Novel Modes : MLP, HONN, GPA
- Filters : Time Varying Volatility Filter 

Research Notebooks
##################

The following research notebooks can be used to better understand the components of the framework described above.

* `Crack Spread Modeling`_ - showcases the use of the filters and networks on the crack spread.

.. _`Crack Spread Modeling`: https://hudsonthames.org/notebooks/arblab/crack_spread_modeling.html

.. raw:: html

    <a href="https://hudthames.tech/3gFGwy8"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>

* `Fair Value Modeling`_ - showcases the use of the TAR model on the crack spread.

.. _`Fair Value Modeling`: https://hudsonthames.org/notebooks/arblab/fair_value_modeling.html

.. raw:: html

    <a href="https://hudthames.tech/3gFGwy8"><button style="margin: 20px; margin-top: 0px">Download Notebook</button></a>
    <a href="https://hudthames.tech/2S03R58"><button style="margin: 20px; margin-top: 0px">Download Sample Data</button></a>

Presentation Slides
###################

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTQSpwcbKAQB84c8TgT2_4hWsZdazBNCJTdb1sdVBuUQ9Bt4_MkPpEk3DsJL4rl2iocJEYC2HZW54Ef/embed?start=false&loop=false&delayms=3000"
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

* `Sweeney, R.J., 1988. Some new filter rule tests: Methods and results. Journal of Financial and Quantitative Analysis, pp.285-300. <https://www.jstor.org/stable/2331068>`_

* `Butterworth, D. and Holmes, P., 2002. Inter-market spread trading: Evidence from UK index futures markets. Applied Financial Economics, 12(11), pp.783-790. <https://www.tandfonline.com/doi/abs/10.1080/09603100110044236>`_

* `Dunis, C.L., Laws, J. and Evans, B., 2006. Modelling and trading the gasoline crack spread: A non-linear story. Derivatives Use, Trading & Regulation, 12(1-2), pp.126-145. <https://link.springer.com/article/10.1057/palgrave.dutr.1840046>`__

* `Dunis, C. and Miao, J., 2006. Volatility filters for asset management: An application to managed futures. Journal of Asset Management, 7(3-4), pp.179-189. <https://link.springer.com/article/10.1057/palgrave.jam.2240212>`_

* `Dunis, C., Laws, J. and Evans, B., 2006. Modeling and Trading the Soybean-Oil Crush Spread with Recurrent and Higher Order Networks. Artificial Higher Order Neural Networks for Economics and Business, pp.348-366. <https://pdfs.semanticscholar.org/ccc6/d7bb5f591aba83cc191096d18ad78f881347.pdf>`_

* `Dunis, C.L., Laws, J. and Sermpinis, G., 2010. Modelling and trading the EUR/USD exchange rate at the ECB fixing. The European Journal of Finance, 16(6), pp.541-560. <https://www.tandfonline.com/doi/abs/10.1080/13518470903037771>`_

* `Dunis, C.L., Laws, J., Middleton, P.W. and Karathanasopoulos, A., 2015. Trading and hedging the corn/ethanol crush spread using time-varying leverage and nonlinear models. The European Journal of Finance, 21(4), pp.352-375. <https://www.tandfonline.com/doi/abs/10.1080/1351847X.2013.830140>`_
