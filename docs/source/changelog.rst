=========
Changelog
=========

..
    The Following are valid options
    * :release:`0.1.0 <2020-11-14>`
    * :support:`119` Upgrade to pandas 1.0
    * :feature:`50` Add a distutils command for marbles
    * :bug:`58` Fixed test failure on OSX

..
    For Help: https://releases.readthedocs.io/en/latest/index.html

* :feature:`70` Updated requirements versions to newest stable numpy, pandas etc.
* :bug:`70 major` Fixed package breaking due to faulty werkzeug version.
* :bug:`69 major` Fixed SCS package version breaking Sparse MR Module.
* :support:`66` Added presentation slides and videos to documentation.

* :release:`0.6.0 <2021-11-15>`
* :feature:`56` H-Strategy (Renko and Kagi) Model added to the Time Series Approach Module.
* :support:`56` H-Strategy (Renko and Kagi) Model documentation.
* :feature:`55` Scaling function for cointegration vectors added to the Cointegration Approach Module.
* :feature:`54` Markov Regime-Switching Model added to the Time Series Approach Module.
* :support:`54` Markov Regime-Switching Model documentation.
* :feature:`51` OU Optimal Threshold Model Bertram added to the Time Series Approach Module.
* :feature:`51` OU Optimal Threshold Model Zeng added to the Time Series Approach Module.
* :support:`51` OU Optimal Threshold Model Bertram documentation.
* :support:`51` OU Optimal Threshold Model Zeng documentation.
* :support:`51` Updated requirements - new package (mpmath==1.2.1).
* :bug:`60 major` Fix unit tests not passing due to cvxpy bad installs.
* :bug:`58 major` Fixed maxed connections a day by pointing the get public IP server to AWS.
* :bug:`58 major` Fixed the API key validation and build server authentication.
* :feature:`58` Changed the analytics we track to: MAC Address, Public IP, API_KEY, and Function Calls + time stamps.

* :release:`0.5.0 <2021-04-15>`
* :bug:`52 major` Fixed issue with too many function calls in web analytics.
* :feature:`48` ML Approach Pairs Selection Module made more flexible - clustering and selection steps are now separate.
* :support:`48` ML Approach Pairs Selection Module documentation updated.
* :feature:`48` Hedge Ratio Estimation Module added with OLS, TLS, and Minimum HL Methods.
* :support:`48` Hedge Ratio Estimation Module documentation.
* :bug:`48 major` Fixed bug in ML Approach Pairs Selector hedge ratio calculation (previously had included intercept).
* :feature:`45` Pearson Strategy added to the Distance Approach Module.
* :support:`45` Pearson Strategy documentation.
* :feature:`46` Optimal Convergence Model added to the Stochastic Control Approach Module.
* :support:`46` Optimal Convergence Model documentation.
* :feature:`49` Cointegration and OU Model Tear Sheets added to the Visualization Module.
* :support:`49` Cointegration and OU Model Tear Sheets documentation.
* :support:`50` Updated documentation theme to hudsonthames-sphinx-docs.

* :release:`0.4.1 <2021-04-15>`
* :feature:`43` OU Model Jurek and OU Model Mudchanatongsuk added to the Stochastic Control Approach Module.
* :support:`43` OU Model Jurek and OU Model Mudchanatongsuk documentation.
* :feature:`44` CVine Copula and CVine Copula Strategy added to the Copula Approach Module.
* :support:`44` CVine Copula and CVine Copula Strategy documentation.
* :feature:`42` Options to sort pairs by zero-crossings, variance, same industry group added to the Basic Distance Strategy.
* :support:`42` Updated Basic Distance Strategy documentation.
* :feature:`40` Vine Copula Partner Selection Approaches added to the Copula Approach Module.
* :support:`40` Vine Copula Partner Selection Approaches documentation.

* :release:`0.3.1 <2021-02-19>`
* :support:`38` Removed TensorFlow from requirements and adjusted installation guide.

* :release:`0.3.0 <2021-02-16>`
* :feature:`33` Sparse Mean-Reverting Portfolios Model added to the Cointegration Approach Module.
* :support:`33` Sparse Mean-Reverting Portfolios Model documentation.
* :support:`35` Updated requirements - new package (cvxpy==1.1.10).
* :support:`33` Installation guide for Windows updated (cvxpy from conda).
* :feature:`25` Spread Modeling using Neural Networks, Filters and Fair Value Model added to the ML Approach Module.
* :feature:`25` Futures Rollover added to the Data Module.
* :support:`25` Spread Modeling, Filters and Fair Value Model documentation.
* :support:`25` Futures Rollover documentation.
* :support:`25` Updated requirements - new packages (keras==2.3.1, tensorflow==2.2.1, arch==4.16.1).
* :feature:`28` CopulaStrategy replaced with improved BasicCopulaStrategy in the Copula Approach Module.
* :feature:`28` Support of Clayton-Frank-Gumbel and Clayton-Student-Gumbel mixed copulas added to the Copula Approach Module.
* :feature:`28` Mispricing Index Trading Strategy added to the Copula Approach Module.
* :feature:`28` Quick Pairs Selection and ECDF added to the Copula Approach Module.
* :support:`28` Updated Copula Brief Intro and added Copula Deeper Intro to documentation.
* :support:`28` Mispricing Index Trading Strategy, Quick Pairs Selection and ECDF documentation.
* :support:`28` Equity Curve Convention documentation.
* :feature:`26` Multivariate Cointegration strategy (Galenko et al. 2010) added to the Cointegration Approach Module.
* :support:`26` Multivariate Cointegration strategy documentation.
* :support:`35` Updated requirements versions (numpy==1.20.1, matplotlib==3.2.2
  pandas==1.1.5, scikit-learn==0.24.1, scipy==1.6.0, statsmodels==0.12.2).
* :support:`35` Moved package to python version 3.8.
* :bug:`34` Data Importer, Distance Approach, ML Approach modules imports were not exposed.

* :release:`0.2.2 <2020-12-24>`
* :bug:`32` Copulas module imports were not exposed.

* :release:`0.2.1 <2020-12-22>`
* :bug:`Hot` Error with environment variables in analytics module.

* :release:`0.2.0 <2020-12-14>`
* :feature:`5` ML Based Pairs Selection (Horta, 2020) and Data Importer added.
* :support:`5` ML Based Pairs Selection and Data Importer documentation.
* :feature:`8` Copula strategy added (Liew et al. 2013): Log price (or equivalently, cumulative log returns) based copula strategy.
* :feature:`8` Copulas supported: Gumbel, Frank, Clayton, Joe, N13, N14, Gaussian, Student(Student-t).
* :support:`8` Copula strategy documentation (Liew et al. 2013) for log price based copula strategy.
* :feature:`19` Minimum profit optimization module added (Lin et al. 2006, Puspaningrum et al. 2010): Finding optimal pre-set boundaries for cointegrated pairs trading strategy.
* :feature:`19` Cointegrated time series simulation module added (Lin et al. 2006): Simulate cointegrated series that follows AR(1) dynamics.
* :support:`19` Minimum profit optimization documentation for cointegrated pairs trading strategy.
* :support:`19` Cointegrated time series simulation documentation.
* :feature:`22` XOU-model to the Optimal Mean Reversion module added.
* :support:`22` XOU-model documentation.
* :feature:`23` Heat potential approach module added.
* :support:`23` Heat potential approach documentation.
* :feature:`24` Quantile Time Series Strategy (SM Sarmento, N Horta, 2020) and Auto ARIMA model added.
* :support:`24` Quantile Time Series Strategy and Auto ARIMA model documentation.
* :feature:`27` CIR-model to the Optimal Mean Reversion module added.
* :support:`27` CIR-model documentation.
* :feature:`30` Track user analytics using Segment
* :support:`30` Analytics documentation.

* :release:`0.1.0 <2020-11-18>`
* :feature:`2` Kalman Filter + Kalman strategy added.
* :support:`2` Kalman Filter documentation.
* :feature:`3` Landmark techniques: Engle Granger and Johansen tests for co-integration.
* :feature:`3` Method for Half-Life of mean reverting process.
* :feature:`3` Linear & Bollinger Band strategy by EP Chan.
* :support:`3` Co-integration approach documentation.
* :feature:`4` Landmark paper: PCA Approach (Avellaneda and Lee, 2010)
* :support:`4` Documentation for PCA approach.
* :feature:`14` Landmark paper: The Distance Approach (Gatev et al. 2006).
* :support:`14` Distance approach documentation.
* :support:`14` Added a number of new tools to improve our deployment and align us with best practices. They include: CircleCI, VersionBump, Update Issue Templates, ChangeLog, Logo, Favicon.
* :feature:`15` Codependence module added.
* :support:`15` Codependence module documentation.
* :feature:`16` OU-model to the Optimal Mean Reversion module added.
* :support:`16` OU-model documentation.
* :support:`17` Added Licence, ReadMe, and RoadMap
* :support:`20` Added API Key Validation.
* :support:`20` Add install documentation and test on OS/Ubuntu/Windows.
