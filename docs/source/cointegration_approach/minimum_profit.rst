.. _cointegration_approach-minimum_profit:

.. note::
    The following documentation follows closely two papers:

    - `Loss protection in pairs trading through minimum profit bounds: a cointegration approach <http://downloads.hindawi.com/archive/2006/073803.pdf>`__ by Lin, Y.-X., McCrae, M., and Gulati, C. (2006)
    - `Finding the optimal pre-set boundaries for pairs trading strategy based on cointegration technique <https://ro.uow.edu.au/cgi/viewcontent.cgi?article=1040&context=cssmwp>`__ by Puspaningrum, H., Lin, Y.-X., and Gulati, C. M. (2010)

========================
Minimum Profit Condition
========================

After finding cointegrated asset pairs with Engle-Granger or Johansen test, a statistical arbitrageur would be most
interested in analyzing the cointegration error, which is the linear combination of the two assets based on the
cointegration vector. In this documentation, we will use "cointegration error" and the "spread" interchangeably.

A common trading strategy determining the optimal boundaries to "fade the spread", i.e. to open a trade when the current
cointegration error is sufficiently far away from its mean and expect it to mean revert.

We assume that the cointegration error follows an stationary AR(1) process. We numerically estimate the average trade
duration, average inter-trade interval, and the average number of trades based on mean first-passage time.
These values will help determine the optimal boundary that maximizes the minimum total profit (MTP) over a specific
trading horizon. It is noteworthy that the MTP here is achieved by trading one unit of the spread throughout
the trading period. A higher MTP can be achieved by trading multiple units of the spread in each trade; this module thus
allows users to define a MTP and help calculate the number of units to trade in order to secure the MTP.

We further assume that the cointegration error has symmetric distributions, which allows the optimal boundary to be set
on both sides of the spread mean. The module will output a final trading strategy that includes the entry time points
as well as the number of shares to trade, which can be used as an input for a backtester.

Mean First-passage Time of an AR(1) Process
###########################################

Consider a stationary AR(1) process:

.. math::

    Y_t = \phi Y_{t-1} + \xi_t

where :math:`-1 < \phi < 1`, and :math:`\xi_t \sim N(0, \sigma_{\xi}^2) \quad \mathrm{i.i.d}`. The mean first-passage
time over interval :math:`\lbrack a, b \rbrack` of :math:`Y_t`, starting at initial state
:math:`y_0 \in \lbrack a, b \rbrack`, which is denoted by :math:`E(\mathcal{T}_{a,b}(y_0))`, is given by

.. math::

    E(\mathcal{T}_{a,b}(y_0)) = \frac{1}{\sqrt{2 \pi}\sigma_{\xi}}\int_a^b E(\mathcal{T}_{a,b}(u)) \> \mathrm{exp} \Big( - \frac{(u-\phi y_0)^2}{2 \sigma_{\xi}^2} \Big) du + 1

This integral equation can be solved numerically using the Nystrom method, i.e. by solving the following linear
equations:

.. math::

    \begin{pmatrix}
    1 - K(u_0, u_0) & -K(u_0, u_1) & \ldots & -K(u_0, u_n) \\
    -K(u_1, u_0) & 1 - K(u_1, u_1) & \ldots & -K(u_1, u_n) \\
    \vdots & \vdots & \vdots & \vdots \\
    -K(u_n, u_0) & -K(u_n, u_1) & \ldots & 1-K(u_n, u_n)
    \end{pmatrix}
    \begin{pmatrix}
    E_n(\mathcal{T}_{a,b}(u_0)) \\
    E_n(\mathcal{T}_{a,b}(u_1)) \\
    \vdots \\
    E_n(\mathcal{T}_{a,b}(u_n)) \\
    \end{pmatrix}
    =
    \begin{pmatrix}
    1 \\
    1 \\
    \vdots \\
    1 \\
    \end{pmatrix}

where :math:`E_n(\mathcal{T}_{a,b}(u_0))` is a discretized estimate of the integral, and the Gaussian kernel function
:math:`K(u_i, u_j)` is defined as:

.. math::

    K(u_i, u_j) = \frac{h}{2 \sqrt{2 \pi} \sigma_{\xi}} w_j  \> \mathrm{exp} \Big( - \frac{(u_j - \phi u_i)^2}{2 \sigma_{\xi}^2} \Big)

and the weight :math:`w_j` is defined as the trapezoid integration rule:

.. math::

    w_j = \begin{cases}
    1 & j = 0 \quad \mathrm{and} \quad j = n \\
    2 & 0 < j < n, j \in \mathbb{N}
    \end{cases}

The time complexity for solving the above linear equation system is :math:`O(n^3)` (see `here <https://www.netlib.org/lapack/lug/node71.html>`__ for
an introduction of the time complexity of :code:`numpy.linalg.solve`), which is the most time-consuming part of this
procedure. Therefore, expect slow running time for the boundary optimization.

Minimum Total Profit (MTP)
##########################

The minimum total profit (MTP) within a specific trading horizon :math:`\lbrack 0, T \rbrack` with a specific
upper-bound :math:`U` is defined by:

.. math::

    MTP(U) = \Big( \frac{T}{TD_U + I_U} - 1 \Big) U

From the definition, the MTP is simultaneously determined by the trade duration :math:`TD_U` and the inter-trade
interval :math:`I_U`, both of which can be derived from the mean first-passage time.

Since the core idea of the approach is to "fade the spread" at :math:`U`, the trade duration can be defined
as the average time of the de-meaned cointegration error to pass 0 for the first time given that its initial value
is :math:`U`. Thus using the definition of the mean first-passage time of the cointegration error process:

.. math::

    TD_U = E(\mathcal{T}_{0, \infty}(U)) = \lim_{b \to \infty} \frac{1}{\sqrt{2 \pi} \sigma_a} \int_0^b E(\mathcal{T}_{0, b}(s)) \> \mathrm{exp} \Big( - \frac{(s- \phi U)^2}{2 \sigma_a^2} \Big) ds + 1

The inter-trade interval is defined as the average time of the de-meaned cointegration error to pass :math:`U` the first
time given its initial value is 0.

.. math::

    I_U = E(\mathcal{T}_{- \infty, U}(0)) = \lim_{-b \to - \infty} \frac{1}{\sqrt{2 \pi} \sigma_a} \int_{-b}^U E(\mathcal{T}_{-b, U}(s)) \> \mathrm{exp} \Big( - \frac{s^2}{2 \sigma_a^2} \Big) ds + 1

In both equations, :math:`\sigma_a` denotes the standard deviation of the fitted AR(1) process residual on the
cointegration error. Under the assumption that the cointegration error follows a stationary AR(1) process, the standard
deviation of the fitted residual :math:`\sigma_a` and the standard deviation of the cointegration
error :math:`\sigma_{\varepsilon}` has the following relationship:

.. math::

    \sigma_a = \sqrt{1 - \phi^2} \sigma_{\varepsilon}

To approximate the infinity limit for both integrals, we use the following stylized fact: for a stationary AR(1) process
:math:`\{ \varepsilon_t \}`, the probability that the absolute value of the process :math:`\vert \varepsilon_t \vert` is
greater than 5 times the standard deviation of the process :math:`5 \sigma_{\varepsilon}` is close to 0. Therefore, we
will use :math:`5 \sigma_{\varepsilon}` as an approximation of the infinity limit in the integrals.

Optimize the Boundaries that Maximizes MTP
##########################################

Implementation
**************

.. automodule:: arbitragelab.cointegration_approach.minimum_profit

    .. autoclass:: MinimumProfit
        :members:
        :inherited-members:

        .. automethod:: __init__