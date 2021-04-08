# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
This module implements the PCA approach described by by Marco Avellaneda and Jeong-Hyun Lee in
`"Statistical Arbitrage in the U.S. Equities Market"
<https://math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb20090616.pdf>`_.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# pylint: disable=invalid-name, too-many-arguments
from arbitragelab.util import devadarsh


class ETFStrategy:
    """
    This strategy creates mean reverting portfolios using Principal Components Analysis. Unlike PCA approach, each
    ETF would be treatred as a dependent variable in linear regression model and compute the residual.Similar to PCA approach,
    these residuals are used to calculate S-scores to generate trading signals. If a ETF shows good mean-reverting properties
    and the S-score deviates enough from its mean value, then a ETF is being traded. The output trading signals of
    this strategy are weights for each asset in a portfolio at each given time.
    """

    def __init__(self, n_components: int = 15):
        """
        Initialize ETF StatArb Strategy.

        The original paper suggests that the number of components would be chosen to explain at least
        50% of the total variance in time. Authors also denote that for G8 economies, stock returns are explained
        by approximately 15 factors (or between 10 and 20 factors).

        :param n_components: (int) Number of PCA principal components to use in order to build factors.
        """

        self.n_components = n_components  # Number of PCA components
        self.pca_model = PCA(n_components)  # Model for PCA calculation
        devadarsh.track('PCAStrategy')

    @staticmethod
    def volume_modified_return(matrix: pd.DataFrame, vol_matrix: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        A function to adjust the return dataframe with historical trading volume data.

        The volume-adjusted returns is calculated as:

        vol_ajust_R = R * (k-day moving average of volume) / volume_change

        :param matrix: (pd.DataFrame) DataFrame with returns that need to be standardized.
        :param vol_matrix: (pd.DataFrame) DataFrame with histoircal trading volume data.
        :param k: (int) Look-back window used for volume moving average.
        :return: (pd.DataFrame) A volume-adjusted returns dataFrame
        """

        # Fill missing data with preceding values
        returns = matrix.fillna(method='ffill')

        # Vol change
        volume_diff = vol_matrix.diff()

        # Moving Average of historical volume data
        volume_mv = vol_matrix.rolling(window=k).mean()

        # Adjustment term
        adjust_term = volume_mv / volume_diff

        # Fill missing values in adjustment term with 1s
        adjust_term = adjust_term.replace([np.inf, -np.inf], np.nan)
        adjust_term = adjust_term.fillna(1)

        # Find common columns between dataframe returns and dataframe volume_chg
        common_index = returns.index.intersection(adjust_term.index)

        # Make sure they have the same date indexes
        returns = returns.loc[common_index]
        adjust_term = adjust_term.loc[common_index]

        # Find common columns between dataframe returns and dataframe volume_chg
        common_columns = returns.columns.intersection(adjust_term.columns)

        # Make sure they have the same columns since some stocks lack volume data.
        returns = returns[common_columns]
        adjust_term = adjust_term[common_columns]

        # Modified returns after taking trading volume into account
        modified_returns = returns * adjust_term

        return modified_returns

    @staticmethod
    def standardize_data(matrix: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        """
        A function to standardize data (returns).

        The standardized returns (R)are calculated as:

        R_standardized = (R - mean(R)) / st.d.(R)

        :param matrix: (pd.DataFrame) DataFrame with returns that need to be standardized.
        :return: (pd.DataFrame. pd.Series) a tuple with two elements: DataFrame with standardized returns and Series of
            standard deviations.
        """

        # Standardizing data
        standardized = (matrix - matrix.mean()) / matrix.std()

        return standardized, matrix.std()

    @staticmethod
    def get_residuals(matrix: pd.DataFrame, etf_matrix: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.Series):
        """
        A function to calculate residuals given matrix of returns and factor returns.

        First, for each asset in a portfolio, we fit its returns to dependent variables(ETFs) returns as:

        Returns = beta_0 + betas * ETF_returns + residual

        Residuals are used to generate trading signals and beta coefficients are used as
        weights for each ETF.

        :param matrix: (pd.DataFrame) Dataframe with index and columns containing asset returns.
        :param etf_matrix: (pd.DataFrame) DataFrame with index an columns containing ETF returns.
        :return: (pd.DataFrame, pd.Series, pd.Series) Dataframe with residuals and series of beta coefficients,
                intercept.
        """

        # Creating a DataFrame to store residuals
        residual = pd.DataFrame(columns=matrix.columns, index=matrix.index)

        # And a DataFrame to store regression coefficients
        coefficient = pd.DataFrame(columns=matrix.columns, index=range(etf_matrix.shape[1]))

        # And a pd.Series to store regression intercept(beta0)
        intercept = pd.Series(index=matrix.columns, dtype=float)

        # A class for regression
        regression = LinearRegression()

        # Iterating through all tickers - to create residuals for every ETF
        for ticker in matrix.columns:
            # Fitting a regression
            regression.fit(etf_matrix, matrix[ticker])

            # Calculating residual for ETFs
            residual[ticker] = matrix[ticker] - regression.intercept_ - np.dot(etf_matrix, regression.coef_)

            # Writing down the regression coefficient
            coefficient[ticker] = regression.coef_

            # Writing down the regression intercept
            intercept[ticker] = regression.intercept_

        return residual, coefficient, intercept

    @staticmethod
    def get_sscores(residuals: pd.DataFrame, intercept: pd.Series, k: float, drift: bool = False,
                    p_value: float = None) -> pd.Series:
        """
        A function to calculate S-scores for ETFs given dataframes of residuals
        and a mean reversion speed threshold.

        From residuals, a discrete version of the OU process is created for each ETF.

        If the OU process of the asset shows a mean reversion speed above the given
        threshold k, it can be traded and the S-score is being calculated for it.

        The output of this function is a dataframe with S-scores that are directly used
        to determine if the ETFs of a given asset should be traded at this period.

        In the original paper, it is advised to choose k being less than half of a
        window for residual estimation. If this window is 60 days, half of it is 30 days.
        So k > 252/30 = 8.4. (Assuming 252 trading days in a year)

        :param residuals: (pd.DataFrame) Dataframe with residuals after fitting returns to PCA
                          factor returns.
        :param intercept: (pd.Series) Pandas Series containining intercept(beta0) of each stocks.
        :param k: (float) Required speed of mean reversion to use the ETFs in trading.
        :param drift: (bool) True if the user want to take drift into consideration, Flase, otherwise.
        :param p_value (float) The p value criteria to determine whether a residual is stationary.
        :return: (pd.Series) Series of S-scores for each asset for a given residual dataframe.
        """
        # Check residual stationarity(Drop a ticker if its residual not stationary.)
        if p_value is not None:
            for ticker in residuals.columns:
                p = sm.tsa.stattools.adfuller(residuals[ticker])[1]
                if p > p_value:
                    residuals.drop([ticker], axis=1)

        # Creating the auxiliary process K_k - discrete version of X(t)
        X_k = residuals.cumsum()

        # Variable for mean - m
        m = pd.Series(index=X_k.columns, dtype=np.float64)

        # Variable sigma for S-score calculation
        sigma_eq = pd.Series(index=X_k.columns, dtype=np.float64)

        # Variable tau for modified S-socore calculation
        tau = pd.Series(index=X_k.columns, dtype=np.float64)

        # Update (pd.Series) intercept's index
        intercept = intercept[X_k.columns]

        # Iterating over tickers
        for ticker in X_k.columns:

            # Calculate parameter b using auto-correlations
            b = X_k[ticker].autocorr()

            # Calculating the tau for every ticker
            tau[ticker] = 1 / (-np.log(b) * 252)

            # If mean reversion times are good, enter trades
            if -np.log(b) * 252 > k:
                # Temporary variable for a + zeta_n
                a_zeta = (X_k[ticker] - X_k[ticker].shift(1) * b)[1:]

                # Deriving the a parameter
                a = a_zeta.mean()

                # Deriving zeta_n series
                zeta = a_zeta - a

                # Calculating the mean parameter for every ticker
                m[ticker] = a / (1 - b)

                # Calculating sigma for S-score of each ticker
                sigma_eq[ticker] = np.sqrt(zeta.var() / (1 - b * b))

        # Small filtering for parameter m and sigma
        m = m.dropna()
        sigma_eq = sigma_eq.dropna()
        tau = tau.dropna()

        # Original paper suggests that centered means show better results
        m = m - m.mean()

        # S-score calculation for each ticker
        s_score = -m / sigma_eq

        if drift:
            m = -m - intercept * tau
            s_score = m / sigma_eq
            s_score = s_score.dropna()

        return s_score

    @staticmethod
    def _generate_signals(position_stock: pd.DataFrame, s_scores: pd.Series, coeff: pd.DataFrame,
                          sbo: float, sso: float, ssc: float, sbc: float, size: float) -> pd.DataFrame:
        """
        A helper function to generate trading signals based on S-scores.

        This function follows the logic:

        Enter a long position if s-score < −sbo
        Close a long position if s-score > −ssc
        Enter a short position if s-score > +sso
        Close a short position if s-score < +sbc

        :param position_stock: (pd.DataFrame) Dataframe with current positions for each ETF.
        :param s_scores: (pd.Series) Series with S-scores used to generate trading signals.
        :param coeff: (pd.DataFrame) Dataframe with regression coefficients od ETFs.
        :param sbo: (float) Parameter for signal generation for the S-score.
        :param sso: (float) Parameter for signal generation for the S-score.
        :param ssc: (float) Parameter for signal generation for the S-score.
        :param sbc: (float) Parameter for signal generation for the S-score.
        :param size: (float) Number of units invested in assets when opening trades. So when opening
            a long position, buying (size) units of stock and selling (size) * betas units of other
            stocks.
        :return: (pd.DataFrame) Updated dataframe with positions of each asset for each ETF.
        """

        # Generating signals using obtained s-scores
        for ticker in position_stock.columns:

            # If no generated S-score then we exit the current position
            if ticker not in s_scores.index:
                if position_stock[ticker][-1] != 0:
                    position_stock[ticker] = 0

            # If we have an S-score generated
            else:
                if position_stock[ticker][-1] == 0:

                    # Entering a long position
                    if s_scores[ticker] < -sbo:
                        position_stock.loc[-1, ticker] = size
                        position_stock.loc[0:, ticker] = -size * coeff[ticker]

                    # Entering a short position
                    elif s_scores[ticker] > sso:
                        position_stock.loc[-1, ticker] = - size
                        position_stock.loc[0:, ticker] = size * coeff[ticker]

                # Exiting a long position
                elif position_stock[ticker][0] > 0 and s_scores[ticker] > -ssc:
                    position_stock[ticker] = 0

                # Exiting a short position
                elif position_stock[ticker][0] < 0 and s_scores[ticker] < sbc:
                    position_stock[ticker] = 0

        return position_stock

    def get_signals(self, etf_matrix: pd.DataFrame, matrix: pd.DataFrame, vol_matrix: pd.DataFrame = None,
                    k: float = 8.4,
                    corr_window: int = 252,
                    residual_window: int = 60, sbo: float = 1.25, sso: float = 1.25,
                    ssc: float = 0.5, sbc: float = 0.75, size: float = 1,
                    drift: bool = False, p_value: float = None) -> pd.DataFrame:
        """
        A function to generate trading signals for given returns matrix with parameters.

        First, the correlation matrix to get PCA components is calculated using a
        corr_window parameter. From this, we get weights to calculate PCA factor returns.
        These weights are being recalculated each time we generate (residual_window) number
        of signals.

        It is expected that corr_window>residual_window. In the original paper, corr_window is
        set to 252 days and residual_window is set to 60 days. So with corr_window==252, the
        first 252 observation will be used for estimation and the first signal will be
        generated for the 253rd observation.

        Next, we pick the last (residual_window) observations to compute PCA factor returns and
        fit them to residual_window observations to get residuals and regression coefficients.

        Based on the residuals the S-scores are being calculated. These S-scores are calculated as:

        s_i = (X_i(t) - m_i) / sigma_i

        Where X_i(t) is the OU process generated from the residuals, m_i and sigma_i are the
        calculated properties of this process.

        The S-score is being calculated only for ETFs that show mean reversion speed
        above the given threshold k.

        In the original paper, it is advised to choose k being less than half of a
        window for residual estimation. If this window is 60 days, half of it is 30 days.
        So k > 252/30 = 8.4. (Assuming 252 trading days in a year)

        So, we can have mean-reverting ETFs for each asset in our portfolio. But this
        portfolio is worth investing in only if it shows good mean reversion speed and the S-score
        has deviated enough from its mean value. Based on this logic we pick promising ETFs and invest
        in them. The trading signals we get are the target weights for each of the assets
        in our portfolio at any given time.

        Trading rules to enter a mean-reverting portfolio based on the S-score are:

        Enter a long position if s-score < −sbo
        Close a long position if s-score > −ssc
        Enter a short position if s-score > +sso
        Close a short position if s-score < +sbc

        The authors empirically chose the optimal values for the above parameters based on stock
        prices for years 2000-2004 as: sbo = sso = 1.25; sbc = 0.75; ssc = 0.5.

        Opening a long position on a ETF means buying one dollar of the corresponding asset
        and selling beta_i1 dollars of ETF1, beta_i2 dollars of ETF2 and so on. Opening a short position means selling the
        corresponding asset and buying betas of ETFs.

        :param etf_matrix: (pd.DataFrame) DataFrame with index an columns containing ETF returns.
        :param matrix: (pd.DataFrame) DataFrame with returns for assets.
        :param vol_matrix: (pd.DataFrame) DataFrame with historical volume data.
        :param k: (float) Required speed of mean reversion to use the ETFs in trading.
        :param corr_window: (int) Look-back window used for correlation matrix estimation.
        :param residual_window: (int) Look-back window used for residuals calculation.
        :param sbo: (float) Parameter for signal generation for the S-score.
        :param sso: (float) Parameter for signal generation for the S-score.
        :param ssc: (float) Parameter for signal generation for the S-score.
        :param sbc: (float) Parameter for signal generation for the S-score.
        :param size: (float) Number of units invested in assets when opening trades. So when opening
            a long position, buying (size) units of stock and selling (size) * betas units of other
            stocks.
        :param drift: (bool) True if a user want to take drift into consideration, Flase, otherwise.
        :param p_value (float) The p value criteria to determine whether a residual is stationary.
        :return: (pd.DataFrame) DataFrame with target weights for each asset at every observation.
            It is being calculated as a combination of ETFs that are satisfying the mean reversion
            speed requirement and S-score values.
        """
        # pylint: disable=too-many-locals

        if vol_matrix is not None:
            matrix = self.volume_modified_return(matrix, vol_matrix, residual_window)

        # Dataframe containing target quantities - trading signals
        target_quantities = pd.DataFrame()

        # Iterating through time windows
        for t in range(corr_window - 1, len(matrix.index) - 1):
            # Look-back window of observations used
            obs_residual = matrix[(t - residual_window + 1):(t + 1)]
            etf = etf_matrix[(t - residual_window + 1):(t + 1)]

            # Calculating residuals for this window
            resid, coeff, intercept = self.get_residuals(obs_residual, etf)

            # Finding the S-scores for ETFs in this period (no change!)
            s_scores = self.get_sscores(resid, intercept, k, drift, p_value)

            # Series of current positions for assets in our portfolio
            position_stock = pd.DataFrame(0, columns=matrix.columns, index=[-1] + list(range(etf_matrix.shape[1])))

            # Generating signals using obtained S-scores
            position_stock = self._generate_signals(position_stock, s_scores, coeff,
                                                    sbo, sso, ssc, sbc, size)

            # Combine columns in DataFrame matrix, and DataFrame etf_matrix
            tol_col = matrix.columns.append(etf_matrix.columns)

            # Temporary series to store all weights
            position_stock_temp = pd.Series(0, index=tol_col, dtype=np.float64)

            # Adding also first stocks from all ETFs
            position_stock_temp = position_stock_temp + position_stock.iloc[0]
            position_stock_temp = position_stock_temp.fillna(0.0)

            # Adding final Series of weights to a general DataFrame with weights
            target_quantities[matrix.index[t]] = position_stock_temp

        # Transposing to make dates as an index of the resulting DataFrame
        target_quantities = target_quantities.T

        return target_quantities
