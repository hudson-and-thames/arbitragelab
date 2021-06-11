# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
This module implements interactive Tear Sheets for various modules of the ArbitrageLab package.
"""
# pylint: disable=too-many-lines, too-many-locals, invalid-name, unused-argument
# pylint: disable=too-many-arguments, too-many-statements, unused-variable, broad-except

import warnings

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (kurtosis, skew, shapiro)
from statsmodels.tsa.stattools import (pacf, adfuller, acf)
import dash
from dash.dependencies import (Input, Output)
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go
from jupyter_dash import JupyterDash

from arbitragelab.optimal_mean_reversion import OrnsteinUhlenbeck
from arbitragelab.cointegration_approach import (get_half_life_of_mean_reversion,
                                                 EngleGrangerPortfolio, JohansenPortfolio)
from arbitragelab.util import devadarsh


class TearSheet:
    """
    This class implements an interactive visualization tool in the form of a Dash-based web application for
    showcasing analytics for a provided pair of assets using modules and techniques implemented in the ArbitrageLab
    package.

    In the current state, this module includes cointegration analysis using both Engle-Granger and Johansen tests,
    and Ornstein-Uhlenbeck model-based analytics alongside with model fitting and optimal entry/liquidation levels
    calculation.
    """

    def __init__(self):
        """
        Initialize.
        """

        self.data = None  # The dataframe of two assets
        self.blue = '#0C9AAC'  # Blue Munsell brand color
        self.orange = '#DE612F'  # Flame brand color
        self.grey = '#949494'  # Spanish Grey brand color
        self.light_grey = '#F2F3F4'  # Alice Blue brand color
        self.black = '#0B0D13'  # Rich Black brand color
        self.white = '#ffffff'  # White brand color
        self.text_font = 'Arial'  # Font used for test outputs
        self.header_font = 'Arial'  # Font used for headers

        devadarsh.track('TearSheet')

    @staticmethod
    def _get_app(app_display='default'):
        """
        Sets the default space for building the visualization app.

        :param app_display: (str) Parameter that signifies whether to open a web app in a separate tab or inside
            a jupyter notebook ['default' or 'jupyter'].
        :return: (Dash) The Dash app object, which can be run using run_server.
        """

        if app_display == 'jupyter':
            app = JupyterDash()
        else:
            app = dash.Dash()

        return app

    def _get_basic_assets_data(self):
        """
        Gets the basic data that characterizes each of the assets from the provided pair, such as
        asset name, normalized asset price, ADF test result dataframe and ADF test statistic.

        :return: (tuple) A combined tuple of the basic data for tear sheet output.
        """

        # Assigning the data variable
        data = self.data

        # Getting the name of the first asset
        asset_name_1 = data.columns[0]

        # Getting the normalized price time series for the first asset
        norm_asset_price_1 = (data[asset_name_1] - data[asset_name_1].min()) /\
                             (data[asset_name_1].max() - data[asset_name_1].min())

        asset_price_1 = pd.DataFrame(data=data[asset_name_1])

        # Creating the pd.DataFrame that represents the ADF test results for the first asset
        adf_asset_1 = pd.DataFrame(data={'Confidence level': ['99%', '95%', '90%'],
                                         'Values': list(adfuller(asset_price_1, autolag='AIC')[4].values())})

        # Getting the ADF test statistic value for the first asset
        test_stat_1 = adfuller(asset_price_1, autolag='AIC')[0]

        # Getting the name of the second asset
        asset_name_2 = data.columns[1]

        # Getting the normalized price time series for the second asset
        norm_asset_price_2 = (data[asset_name_2] - data[asset_name_2].min()) /\
                             (data[asset_name_2].max() - data[asset_name_2].min())

        asset_price_2 = pd.DataFrame(data=data[asset_name_2])

        # Creating the pd.DataFrame that represents the ADF test results for the second asset
        adf_asset_2 = pd.DataFrame(data={'Confidence level': ['99%', '95%', '90%'],
                                         'Values': list(adfuller(asset_price_2, autolag='AIC')[4].values())})

        # Getting the ADF test statistic value for the second asset
        test_stat_2 = adfuller(asset_price_2, autolag='AIC')[0]

        # Creating the output tuple containing all the obtained data
        output = (asset_name_1, norm_asset_price_1, adf_asset_1, test_stat_1,
                  asset_name_2, norm_asset_price_2, adf_asset_2, test_stat_2)

        return output

    @staticmethod
    def _residual_analysis(residuals):
        """
        Calculates all the data connected to residual analysis such as: standard deviation,
        half-life, skewness, kurtosis, Shapiro-Wilk normality test results, QQ-plot data,
        result of ACF and PACF calculations, and returns it in a form of a tuple.

        :param residuals: (np.array) Residuals results from the Engle-Granger test.
        :return: (tuple) Combined results of the analysis of residuals.
        """

        # Calculating the statistical measures connected
        standard_deviation = residuals.std()
        half_life = get_half_life_of_mean_reversion(residuals)
        skewness = skew(residuals)
        kurtosis_ = kurtosis(residuals)

        # Performing the Shapiro-Wilk normality test
        _, p = shapiro(residuals)

        # Interpreting the normality test results
        if p > 0.05:
            shapiro_wilk = 'Passed'
        else:
            shapiro_wilk = 'Failed'

        # Creating a representative dataframe for all statistical characteristics
        residuals_dataframe = pd.DataFrame(data={
            'Characteristic': ['Standard Deviation', 'Half-life', 'Skewness', 'Kurtosis',
                               'Shapiro-Wilk normality test'],
            'Value': [round(standard_deviation, 5), round(half_life, 5), round(skewness, 5), round(kurtosis_, 5),
                      shapiro_wilk]})

        # Calculate the QQ-plot values
        qq_plot_y = stats.probplot(residuals, dist='norm', sparams=1)
        qq_plot_x = np.array([qq_plot_y[0][0][0], qq_plot_y[0][0][-1]])

        # Calculate auto-correlation and partial auto-correlation function
        pacf_result = pacf(residuals.diff()[1:], nlags=20)[1:]
        acf_result = acf(residuals.diff()[1:], nlags=20, fft=True)[1:]

        # Combining the output tuple
        output = (residuals, residuals_dataframe,
                  qq_plot_y, qq_plot_x,
                  pacf_result, acf_result)

        return output

    def _get_engle_granger_data(self, data):
        """
        Calculates all the data connected to a portfolio created with the Engle-Granger approach,
        such as: ADF test result, ADF test statistic, cointegration vector, portfolio returns,
        portfolio price, residuals, residual analysis results.

        :param data: (pd.DataFrame) A dataframe of two assets.
        :returns: (tuple) A tuple of combined results connected to an Engle-Granger portfolio.
        """

        # Calculating the data returns
        data_returns = (data / data.shift(1) - 1)[1:]

        # Initializing an instance of EngleGrangerPortfolio class
        portfolio = EngleGrangerPortfolio()

        # Attempting to create a cointegrated portfolio
        portfolio.fit(data, add_constant=True)

        # Getting the ADF test results
        adf = portfolio.adf_statistics

        # Getting the ADF test statistic
        adf_test_stat = adf.loc['statistic_value'][0]

        # Creating and ADF test statistic dataframe
        adf_dataframe = pd.DataFrame(
            data={'Confidence levels': ['99%', '95%', '90%'], 'Values': list(adf[:-1][0].round(5))})

        # Getting the cointegration vector
        cointegration_vector = portfolio.cointegration_vectors

        # Calculating the scaled cointegration vector
        scaled_vector = (cointegration_vector.loc[0] / abs(cointegration_vector.loc[0]).sum())

        # Calculating the portfolio returns
        portfolio_returns = (data_returns * scaled_vector).sum(axis=1)

        # Calculating the portfolio price
        portfolio_price = (portfolio_returns + 1).cumprod()

        # Getting the results of a residual analysis
        residuals, residuals_dataframe, qq_y, x, pacf_result, acf_result = self._residual_analysis(portfolio.residuals)

        # Combining the output tuple
        output = (adf_dataframe, adf_test_stat, cointegration_vector, portfolio_returns, portfolio_price, residuals,
                  residuals_dataframe, qq_y, x, pacf_result, acf_result)

        return output

    def _get_johansen_data(self):
        """
        Calculates all the data connected to a portfolio created with Johansen approach,
        such as: eigen test results, trace test results, test statistics, cointegration vectors, portfolio returns,
        portfolio prices.

        :return: (tuple) A tuple of combined results connected to a Johansen portfolio(s).
        """

        data = self.data

        # Get asset names
        asset_name_1 = data.columns[0]
        asset_name_2 = data.columns[1]

        # Calculating the data returns
        data_returns = (data / data.shift(1) - 1)[1:]

        # Initializing an instance of JohansenPortfolio class
        portfolio = JohansenPortfolio()

        # Attempting to create a cointegrated portfolio
        portfolio.fit(data, det_order=0)

        # Getting teh raw results of eigen and trace cointegration tests
        test_eigen = portfolio.johansen_eigen_statistic

        test_trace = portfolio.johansen_trace_statistic

        # Creating a representative dataframes for the cointegration test results
        test_eigen_dataframe = pd.DataFrame(data={'Confidence levels': ['99%', '95%', '90%'],
                                                  'Values for {}'.format(asset_name_1): list(
                                                      test_eigen.iloc[2::-1][asset_name_1].round(5)),
                                                  'Values for {}'.format(asset_name_2): list(
                                                      test_eigen.iloc[2::-1][asset_name_2].round(5))})
        test_trace_dataframe = pd.DataFrame(data={'Confidence levels': ['99%', '95%', '90%'],
                                                  'Values for {}'.format(asset_name_1): list(
                                                      test_trace.iloc[2::-1][asset_name_1].round(5)),
                                                  'Values for {}'.format(asset_name_2): list(
                                                      test_trace.iloc[2::-1][asset_name_2].round(5))})

        # Getting the test statistic of every asset for both tests
        eigen_test_statistic_1 = test_eigen[asset_name_1][-1].round(5)

        eigen_test_statistic_2 = test_eigen[asset_name_2][-1].round(5)

        trace_test_statistic_1 = test_trace[asset_name_1][-1].round(5)

        trace_test_statistic_2 = test_trace[asset_name_2][-1].round(5)

        # Calculating the scaled cointegration vectors
        scaled_vector_1 = (portfolio.cointegration_vectors.loc[0] /
                           abs(portfolio.cointegration_vectors.loc[0]).sum()).round(5)

        scaled_vector_2 = (portfolio.cointegration_vectors.loc[1] /
                           abs(portfolio.cointegration_vectors.loc[1]).sum()).round(5)

        # Calculating portfolio returns and portfolio price for the first cointegration vector
        portfolio_returns_1 = (data_returns * scaled_vector_1).sum(axis=1)

        portfolio_price_1 = (portfolio_returns_1 + 1).cumprod()

        # Calculating portfolio returns and portfolio price for the second cointegration vector
        portfolio_returns_2 = (data_returns * scaled_vector_2).sum(axis=1)

        portfolio_price_2 = (portfolio_returns_2 + 1).cumprod()

        # Combining all the data into the output tuple
        output = (test_eigen_dataframe, test_trace_dataframe, eigen_test_statistic_1, eigen_test_statistic_2,
                  trace_test_statistic_1, trace_test_statistic_2, scaled_vector_1, scaled_vector_2, portfolio_returns_1,
                  portfolio_price_1, portfolio_returns_2, portfolio_price_2)

        return output

    def _adf_test_result(self, dataframe, test_statistic):
        """
        Returns the interpretation of the ADF test along with specifics of the
        message style that should be used.

        :param dataframe: (pd.DataFrame) Dataframe that contains the result of the ADF test.
        :param test_statistic: (float) ADF test statistic value.
        :return: (tuple) A combined tuple with test interpretation message, color and font.
        """

        # Setting the default message font and color
        font = self.text_font
        color = self.black

        # Interpreting the ADF test results
        if test_statistic < dataframe.iloc[0][1]:

            message = 'The hypothesis is not rejected at 99% confidence level'

        elif test_statistic < dataframe.iloc[1][1]:

            message = 'The hypothesis is not rejected at 95% confidence level'

        elif test_statistic < dataframe.iloc[2][1]:

            message = 'The hypothesis is not rejected at 90% confidence level'

        else:

            # Specifying the color, font and message for hypothesis rejection
            message = 'HYPOTHESIS REJECTED'
            color = self.orange
            font = self.header_font

        # Combining the message and output style into an input tuple
        output = (message, color, font)

        return output

    def _johansen_test_result(self, dataframe, test_statistic_1, test_statistic_2):
        """
        Returns the interpretation of the ADF test along with specifics of the
        message style that should be used.

        :param dataframe: (pd.DataFrame) Dataframe that contains the result of the ADF test.
        :param test_statistic_1: (float) Test statistic value for the first asset.
        :param test_statistic_2: (float) Test statistic value for the first asset.
        :return: (tuple) A combined tuple with test interpretation message, color and font.
        """

        # Setting the default message font and color
        font = self.text_font
        color = self.black

        # Interpreting the ADF test results
        if (test_statistic_1 > dataframe.iloc[0][1]) and test_statistic_2 > dataframe.iloc[0][2]:

            message = 'The hypothesis is not rejected at 99% confidence level'

        elif test_statistic_1 > dataframe.iloc[1][1] and test_statistic_2 > dataframe.iloc[1][2]:

            message = 'The hypothesis is not rejected at 95% confidence level'

        elif test_statistic_1 > dataframe.iloc[2][1] and test_statistic_2 > dataframe.iloc[2][2]:

            message = 'The hypothesis is not rejected at 90% confidence level'

        else:

            # Specifying the color, font and message for hypothesis rejection
            message = 'HYPOTHESIS REJECTED'
            color = self.orange
            font = self.header_font

        # Combining the message and output style into an input tuple
        output = [message, color, font]

        return output

    def _asset_prices_plot(self, norm_1, norm_2, name_1, name_2):
        """
        Creates a plot of a normalized price series of a pair of assets.

        :param norm_1: (pd.Series) Normalized price series of the first asset.
        :param norm_2: (pd.Series) Normalized price series of the second asset.
        :param name_1: (str) The name of the first asset.
        :param name_2: (str) The name of the second asset.
        :return: (go.Figure) Plot of the normalized prices.
        """

        # Create a figure
        asset_prices = go.Figure()

        # Adding the two normalized price plots
        asset_prices.add_trace(
            go.Scatter(x=norm_1.index,
                       y=norm_1,
                       mode='lines',
                       line=dict(color=self.blue), name=name_1))
        asset_prices.add_trace(
            go.Scatter(x=norm_2.index,
                       y=norm_2,
                       mode='lines',
                       line=dict(color=self.orange), name=name_2))

        # Updating the plot characteristics
        asset_prices.update_layout(legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.925),
                                   title="Asset prices",
                                   xaxis_title="Date",
                                   yaxis_title="Price",
                                   font_family=self.header_font,
                                   font_size=18,
                                   height=500,
                                   hovermode='x unified')

        # Adding the range slider
        asset_prices.update_xaxes(rangeslider_visible=True)

        return asset_prices

    def _portfolio_plot(self, portfolio_price, portfolio_return):
        """
        Creates a plot of a normalized price series of a pair of assets.

        :param portfolio_price: (pd.Series) Normalized portfolio price.
        :param portfolio_return: (pd.Series) Portfolio return series.
        :return: (go.Figure) Plot of the normalized prices.
        """

        # Create a figure
        portfolio = go.Figure()

        # Adding the portfolio price and returns to the plot
        portfolio.add_trace(
            go.Scatter(x=portfolio_price.index,
                       y=portfolio_price,
                       mode='lines',
                       line=dict(color=self.blue),
                       name='Portfolio price'))
        portfolio.add_trace(
            go.Scatter(x=portfolio_return.index,
                       y=portfolio_return,
                       mode='lines',
                       line=dict(color=self.orange),
                       name='Portfolio return',
                       visible='legendonly'))

        # Updating the plot characteristics
        portfolio.update_layout(legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.77),
                                title="Normalized portfolio",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                font_family=self.header_font,
                                font_size=18,
                                height=500,
                                margin=dict(l=30, r=30, t=50, b=30))

        # Adding the range slider
        portfolio.update_xaxes(rangeslider_visible=True)

        return portfolio

    def _pacf_plot(self, pacf_data):
        """
        Creates a plot of a partial auto-correlation function.

        :param pacf_data: (pd.Series) PACF function value.
        :return: (go.Figure) Plot of the normalized prices.
        """

        # Establishing the data for a PACF graph
        trace = {"name": "PACF", "type": "bar", "marker_color": self.blue, "y": pacf_data, "x": np.arange(1, 20)}

        # Creating a figure with established data
        pacf_plot = go.Figure(data=trace)

        # Updating the plot characteristics
        pacf_plot.update_layout(title="PACF",
                                xaxis_title="Lag",
                                font_family=self.header_font,
                                font_size=14,
                                height=250,
                                margin=dict(l=30, r=30, t=50, b=30))

        # Updating the axis range
        pacf_plot.update_yaxes(range=[-1, 1])

        return pacf_plot

    def _acf_plot(self, acf_data):
        """
        Creates a plot of an auto-correlation function.

        :param acf_data: (pd.Series) ACF function value.
        :return: (go.Figure) Plot of the normalized prices.
        """

        # Establishing the data for a ACF graph
        trace = {"name": "ACF", "type": "bar", "marker_color": self.blue, "y": acf_data, "x": np.arange(1, 20)}

        # Creating a figure with established data
        acf_plot = go.Figure(data=trace)

        # Updating the plot characteristics
        acf_plot.update_layout(title="ACF",
                               xaxis_title="Lag",
                               font_family=self.header_font,
                               font_size=14,
                               height=250,
                               margin=dict(l=30, r=30, t=50, b=30))

        # Updating the axis range
        acf_plot.update_yaxes(range=[-1, 1])

        return acf_plot

    def _residuals_plot(self, residuals):
        """
        Creates a plot of residuals.

        :param residuals: (pd.Series) Residuals series.
        :return: (go.Figure) Plot of the residuals.
        """

        # Creating a figure
        resid_plot = go.Figure()

        # Adding the residuals to the plot
        resid_plot.add_trace(go.Scatter(x=residuals.index,
                                        y=residuals,
                                        mode='lines',
                                        line=dict(color=self.orange)))

        # Adding the mean line
        resid_plot.add_shape(type='line',
                             x0=residuals.index[0],
                             y0=residuals.mean(),
                             x1=residuals.index[-1],
                             y1=residuals.mean(),
                             line=dict(color=self.grey, dash='dash'))

        # Updating the plot characteristics
        resid_plot.update_layout(title="Residuals plot",
                                 font_family=self.header_font,
                                 font_size=14,
                                 height=250,
                                 margin=dict(l=30, r=30, t=50, b=30))

        return resid_plot

    def _qq_plot(self, qq_data, x_data):
        """
        Creates a QQ-plot of residuals.

        :param qq_data: (np.array) Sample quantiles.
        :param x_data: (np.array) Theoretical quantiles.
        :return: (go.Figure) Q-Q plot.
        """

        # Creating a figure
        qq_plot = go.Figure()

        # Plotting the quantile-quantile scatter plot and the comparison line
        qq_plot.add_scatter(x=qq_data[0][0],
                            y=qq_data[0][1],
                            mode='markers',
                            line=dict(color=self.blue))
        qq_plot.add_scatter(x=x_data,
                            y=qq_data[1][1] + qq_data[1][0] * x_data,
                            mode='lines',
                            line=dict(color=self.grey))

        # Updating the plot characteristics
        qq_plot.update_layout(title="Q-Q Plot",
                              font_family=self.header_font,
                              font_size=14,
                              height=550,
                              margin=dict(l=30, r=30, t=50, b=30),
                              showlegend=False)
        return qq_plot

    def _jh_coint_test_div(self, asset_1, asset_2, cointegration_test_eigen, cointegration_test_trace,
                           eigen_test_statistic_1, eigen_test_statistic_2, trace_test_statistic_1,
                           trace_test_statistic_2):
        """
        Creates a web application layout for the Johansen cointegration tests segment.

        :param asset_1: (str) The name of the first asset.
        :param asset_2: (str) The name of the second asset.
        :param cointegration_test_eigen: (pd.Dataframe) The dataframe for the eigen cointegration test results.
        :param cointegration_test_trace: (pd.Dataframe) The dataframe for the trace cointegration test results.
        :param eigen_test_statistic_1: (float) Test statistic of the first asset for the eigen test results.
        :param eigen_test_statistic_2: (float) Test statistic of the second asset for the eigen test results.
        :param trace_test_statistic_1: (float) Test statistic of the first asset for the trace test results.
        :param trace_test_statistic_2: (float) Test statistic of the second asset for the trace test results.
        :return: (html.Div) Div for the Johansen cointegration tests results.
        """

        # Establish the outer div
        output = html.Div(
            style={'padding-left': 50, 'padding-right': 0, 'padding-top': 20, 'padding-bottom': 50, 'margin-left': 50,
                   'margin-right': 0, 'margin-top': 0, 'margin-bottom': 0, 'backgroundColor': self.white,
                   'horizontal-align': 'center', }, children=[

                # Cointegration test results header
                html.H3(children='Cointegration tests results:',
                        style={'textAlign': 'left', 'color': self.black, 'font-family': self.header_font,
                               'font-weight': '500', 'font-size': 24, 'padding-bottom': '1%', 'padding-top': '1%',
                               'display': 'block'}),

                # Eigenvalue test div
                html.Div(style={'backgroundColor': self.white, 'display': 'inline-block', 'padding-left': 0,
                                'padding-right': 50, 'padding-top': 0, 'padding-bottom': 0, 'margin-left': 10,
                                'margin-right': 50, 'margin-top': 0, 'margin-bottom': 0, 'vertical-align': 'center',
                                'horizontal-align': 'center', 'width': '34%'}, children=[

                    html.H3(children='Eigenvalue test:',
                            style={'textAlign': 'left', 'color': self.black, 'font-family': self.header_font,
                                   'font-weight': '500', 'font-size': 22, 'padding-bottom': '2%', 'display': 'block'}),

                    # Test result interpretation
                    html.P(children=self._johansen_test_result(cointegration_test_eigen,
                                                               eigen_test_statistic_1,
                                                               eigen_test_statistic_2)[0],
                           style={'textAlign': 'left',
                                  'color': self._johansen_test_result(cointegration_test_eigen,
                                                                      eigen_test_statistic_1,
                                                                      eigen_test_statistic_2)[1],
                                  'font-family': self._johansen_test_result(cointegration_test_eigen,
                                                                            eigen_test_statistic_1,
                                                                            eigen_test_statistic_2)[2],
                                  'font-weight': '350',
                                  'font-size': 20,
                                  'padding-bottom': '2%',
                                  'padding-top': 0,
                                  'display': 'block'
                                  }
                           ),

                    # Test result dataframe
                    dash_table.DataTable(data=cointegration_test_eigen.to_dict('records'),
                                         columns=[{'id': c, 'name': c} for c in cointegration_test_eigen.columns],
                                         style_as_list_view=True,

                                         style_cell={'padding': '10px',
                                                     'backgroundColor': 'white',
                                                     'fontSize': 14,
                                                     'font-family': self.text_font},

                                         style_header={'backgroundColor': 'white',
                                                       'fontWeight': 'bold',
                                                       'fontSize': 14,
                                                       'font-family': self.header_font

                                                       }),

                    # Test statistics values
                    html.Div(style={'display': 'block'}, children=[

                        html.P(children='Test statistic value for {}: '.format(asset_1),
                               style={'textAlign': 'left',
                                      'color': self.black,
                                      'font-family': self.header_font,
                                      'font-weight': '300',
                                      'font-size': 18,
                                      'padding-bottom': 0,
                                      'display': 'inline-block'
                                      }
                               ),

                        html.P(children='⠀{}⠀'.format(round(eigen_test_statistic_1, 5)),
                               style={'textAlign': 'center',
                                      'color': self.blue,
                                      'font-family': self.header_font,
                                      'font-weight': '400',
                                      'font-size': 18,
                                      'padding-bottom': 0,
                                      'display': 'inline-block'
                                      }
                               ),
                        html.Div(children=[

                            html.P(children='Test statistic value for {}: '.format(asset_2),
                                   style={'textAlign': 'left',
                                          'color': self.black,
                                          'font-family': self.header_font,
                                          'font-weight': '300',
                                          'font-size': 18,
                                          'padding-bottom': '5%',
                                          'display': 'inline-block'
                                          }
                                   ),

                            html.P(children='⠀{}⠀'.format(round(eigen_test_statistic_2, 5)),
                                   style={'textAlign': 'center',
                                          'color': self.blue,
                                          'font-family': self.header_font,
                                          'font-weight': '400',
                                          'font-size': 18,
                                          'padding-bottom': '5%',
                                          'display': 'inline-block'
                                          }
                                   )
                        ])

                    ]),

                ]),

                # Trace test div
                html.Div(style={'backgroundColor': self.white,
                                'display': 'inline-block',
                                'padding-left': 50,
                                'padding-right': 0,
                                'padding-top': 0,
                                'padding-bottom': 0,
                                'margin-left': 0,
                                'margin-right': 0,
                                'margin-top': 0,
                                'margin-bottom': 0,
                                'vertical-align': 'center',
                                'horizontal-align': 'center',
                                'width': '34%'
                                }, children=[

                    html.H3(children='Trace test:',
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '500',
                                   'font-size': 22,
                                   'padding-bottom': '2%',
                                   'display': 'block'
                                   }
                            ),
                    # Test result interpretation
                    html.P(children=self._johansen_test_result(cointegration_test_trace,
                                                               trace_test_statistic_1,
                                                               trace_test_statistic_2)[0],
                           style={'textAlign': 'left',
                                  'color': self._johansen_test_result(cointegration_test_trace,
                                                                      trace_test_statistic_1,
                                                                      trace_test_statistic_2)[1],
                                  'font-family': self._johansen_test_result(cointegration_test_trace,
                                                                            trace_test_statistic_1,
                                                                            trace_test_statistic_2)[2],
                                  'font-size': 20,
                                  'padding-bottom': '2%',
                                  'padding-top': 0,
                                  'display': 'block'
                                  }
                           ),

                    # Test result dataframe
                    dash_table.DataTable(data=cointegration_test_trace.to_dict('records'),
                                         columns=[{'id': c, 'name': c} for c in cointegration_test_trace.columns],
                                         style_as_list_view=True,
                                         style_cell={'padding': '10px',
                                                     'backgroundColor': 'white',
                                                     'fontSize': 14,
                                                     'font-family': self.text_font},

                                         style_header={'backgroundColor': 'white',
                                                       'fontWeight': 'bold',
                                                       'fontSize': 14,
                                                       'font-family': self.header_font}),
                    # Test statistic values
                    html.Div(style={'display': 'block'}, children=[

                        html.P(children='Test statistic value for {}: '.format(asset_1),

                               style={'textAlign': 'left',
                                      'color': self.black,
                                      'font-family': self.header_font,
                                      'font-weight': '300',
                                      'font-size': 18,
                                      'padding-bottom': 0,
                                      'display': 'inline-block'
                                      }
                               ),

                        html.P(children='⠀{}⠀'.format(round(trace_test_statistic_1, 5)),

                               style={'textAlign': 'center',
                                      'color': self.blue,
                                      'font-family': self.header_font,
                                      'font-weight': '400',
                                      'font-size': 18,
                                      'padding-bottom': 0,
                                      'display': 'inline-block'
                                      }
                               ),
                        html.Div(children=[

                            html.P(children='Test statistic value for {}: '.format(asset_2),
                                   style={'textAlign': 'left',
                                          'color': self.black,
                                          'font-family': self.header_font,
                                          'font-weight': '300',
                                          'font-size': 18,
                                          'padding-bottom': '5%',
                                          'display': 'inline-block'
                                          }
                                   ),

                            html.P(children='⠀{}⠀'.format(round(trace_test_statistic_2, 5)),
                                   style={'textAlign': 'center',
                                          'color': self.blue,
                                          'font-family': self.header_font,
                                          'font-weight': '400',
                                          'font-size': 18,
                                          'padding-bottom': '5%',
                                          'display': 'inline-block'
                                          }
                                   )
                        ])

                    ])

                ])

            ])

        return output

    def _jh_div(self, asset_1, asset_2, coint_vector, portfolio_price, portfolio_return):
        """
        Creates a web application layout for the Johansen cointegrated portfolio depiction segment.

        :param asset_1: (str) The name of the first asset.
        :param asset_2: (str) The name of the second asset.
        :param coint_vector: (pd.Series) Cointegration vector.
        :param portfolio_price: (pd.Series) The series of normalized portfolio price.
        :param portfolio_return: (pd.Series) The series of portfolio returns.
        :return: (html.Div) Div for the Johansen cointegrated portfolio depiction in equation and graph form.
        """

        output = html.Div(
            style={'padding-left': 50, 'padding-right': 50, 'padding-top': 20, 'padding-bottom': 50, 'margin-left': 50,
                   'margin-right': 50, 'margin-top': 0, 'margin-bottom': 50, 'backgroundColor': self.white,
                   'horizontal-align': 'center', }, children=[

                # The cointegrated portfolio equation
                html.Div(style={'backgroundColor': self.white,
                                'padding-bottom': 20,
                                'textAlign': 'center',
                                'horizontal-align': 'center'}, children=[

                    html.H2(children='S⠀',
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '500',
                                   'font-size': 30,
                                   'padding-bottom': 30,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='=',
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '300',
                                   'font-size': 30,
                                   'padding-bottom': 30,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='⠀{}⠀'.format(round(coint_vector.iloc[0], 4)),
                            style={'textAlign': 'left',
                                   'color': self.orange,
                                   'font-family': self.header_font,
                                   'font-weight': '400',
                                   'font-size': 30,
                                   'padding-bottom': 30,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='* {} + '.format(asset_1),
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '300',
                                   'font-size': 30,
                                   'padding-bottom': 30,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='⠀{}⠀'.format(round(coint_vector.iloc[1], 4)),
                            style={'textAlign': 'left',
                                   'color': self.blue,
                                   'font-family': self.header_font,
                                   'font-weight': '400',
                                   'font-size': 30,
                                   'padding-bottom': 30,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='* {}'.format(asset_2),
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '300',
                                   'font-size': 30,
                                   'padding-bottom': 30,
                                   'display': 'inline-block'
                                   }
                            ),

                ]),

                # Outputting graph for portfolio returns and portfolio prices
                html.Div(children=[

                    dcc.Graph(id='portfolio-plot', figure=self._portfolio_plot(portfolio_price, portfolio_return),
                              style={'width': '98%',
                                     'height': '100%',
                                     'padding-left': 0,
                                     'padding-right': 0,
                                     'padding-top': 0,
                                     'padding-bottom': 50,
                                     'margin-right': 0,
                                     'margin-top': 0,
                                     'margin-bottom': 50,
                                     'vertical-align': 'top',
                                     'horizontal-align': 'center',
                                     'display': 'inline-block',
                                     'font-family': self.header_font,
                                     'font-weight': '300',
                                     'font-size': 20,
                                     'textAlign': 'center'
                                     }
                              )

                ]),

            ])

        return output

    def _eg_div(self, asset_1, asset_2, cointegration_test, test_statistic, beta, portfolio_price, portfolio_return,
                pacf_data, acf_data, residuals, qq_y_data, qq_x_data, res_data):
        """
        Creates a web application layout for the Engel-Granger cointegrated portfolio analysis.

        :param asset_1: (str) The name of the first asset.
        :param asset_2: (str) The name of the second asset.
        :param cointegration_test: (pd.Dataframe) The dataframe for the Engle-Granger cointegration test results.
        :param test_statistic: (float) Test statistic for cointegration test results.
        :param beta: (float) Cointegration coefficient.
        :param portfolio_price: (pd.Series) The series of normalized portfolio price.
        :param portfolio_return: (pd.Series) The series of portfolio returns.
        :param pacf_data: (np.array) The values of the PACF function.
        :param acf_data: (np.array) The values of the PACF function.
        :param residuals: (pd.Series) The series of the residuals of the cointegrated portfolio.
        :param qq_y_data: (np.array) Sample quantile values for the Q-Q plot.
        :param qq_x_data: (np.array) Theoretical quantile values for the Q-Q plot.
        :param res_data: (pd.DataFrame) Dataframe containing the statistical charachteristics of the residuals.
        :return: (html.Div) Div for the Engel-Granger cointegrated portfolio analysis.
        """
        output = html.Div(

            style={'padding-left': 50,
                   'padding-right': 0,
                   'padding-top': 20,
                   'padding-bottom': 50,
                   'margin-left': 50,
                   'margin-right': 0,
                   'margin-top': 0,
                   'margin-bottom': 0,
                   'backgroundColor': self.white,
                   'horizontal-align': 'center', }, children=[

                # Cointegrated portfolio equation
                html.Div(style={'backgroundColor': self.white,
                                'padding-top': 30,
                                'textAlign': 'left',
                                'horizontal-align': 'center'}, children=[

                    html.H2(children='S⠀',
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '500',
                                   'font-size': 30,
                                   'padding-bottom': 30,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='=',
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '300',
                                   'font-size': 30,
                                   'padding-bottom': 30,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='⠀{} +'.format(asset_1),
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '300',
                                   'font-size': 30,
                                   'padding-bottom': 30,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='⠀{}⠀'.format(round(beta, 4)),
                            style={'textAlign': 'left',
                                   'color': self.blue,
                                   'font-family': self.header_font,
                                   'font-weight': '400',
                                   'font-size': 30,
                                   'padding-bottom': 30,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='* {}'.format(asset_2),
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '300',
                                   'font-size': 30,
                                   'padding-bottom': 30,
                                   'display': 'inline-block'
                                   }
                            ),

                ]),

                # Cointegration test result
                html.Div(children=[

                    html.Div(style={'backgroundColor': self.white,
                                    'display': 'inline-block',
                                    'padding-left': 0,
                                    'padding-right': 50,
                                    'padding-top': 0,
                                    'padding-bottom': 0,
                                    'margin-left': 0,
                                    'margin-right': 10,
                                    'margin-top': 0,
                                    'margin-bottom': 0,
                                    'vertical-align': 'center',
                                    'horizontal-align': 'center',
                                    'width': '22%'
                                    }, children=[

                        html.H3(children='Cointegration test results:',
                                style={'textAlign': 'left',
                                       'color': self.black,
                                       'font-family': self.header_font,
                                       'font-weight': '500',
                                       'font-size': 24,
                                       'padding-bottom': '2%',
                                       'display': 'block'
                                       }
                                ),

                        # Cointegration test result interpretation
                        html.P(children=self._adf_test_result(cointegration_test, test_statistic)[0],
                               style={'textAlign': 'left',
                                      'color': self._adf_test_result(cointegration_test, test_statistic)[1],
                                      'font-family': self._adf_test_result(cointegration_test, test_statistic)[2],
                                      'font-weight': '350',
                                      'font-size': 20,
                                      'padding-bottom': '15%',
                                      'padding-top': 0,
                                      'display': 'block'
                                      }
                               ),

                        # Cointegration test result dataframe
                        dash_table.DataTable(data=cointegration_test.to_dict('records'),
                                             columns=[{'id': c, 'name': c} for c in cointegration_test.columns],
                                             style_as_list_view=True,

                                             style_cell={'padding': '10px',
                                                         'backgroundColor': 'white',
                                                         'fontSize': 14,
                                                         'font-family': self.text_font
                                                         },

                                             style_header={'backgroundColor': 'white',
                                                           'fontWeight': 'bold',
                                                           'fontSize': 14,
                                                           'font-family': self.header_font
                                                           }
                                             ),

                        # Cointegration test statistic value
                        html.Div(style={'display': 'block'}, children=[

                            html.P(children='Test statistic value: ',
                                   style={'textAlign': 'left',
                                          'color': self.black,
                                          'font-family': self.header_font,
                                          'font-weight': '300',
                                          'font-size': 18,
                                          'padding-bottom': '10%',
                                          'display': 'inline-block'
                                          }
                                   ),

                            html.P(children='⠀{}⠀'.format(round(test_statistic, 5)),
                                   style={'textAlign': 'center',
                                          'color': self.blue,
                                          'font-family': self.header_font,
                                          'font-weight': '400',
                                          'font-size': 18,
                                          'padding-bottom': '10%',
                                          'display': 'inline-block'
                                          }
                                   ),

                        ]),

                    ]),

                    # Cointegrated portfoilo price and returns
                    dcc.Graph(id='portfolio-plot', figure=self._portfolio_plot(portfolio_price, portfolio_return),
                              style={'width': '68%',
                                     'height': '100%',
                                     'padding-left': 50,
                                     'padding-right': 0,
                                     'padding-top': 0,
                                     'padding-bottom': 0,
                                     'margin-right': 0,
                                     'margin-top': 0,
                                     'margin-bottom': 0,
                                     'vertical-align': 'top',
                                     'horizontal-align': 'right',
                                     'display': 'inline-block',
                                     'font-family': self.header_font,
                                     'font-weight': '300',
                                     'font-size': 20
                                     }
                              )

                ]),

                # Residuals analysis
                html.Div(id='resid', children=[

                    html.Div(style={'backgroundColor': self.white,
                                    'display': 'inline-block',
                                    'padding-left': 0,
                                    'padding-right': 0,
                                    'padding-top': 0,
                                    'padding-bottom': 0,
                                    'margin-left': 0,
                                    'margin-right': 0,
                                    'margin-top': 0,
                                    'margin-bottom': 0,
                                    'vertical-align': 'center',
                                    'horizontal-align': 'center',
                                    'width': '100%'
                                    }, children=[

                        html.H3(children='Residuals analysis results:',
                                style={'textAlign': 'left',
                                       'color': self.black,
                                       'font-family': self.header_font,
                                       'font-weight': '500',
                                       'font-size': 22,
                                       'padding-bottom': '2%',
                                       'display': 'block'
                                       }
                                ),
                        html.Div(
                            style={'backgroundColor': self.white,
                                   'display': 'inline-block',
                                   'padding-left': 0,
                                   'padding-right': 20,
                                   'padding-top': 0,
                                   'padding-bottom': 0,
                                   'margin-left': 0,
                                   'margin-right': 0,
                                   'margin-top': 0,
                                   'margin-bottom': 0,
                                   'vertical-align': 'top',
                                   'horizontal-align': 'center',
                                   'width': '30%'
                                   }, children=[

                                # Statistical characteristics
                                dash_table.DataTable(data=res_data.to_dict('records'),
                                                     columns=[{'id': c, 'name': c} for c in res_data.columns],
                                                     style_as_list_view=True,

                                                     style_cell={'padding': '10px',
                                                                 'backgroundColor': self.white,
                                                                 'fontSize': 14,
                                                                 'font-family': self.text_font,
                                                                 'textAlign': 'left'
                                                                 },

                                                     style_header={'padding': '15px',
                                                                   'backgroundColor': self.light_grey,
                                                                   'fontSize': 18,
                                                                   'font-family': self.header_font,
                                                                   'textAlign': 'left'
                                                                   }
                                                     ),
                                # Residuals plot
                                dcc.Graph(id='residuals-plot', figure=self._residuals_plot(residuals),
                                          style={'width': '100%',
                                                 'height': '100%',
                                                 'padding-left': 0,
                                                 'padding-right': 0,
                                                 'padding-top': 50,
                                                 'padding-bottom': 0,
                                                 'margin-right': 0,
                                                 'margin-top': 0,
                                                 'margin-bottom': 0,
                                                 'vertical-align': 'top',
                                                 'horizontal-align': 'right',
                                                 'display': 'block',
                                                 'font-family': self.header_font,
                                                 'font-weight': '300',
                                                 'font-size': 20
                                                 }
                                          ),

                            ]),

                        # PACF and ACF results plot
                        html.Div(style={'backgroundColor': self.white,
                                        'display': 'inline-block',
                                        'width': '30%',
                                        'vertical-align': 'center',
                                        'horizontal-align': 'center'
                                        }, children=[

                            dcc.Graph(id='pacf_data-plot', figure=self._pacf_plot(pacf_data),
                                      style={'width': '100%',
                                             'height': '100%',
                                             'padding-left': 0,
                                             'padding-right': 0,
                                             'padding-top': 0,
                                             'padding-bottom': 50,
                                             'margin-right': 0,
                                             'margin-top': 0,
                                             'margin-bottom': 0,
                                             'vertical-align': 'top',
                                             'horizontal-align': 'right',
                                             'display': 'block',
                                             'font-family': self.header_font,
                                             'font-weight': '300',
                                             'font-size': 20
                                             }
                                      ),

                            dcc.Graph(id='acf_data-plot', figure=self._acf_plot(acf_data),
                                      style={'width': '100%',
                                             'height': '100%',
                                             'padding-left': 0,
                                             'padding-right': 0,
                                             'padding-top': 0,
                                             'padding-bottom': 50,
                                             'margin-right': 0,
                                             'margin-top': 0,
                                             'margin-bottom': 0,
                                             'vertical-align': 'top',
                                             'horizontal-align': 'right',
                                             'display': 'block',
                                             'font-family': self.header_font,
                                             'font-weight': '300',
                                             'font-size': 20
                                             }
                                      )
                        ]),

                        # Q-Q plot
                        html.Div(style={'backgroundColor': self.white,
                                        'display': 'inline-block',
                                        'padding-left': 0,
                                        'padding-right': 0,
                                        'padding-top': 0,
                                        'padding-bottom': 0,
                                        'margin-left': 0,
                                        'margin-right': 0,
                                        'margin-top': 0,
                                        'margin-bottom': 0,
                                        'vertical-align': 'center',
                                        'horizontal-align': 'center',
                                        'width': '30%'},
                                 children=[

                                     dcc.Graph(id='qq-plot', figure=self._qq_plot(qq_y_data, qq_x_data),
                                               style={'width': '100%',
                                                      'height': '100%',
                                                      'padding-left': 10,
                                                      'padding-right': 0, 'padding-top': 0,
                                                      'padding-bottom': 50,
                                                      'margin-right': 0,
                                                      'margin-top': 0,
                                                      'margin-bottom': 0,
                                                      'vertical-align': 'top',
                                                      'horizontal-align': 'right',
                                                      'display': 'block',
                                                      'font-family': self.header_font,
                                                      'font-weight': '300',
                                                      'font-size': 20
                                                      }
                                               )

                                 ])
                    ])
                ])
            ])

        return output

    def cointegration_tearsheet(self, data, app_display='default'):
        """
        Creates a web application that visualizes the results of the cointegration analysis of the provided pair of
        assets. The mentioned pair is subjected to Engle-Granger and Johansen tests and residual analysis if possible.

        Engle-Granger analysis is provided for both combinations of assets:
         asset_1 = b_1 * asset_2 and asset_2 = b_2 * asset_1.

        Johansen analysis results are also available for both found cointegration vectors.

        :param data: (pd.Dataframe) A dataframe of two asset prices with asset names as the names of the columns.
        :param app_display: (str) Parameter that signifies whether to open a web app in a separate tab or inside
            the jupyter notebook ['default' or 'jupyter'].
        :return: (Dash) The Dash app object, which can be run using run_server.
        """

        # Assigning the data attribute
        self.data = data

        # Setting the parameter that refers to the first asset combination
        data_1 = self.data

        # Getting the basic data on both provided assets
        name_1, norm_price_asset_1, adf_asset_1, test_stat_1,\
        name_2, norm_price_asset_2, adf_asset_2, test_stat_2 = self._get_basic_assets_data()

        # Setting the parameter that refers to the second asset combination
        data_2 = self.data[[name_2, name_1]]

        # Getting the Engle-Granger analysis results for the first combination of assets
        eg_adf_dataframe_1, eg_adf_test_stat_1, eg_cointegration_vector_1,\
        eg_portfolio_returns_1, eg_portfolio_price_1, eg_residuals_1, \
        eg_residuals_dataframe_1, eg_qq_plot_y_1, eg_q_plot_x_1, \
        eg_pacf_result_1, eg_acf_result_1 = self._get_engle_granger_data(data_1)

        # Getting the Engle-Granger analysis results for the second combination of assets
        eg_adf_dataframe_2, eg_adf_test_stat_2, eg_cointegration_vector_2, \
        eg_portfolio_returns_2, eg_portfolio_price_2, eg_residuals_2, \
        eg_residuals_dataframe_2, eg_q_plot_y_2, eg_q_plot_x_2, \
        eg_pacf_result_2, eg_acf_result_2 = self._get_engle_granger_data(data_2)

        # Getting the Johansen analysis results for both cointegration vectors
        j_test_eigen_dataframe, j_test_trace_dataframe, j_eigen_test_statistic_1, \
        j_eigen_test_statistic_2, j_trace_test_statistic_1, j_trace_test_statistic_2, \
        j_cointegration_vector_1, j_cointegration_vector_2, j_portfolio_returns_1,\
        j_portfolio_price_1, j_portfolio_returns_2, j_portfolio_price_2 = self._get_johansen_data()

        app = self._get_app(app_display)

        app.layout = html.Div(style={'backgroundColor': self.light_grey, 'padding-bottom': 30}, children=[

            # Adding the ArbitrageLab logo
            html.Img(src='https://hudsonthames.org/wp-content/uploads/2021/03/Asset-7.png',
                     style={'width': '20%',
                            'height': '20%',
                            'padding-top': 50,
                            'display': 'block',
                            'margin-left': 'auto',
                            'margin-right': 'auto'
                            }
                     ),

            html.H1(children='COINTEGRATION ANALYSIS OF {}/{}'.format(name_1, name_2),
                    style={'textAlign': 'center',
                           'color': self.black,
                           'font-family': self.header_font,
                           'font-weight': '300',
                           'font-size': 50,
                           'padding-top': 30,
                           'padding-left': 50,
                           'margin-top': 30,
                           'margin-left': 'auto',
                           'margin-right': 'auto'}),

            # The characteristics of teh two assets
            html.Div(style={'margin-left': '5%',
                            'margin-right': '5%',
                            'margin-top': '5%',
                            'margin-bottom': '5%',
                            'backgroundColor': self.white,
                            'horizontal-align': 'center'
                            }, children=[

                # ADF test results for the respective asset
                html.Div(style={'backgroundColor': self.white,
                                'display': 'inline-block',
                                'padding-left': 30,
                                'padding-right': 50,
                                'padding-top': 60,
                                'padding-bottom': 50,
                                'margin-left': 50,
                                'margin-right': 50,
                                'margin-top': 0,
                                'margin-bottom': 50,
                                'vertical-align': 'center',
                                'horizontal-align': 'center',
                                'height': 315,
                                'width': '20%'
                                }, children=[

                    html.H2(children='The results of the ADF tests:',
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '500',
                                   'font-size': 24,
                                   'padding-bottom': '20%'
                                   }
                            ),

                    html.P(children='{}'.format(name_1),
                           style={'textAlign': 'left',
                                  'color': self.blue,
                                  'font-family': self.header_font,
                                  'font-weight': '500',
                                  'font-size': 20
                                  }
                           ),

                    html.P(children=self._adf_test_result(adf_asset_1, test_stat_1)[0],
                           style={'textAlign': 'left',
                                  'color': self._adf_test_result(adf_asset_1, test_stat_1)[1],
                                  'font-family': self._adf_test_result(adf_asset_1, test_stat_1)[2],
                                  'font-weight': '300',
                                  'font-size': 18,
                                  'padding-bottom': 10}),

                    html.P(children='{}'.format(name_2),
                           style={'textAlign': 'left',
                                  'color': self.blue,
                                  'font-family': self.header_font,
                                  'font-weight': '500',
                                  'font-size': 20
                                  }
                           ),

                    html.P(children=self._adf_test_result(adf_asset_2, test_stat_2)[0],
                           style={'textAlign': 'left',
                                  'color': self._adf_test_result(adf_asset_2, test_stat_2)[1],
                                  'font-family': self._adf_test_result(adf_asset_2, test_stat_2)[2],
                                  'font-weight': '300',
                                  'font-size': 18
                                  }
                           ),

                ]),

                # Plot normalized asset prices
                dcc.Graph(id='asset-prices',
                          figure=self._asset_prices_plot(norm_price_asset_1, norm_price_asset_2, name_1, name_2),
                          style={'width': '60%',
                                 'height': '100%',
                                 'padding-left': 50,
                                 'padding-top': 50,
                                 'padding-bottom': 30,
                                 'margin-top': 0,
                                 'margin-bottom': 0,
                                 'vertical-align': 'top',
                                 'horizontal-align': 'right',
                                 'display': 'inline-block',
                                 'font-family': self.header_font,
                                 'font-weight': '300',
                                 'font-size': 24,

                                 }
                          )

            ]),

            # Engle-Granger cointegration analysis
            html.Div(style={'margin-left': '5%',
                            'margin-right': '5%',
                            'margin-top': '5%',
                            'margin-bottom': '5%',
                            'padding-left': 0,
                            'padding-right': 0,
                            'padding-top': 0,
                            'padding-bottom': 0,
                            'backgroundColor': self.white,
                            'horizontal-align': 'center'
                            }, children=[

                html.H2(children='ENGLE-GRANGER APPROACH',
                        style={'textAlign': 'left',
                               'color': self.black,
                               'font-family': self.header_font,
                               'font-weight': '300',
                               'font-size': 40,
                               'padding-left': 50,
                               'padding-right': 50,
                               'padding-top': 60,
                               'padding-bottom': 0,
                               'margin-left': 50,
                               'margin-right': 50,
                               'margin-top': 0,
                               'margin-bottom': 0,
                               }
                        ),

                # Buttons for each of the respective asset combination
                html.Div(style={'display': 'block',
                                'padding-left': 50,
                                'padding-right': 50,
                                'padding-top': 60,
                                'padding-bottom': 0,
                                'margin-left': 50,
                                'margin-right': 50,
                                'margin-top': 0,
                                'margin-bottom': 0,
                                'vertical-align': 'center',
                                'horizontal-align': 'right'
                                }, children=[

                    html.Button('{}/{}'.format(name_1, name_2), id='button-1', n_clicks=0,
                                style={'BackgroundColor': self.white,
                                       'padding-left': 40,
                                       'padding-right': 40,
                                       'padding-top': 20,
                                       'padding-bottom': 20,
                                       'margin-left': 0,
                                       'margin-right': 25,
                                       'margin-top': 0,
                                       'margin-bottom': 0,
                                       'font-family': self.header_font,
                                       'font-weight': '300',
                                       'font-size': 20
                                       }
                                ),

                    html.Button('{}/{}'.format(name_2, name_1), id='button-2', n_clicks=0,
                                style={'BackgroundColor': self.white,
                                       'padding-left': 40,
                                       'padding-right': 40,
                                       'padding-top': 20,
                                       'padding-bottom': 20,
                                       'margin-left': 25,
                                       'margin-right': 40,
                                       'margin-top': 0,
                                       'margin-bottom': 0,
                                       'font-family': self.header_font,
                                       'font-weight': '300',
                                       'font-size': 20})]),

                # Engle-Granger analysis div varying relatively to the chosen asset combination
                html.Div(id='container-button'),

            ]),

            # Johansen cointegration analysis
            html.Div(style={'margin-left': '5%',
                            'margin-right': '5%',
                            'margin-top': '5%',
                            'margin-bottom': '5%',
                            'padding-left': 0,
                            'padding-right': 0,
                            'padding-top': 0,
                            'padding-bottom': 0,
                            'backgroundColor': self.white,
                            'horizontal-align': 'center'}, children=[

                html.H2(children='JOHANSEN APPROACH',
                        style={'textAlign': 'left',
                               'color': self.black,
                               'font-family': self.header_font,
                               'font-weight': '300',
                               'font-size': 40,
                               'padding-left': 50,
                               'padding-right': 50,
                               'padding-top': 60,
                               'padding-bottom': 0,
                               'margin-left': 50,
                               'margin-right': 50,
                               'margin-top': 0,
                               'margin-bottom': 0}),

                html.Div(style={'horizontal-align': 'center'}, children=[

                    # Johansen cointegration tests
                    html.Div(style={'display': 'inline-block', 'horizontal-align': 'center'}, children=[
                        self._jh_coint_test_div(name_1, name_2, j_test_eigen_dataframe, j_test_trace_dataframe,
                                                j_eigen_test_statistic_1, j_eigen_test_statistic_2,
                                                j_trace_test_statistic_1, j_trace_test_statistic_2)]),
                    # Buttons representative of the coise of the respective cointegration vector
                    html.Div(

                        style={'display': 'inline-block',
                               'padding-left': '5%',
                               'padding-right': 0,
                               'padding-top': 0,
                               'padding-bottom': '5%',
                               'margin-left': 10,
                               'margin-right': 0,
                               'margin-top': 0,
                               'margin-bottom': 0,
                               'vertical-align': 'bottom',
                               'width': '25%',
                               'height': '100%',
                               'horizontal-align': 'center'
                               }, children=[

                            html.Div(style={'display': 'block'}, children=[

                                html.Button('Cointegration vector 1', id='button-3', n_clicks=0,
                                            style={'BackgroundColor': self.white,
                                                   'padding-left': 40,
                                                   'padding-right': 40,
                                                   'padding-top': 20,
                                                   'padding-bottom': 20,
                                                   'margin-left': 50,
                                                   'margin-right': 50,
                                                   'margin-top': 50,
                                                   'margin-bottom': 50,
                                                   'font-family': self.header_font,
                                                   'font-weight': '300',
                                                   'font-size': 20,
                                                   'vertical-align': 'center',
                                                   'horizontal-align': 'center'
                                                   }
                                            )
                            ]),

                            html.Div(style={'display': 'block', 'padding-bottom': "5%", }, children=[

                                html.Button('Cointegration vector 2', id='button-4', n_clicks=0,
                                            style={'BackgroundColor': self.white,
                                                   'padding-left': 40,
                                                   'padding-right': 40,
                                                   'padding-top': 20,
                                                   'padding-bottom': 20,
                                                   'margin-left': 50,
                                                   'margin-right': 50,
                                                   'margin-top': 50,
                                                   'margin-bottom': 50,
                                                   'font-family': self.header_font,
                                                   'font-weight': '300',
                                                   'font-size': 20
                                                   }
                                            )
                            ])

                        ])
                ]),

                # Cointegrated portfolio visualization depending on chosen cointegration vector
                html.Div(id='johansen_container')
            ])
        ])

        # Defining a callback for the Engle-Granger analysis buttons
        @app.callback(Output('container-button', 'children'),
                      Input('button-1', 'n_clicks'),
                      Input('button-2', 'n_clicks'))
        def displayclick(btn1, btn2):  # pragma: no cover
            """
            Returns the Engle-Granger analysis div depending on what asset combination was chosen by pressing
            a respective button.

            :param btn1: (int) Prop parameter that represents the first button.
            :param btn2: (int) Prop parameter that represents the second button.
            :return: (html.Div) Engle-Granger analysis div.
            """
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'button-1' in changed_id:
                # Assigning the values that correspond to the first asset combination
                asset_1 = name_1
                asset_2 = name_2
                coint_test = eg_adf_dataframe_1
                test_statistic = eg_adf_test_stat_1
                beta = eg_cointegration_vector_1.loc[0][1]
                portfolio_price = eg_portfolio_price_1
                portfolio_return = eg_portfolio_returns_1
                pacf_data = eg_pacf_result_1
                acf_data = eg_acf_result_1
                residuals = eg_residuals_1
                qq_data = eg_qq_plot_y_1
                x_data = eg_q_plot_x_1
                res_data = eg_residuals_dataframe_1

            elif 'button-2' in changed_id:
                # Assigning the values that correspond to the second asset combination
                asset_1 = name_2
                asset_2 = name_1
                coint_test = eg_adf_dataframe_2
                test_statistic = eg_adf_test_stat_2
                beta = eg_cointegration_vector_2.loc[0][1]
                portfolio_price = eg_portfolio_price_2
                portfolio_return = eg_portfolio_returns_2
                pacf_data = eg_pacf_result_2
                acf_data = eg_acf_result_2
                residuals = eg_residuals_2
                qq_data = eg_q_plot_y_2
                x_data = eg_q_plot_x_2
                res_data = eg_residuals_dataframe_2

            else:
                # Assigning the values that correspond to the first asset combination
                asset_1 = name_1
                asset_2 = name_2
                coint_test = eg_adf_dataframe_1
                test_statistic = eg_adf_test_stat_1
                beta = eg_cointegration_vector_1.loc[0][1]
                portfolio_price = eg_portfolio_price_1
                portfolio_return = eg_portfolio_returns_1
                pacf_data = eg_pacf_result_1
                acf_data = eg_acf_result_1
                residuals = eg_residuals_1
                qq_data = eg_qq_plot_y_1
                x_data = eg_q_plot_x_1
                res_data = eg_residuals_dataframe_1

            # Creating the div with obtained data
            output = self._eg_div(asset_1, asset_2, coint_test, test_statistic,
                                  beta, portfolio_price, portfolio_return,
                                  pacf_data, acf_data, residuals, qq_data, x_data, res_data)
            return output

        # Defining a callback for the Johansen analysis buttons
        @app.callback(Output('johansen_container', 'children'),
                      Input('button-3', 'n_clicks'),
                      Input('button-4', 'n_clicks'))
        def displayClick(btn1, btn2):  # pragma: no cover
            """
            Returns the Johansen cointegrated portfolio div depending on what cointegration vector was chosen by
            pressing the respective button.

            :param btn1: (int) Prop parameter that represents the first button.
            :param btn2: (int) Prop parameter that represents the second button.
            :return: (html.Div) Johansen cointegrated portfolio div.
            """
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'button-3' in changed_id:
                # Assigning the values that correspond to the first cointegration vector
                asset_1 = name_1
                asset_2 = name_2
                coint_vector = j_cointegration_vector_1
                portfolio_price = j_portfolio_price_1
                portfolio_return = j_portfolio_returns_1

            elif 'button-4' in changed_id:
                # Assigning the values that correspond to the second cointegration vector
                asset_1 = name_1
                asset_2 = name_2
                coint_vector = j_cointegration_vector_2
                portfolio_price = j_portfolio_price_2
                portfolio_return = j_portfolio_returns_2

            else:
                # Assigning the values that correspond to the first cointegration vector
                asset_1 = name_1
                asset_2 = name_2
                coint_vector = j_cointegration_vector_1
                portfolio_price = j_portfolio_price_1
                portfolio_return = j_portfolio_returns_1

            # Creating the div with obtained data
            output = self._jh_div(asset_1, asset_2, coint_vector, portfolio_price, portfolio_return)

            return output

        return app

    @staticmethod
    def _spread_analysis(model):
        """
        Consolidates all the characteristics of the OU process fitted to a mean-reverting spread such as: dataframe
        containing models statistical characteristics (mean-reversion speed, long-term mean, standard deviation,
         max log-likelihood), normalized spread price, simulated OU-process with the same statistical characteristics.

        :param model: (OrnsteinUhlenbeck) OU model fitted to the optimized portfolio created from given data.
        :return: (tuple) Consolidated data connected to the fitted OU process.
        """

        # Get the asset names
        asset_1 = model.data.columns[0]
        asset_2 = model.data.columns[1]

        # Assigning b coefficient
        b = model.B_value

        # Getting the model statistical parameters
        mu = model.mu

        theta = model.theta

        sigma = model.sigma_square ** (1/2)

        mll = model.mll

        # Creating a representative dataframe for models parameters
        spread_dataframe = pd.DataFrame(data={
            'Characteristic': ['Mean-reversion speed', 'Long-term mean', 'Standard deviation', 'Max log-likelihood'],
            'Value': [round(mu, 5), round(theta, 5), round(sigma, 5), round(mll, 5)]})

        # Calculating coefficients for the portfolio price calculation
        alpha = 1 / model.data[asset_1][0]
        beta = b / model.data[asset_2][0]

        # Calculating the spread price
        spread_price = alpha * model.data[asset_1] - beta * model.data[asset_2]

        # Simulating an OU process with the same parameters as the fitted model
        ou_modelled_process = model.ou_model_simulation(model.data.shape[0])

        # Combining the results
        output = (spread_dataframe, spread_price, ou_modelled_process)

        return output

    def _ou_optimal_plot(self, data, spread_price, b=None, d=None, b_sl=None, d_sl=None):
        """
        Creates a plot of the spread alongside the calculated optimal levels.

        :param data: (pd.DataFrame) A dataframe of asset prices.
        :param spread_price: (pd.Series) A series of spread prices.
        :param b: (float) An optimal liquidation level.
        :param d: (float) An optimal entry level.
        :param b_sl: (float) An optimal liquidation level that accounts for stop-loss.
        :param d_sl: (float) An optimal entry level that accounts for stop-loss.
        :return: (go.Figure) Plot of the optimal entry/exit levels alongside the spread price.
        """

        # Creating a figure
        asset_prices = go.Figure()

        # Add the spread price plot
        asset_prices.add_trace(
            go.Scatter(x=data.index, y=spread_price, mode='lines', line=dict(color='grey'), name='Spread price'))

        if b_sl is not None:

            # Plotting optimal levels calculated with regards to stop-loss
            asset_prices.add_shape(type='line',
                                   x0=data.index[0],
                                   y0=b_sl,
                                   x1=data.index[-1],
                                   y1=b_sl,
                                   line=dict(color=self.blue, dash='dash'))
            asset_prices.add_shape(type='line',
                                   x0=data.index[0],
                                   y0=d_sl[0],
                                   x1=data.index[-1],
                                   y1=d_sl[0],
                                   line=dict(color=self.orange, dash='dash'))
            asset_prices.add_shape(type='line',
                                   x0=data.index[0],
                                   y0=d_sl[1],
                                   x1=data.index[-1],
                                   y1=d_sl[1],
                                   line=dict(color=self.orange, dash='dash'))
        else:
            # Plotting optimal levels
            asset_prices.add_shape(type='line',
                                   x0=data.index[0],
                                   y0=b,
                                   x1=data.index[-1],
                                   y1=b,
                                   line=dict(color=self.blue, dash='dash'))
            asset_prices.add_shape(type='line',
                                   x0=data.index[0],
                                   y0=d,
                                   x1=data.index[-1],
                                   y1=d,
                                   line=dict(color=self.orange, dash='dash'))

        # Updating the general characteristics
        asset_prices.update_layout(legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.80),
                                   title="Spread price",
                                   xaxis_title="Date",
                                   yaxis_title="Price",
                                   font_family=self.header_font, font_size=18,
                                   height=400,
                                   hovermode='x unified', margin=dict(l=30, r=30, t=60, b=30),
                                   showlegend=True)

        asset_prices.update_xaxes(rangeslider_visible=True)

        return asset_prices

    def _ou_asset_prices_plot(self, norm_1, norm_2, name_1, name_2):
        """
        Creates a plot of a normalized price series of a pair of assets used to create an optimal spread.

        :param norm_1: (pd.Series) Normalized price series of the first asset.
        :param norm_2: (pd.Series) Normalized price series of the second asset.
        :param name_1: (str) The name of the first asset.
        :param name_2: (str) The name of the second asset.
        :return: (go.Figure) Plot of the normalized prices.
        """

        asset_prices = go.Figure()
        asset_prices.add_trace(
            go.Scatter(x=self.data.index,
                       y=norm_1,
                       mode='lines',
                       line=dict(color=self.blue),
                       name=name_1))
        asset_prices.add_trace(
            go.Scatter(x=self.data.index,
                       y=norm_2,
                       mode='lines',
                       line=dict(color=self.orange),
                       name=name_2))
        asset_prices.update_layout(legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.90),
                                   title="Normalized asset prices",
                                   xaxis_title="Date",
                                   yaxis_title="Price",
                                   font_family=self.header_font,
                                   font_size=18,
                                   height=400,
                                   hovermode='x unified',
                                   margin=dict(l=30, r=30, t=60, b=30))

        asset_prices.update_xaxes(rangeslider_visible=True)

        return asset_prices

    def _ou_spread_plot(self, spread_price, ou_modelled_process):
        """
        Creates a plot of a normalized price series of a pair of assets used to create an optimal spread.

        :param spread_price: (pd.Series) Spread price series.
        :param ou_modelled_process: (pd.Series) Simulated ou process that possesses the same statistical
            characteristics as the fitted model.
        :return: (go.Figure) Plot of the spread prices and simulated OU process.
        """

        # Creating a figure
        asset_prices = go.Figure()

        # Adding the spread price and simulated OU process
        asset_prices.add_trace(go.Scatter(x=self.data.index,
                                          y=spread_price,
                                          mode='lines',
                                          line=dict(color=self.blue),
                                          name='Spread price'))
        asset_prices.add_trace(go.Scatter(x=self.data.index,
                                          y=ou_modelled_process,
                                          mode='lines',
                                          line=dict(color=self.orange),
                                          name='Simulated process',
                                          visible='legendonly'))

        # Updating the general styling of the plot
        asset_prices.update_layout(legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.80),
                                   title="Spread price",
                                   xaxis_title="Date",
                                   yaxis_title="Price",
                                   font_family=self.header_font,
                                   font_size=18,
                                   height=400,
                                   hovermode='x unified',
                                   margin=dict(l=30, r=30, t=60, b=30))

        # Adding the range slider
        asset_prices.update_xaxes(rangeslider_visible=True)

        return asset_prices

    def _optimal_levels_div(self, data, spread_price, b, d, b_sl, d_sl):
        """
        Creates a web application layout for the optimal entry and exit levels alongside with the spread price.

        :param data: (pd.DataFrame) A dataframe of two assets.
        :param spread_price: (pd.Series) The series of spread price.
        :param b: (float) An optimal liquidation level.
        :param d: (float) An optimal entry level.
        :param b_sl: (float) An optimal liquidation level that accounts for stop-loss.
        :param d_sl: (float) An optimal entry level that accounts for stop-loss.
        :return: (html.Div) Div for the optimal entry and exit levels alongside with the spread price.
        """

        output = html.Div(
            style={'padding-left': 0,
                   'padding-right': 0,
                   'padding-top': 0,
                   'padding-bottom': 0,
                   'margin-left': 0,
                   'margin-right': 0,
                   'margin-top': 0,
                   'margin-bottom': 0,
                   'backgroundColor': self.white,
                   'horizontal-align': 'center'
                   }, children=[

                html.Div(children=[
                    # Adding an optimal levels plot
                    dcc.Graph(id='portfolio-plot',
                              figure=self._ou_optimal_plot(data, spread_price, b=b, d=d, b_sl=b_sl, d_sl=d_sl),
                              style={'width': '98%',
                                     'height': '100%',
                                     'padding-left': 0, 'padding-right': 0,
                                     'padding-top': 0,
                                     'padding-bottom': 0,
                                     'margin-right': 0,
                                     'margin-top': 0,
                                     'margin-bottom': 0,
                                     'vertical-align': 'top',
                                     'horizontal-align': 'center',
                                     'display': 'inline-block',
                                     'font-family': self.header_font,
                                     'font-weight': '300',
                                     'font-size': 20,
                                     'textAlign': 'center'
                                     }
                              )

                ])

            ])

        return output

    def _optimal_levels_error_div(self):
        """
        Creates a web application layout for the error message.

        :return: (html.Div) Div for the error message in case there is no optimal solution.
        """

        output = html.Div(
            style={'padding-left': 0,
                   'padding-right': 0,
                   'padding-top': '10%',
                   'padding-bottom': 0,
                   'margin-left': 0,
                   'margin-right': 0,
                   'margin-top': 0,
                   'margin-bottom': 0,
                   'backgroundColor': self.white,
                   'horizontal-align': 'center'
                   }, children=[

                html.Div(children=[
                    # Adding the warning
                    html.H2(children='WARNING:',
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '400',
                                   'font-size': 30,
                                   'padding-bottom': 0,
                                   'display': 'block'
                                   }
                            ),

                    html.H2(children='The optimal solution doesn\'t exist for a given set of parameters.',
                            style={'textAlign': 'left',
                                   'color': self.orange,
                                   'font-family': self.header_font,
                                   'font-weight': '300',
                                   'font-size': 30,
                                   'padding-bottom': 0,
                                   'display': 'block'
                                   }
                            ),
                    html.H2(children='To improve the situation please adjust the values of model parameters.',
                            style={'textAlign': 'left',
                                   'color': self.orange,
                                   'font-family': self.header_font,
                                   'font-weight': '300',
                                   'font-size': 30,
                                   'padding-bottom': 0,
                                   'display': 'block'
                                   }
                            ),

                ])

            ])

        return output

    def _ou_div(self, spread_price, ou_modelled_process, spread_dataframe, model, cointegration_test, test_statistic,
                norm_1, norm_2, name_1, name_2):
        """
        Creates a web application layout for the OU-model analysis and optimal portfolio creation.

        :param spread_price: (pd.Series) The series of an optimal spread portfolio price.
        :param ou_modelled_process: (pd.Series) The series of simulated OU process.
        :param spread_dataframe: (pd.DataFrame) The dataframe of OU model parameters.
        :param model: (OrnsteinUhlenbeck) OU model.
        :param cointegration_test: (pd.Dataframe) The dataframe for the Engle-Granger cointegration test results.
        :param test_statistic: (float) Test statistic for cointegration test results.
        :param norm_1: (pd.Series) Normalized price of the first asset.
        :param norm_2: (pd.Series) Normalized price of the second asset.
        :param name_1: (str) The name of the first asset.
        :param name_2: (str) The name of the second asset.
        :return: (html.Div) Div for the OU optimal portfolio analysis.
        """

        # Getting the asset names
        asset_1 = model.data.columns[0]
        asset_2 = model.data.columns[1]

        # Getting the initial coefficient value
        b = model.B_value

        output = html.Div(
            style={'padding-left': 50,
                   'padding-right': 50,
                   'padding-top': 20,
                   'padding-bottom': 50,
                   'margin-left': 50,
                   'margin-right': 50,
                   'margin-top': 0,
                   'margin-bottom': 50,
                   'backgroundColor': self.white,
                   'horizontal-align': 'center'
                   }, children=[

                # The optimal spread equation
                html.Div(style={'backgroundColor': self.white,
                                'padding-bottom': 0,
                                'textAlign': 'left',
                                'horizontal-align': 'center'
                                }, children=[

                    html.H2(children='S⠀',
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '500',
                                   'font-size': 30,
                                   'padding-bottom': 0,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='=',
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '300',
                                   'font-size': 30,
                                   'padding-bottom': 0,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='⠀{} - '.format(asset_1),
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '300',
                                   'font-size': 30,
                                   'padding-bottom': 0,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='⠀{}⠀'.format(round(b, 4)),
                            style={'textAlign': 'left',
                                   'color': self.blue,
                                   'font-family': self.header_font,
                                   'font-weight': '400',
                                   'font-size': 30,
                                   'padding-bottom': 0,
                                   'display': 'inline-block'
                                   }
                            ),

                    html.H2(children='* {}'.format(asset_2),
                            style={'textAlign': 'left',
                                   'color': self.black,
                                   'font-family': self.header_font,
                                   'font-weight': '300',
                                   'font-size': 30,
                                   'padding-bottom': 0,
                                   'display': 'inline-block'
                                   }
                            ),

                ]),

                # Cointegration test results
                html.Div(children=[

                    html.Div(style={'backgroundColor': self.white,
                                    'display': 'inline-block',
                                    'padding-left': 0,
                                    'padding-right': 0,
                                    'padding-top': 0,
                                    'padding-bottom': 0,
                                    'margin-left': 0,
                                    'margin-right': 0,
                                    'margin-top': 0,
                                    'margin-bottom': 0,
                                    'vertical-align': 'center',
                                    'horizontal-align': 'center',
                                    'width': '100%'
                                    }, children=[

                        html.Div(style={'backgroundColor': self.white,
                                        'display': 'inline-block',
                                        'padding-left': 0,
                                        'padding-right': 50,
                                        'padding-top': 0,
                                        'padding-bottom': 0,
                                        'margin-left': 0,
                                        'margin-right': 0,
                                        'margin-top': 0,
                                        'margin-bottom': 0,
                                        'vertical-align': 'top',
                                        'horizontal-align': 'center',
                                        'width': '30%'
                                        }, children=[

                            html.H3(children='Cointegration test results:',
                                    style={'textAlign': 'left',
                                           'color': self.black,
                                           'font-family': self.header_font,
                                           'font-weight': '500',
                                           'font-size': 24,
                                           'padding-bottom': '2%',
                                           'display': 'block'
                                           }
                                    ),
                            html.P(children=self._adf_test_result(cointegration_test, test_statistic)[0],
                                   style={'textAlign': 'left',
                                          'color': self._adf_test_result(cointegration_test, test_statistic)[1],
                                          'font-family': self._adf_test_result(cointegration_test, test_statistic)[2],
                                          'font-weight': '350',
                                          'font-size': 20,
                                          'padding-bottom': '15%',
                                          'padding-top': 0,
                                          'display': 'block'}),

                            dash_table.DataTable(data=cointegration_test.to_dict('records'),
                                                 columns=[{'id': c, 'name': c} for c in cointegration_test.columns],
                                                 style_as_list_view=True,

                                                 style_cell={'padding': '10px',
                                                             'backgroundColor': 'white',
                                                             'fontSize': 14,
                                                             'font-family': self.text_font},

                                                 style_header={'backgroundColor': 'white',
                                                               'fontWeight': 'bold',
                                                               'fontSize': 14,
                                                               'font-family': self.header_font
                                                               }
                                                 ),

                            html.Div(style={'display': 'block'}, children=[

                                html.P(children='Test statistic value: ',
                                       style={'textAlign': 'left',
                                              'color': self.black,
                                              'font-family': self.header_font,
                                              'font-weight': '300',
                                              'font-size': 18,
                                              'padding-bottom': '10%',
                                              'display': 'inline-block'
                                              }
                                       ),

                                html.P(children='⠀{}⠀'.format(round(test_statistic, 5)),
                                       style={'textAlign': 'center',
                                              'color': self.blue,
                                              'font-family': self.header_font,
                                              'font-weight': '400',
                                              'font-size': 18,
                                              'padding-bottom': '10%',
                                              'display': 'inline-block'
                                              }
                                       ),

                            ]),

                            html.H3(children='OU model characteristics:',
                                    style={'textAlign': 'left',
                                           'color': self.black,
                                           'font-family': self.header_font,
                                           'font-weight': '500',
                                           'font-size': 22,
                                           'padding-bottom': '2%',
                                           'display': 'block'
                                           }
                                    ),

                            dash_table.DataTable(data=spread_dataframe.to_dict('records'),
                                                 columns=[{'id': c, 'name': c} for c in spread_dataframe.columns],
                                                 style_as_list_view=True,

                                                 style_cell={'padding': '10px',
                                                             'backgroundColor': 'white',
                                                             'fontSize': 14,
                                                             'font-family': self.text_font,
                                                             'textAlign': 'left'},

                                                 style_header={'padding': '15px',
                                                               'backgroundColor': self.light_grey,
                                                               'fontSize': 18,
                                                               'font-family': self.header_font,
                                                               'textAlign': 'left'
                                                               }
                                                 ),

                        ]),

                        html.Div(style={'backgroundColor': self.white,
                                        'display': 'inline-block',
                                        'width': '65%',
                                        'vertical-align': 'center',
                                        'horizontal-align': 'center'
                                        }, children=[

                            dcc.Graph(id='pacf_data-plot', figure=self._ou_asset_prices_plot(norm_1, norm_2, name_1, name_2),
                                      style={'width': '100%',
                                             'height': '100%',
                                             'padding-left': 0,
                                             'padding-right': 0,
                                             'padding-top': 0,
                                             'padding-bottom': 50,
                                             'margin-right': 0,
                                             'margin-top': 0,
                                             'margin-bottom': 0,
                                             'vertical-align': 'top',
                                             'horizontal-align': 'right',
                                             'display': 'block',
                                             'font-family': self.header_font,
                                             'font-weight': '300',
                                             'font-size': 20
                                             }
                                      ),

                            dcc.Graph(id='acf_data-plot', figure=self._ou_spread_plot(spread_price, ou_modelled_process),
                                      style={'width': '100%',
                                             'height': '100%',
                                             'padding-left': 0,
                                             'padding-right': 0,
                                             'padding-top': 0,
                                             'padding-bottom': 50,
                                             'margin-right': 0,
                                             'margin-top': 0,
                                             'margin-bottom': 0,
                                             'vertical-align': 'top',
                                             'horizontal-align': 'right',
                                             'display': 'block',
                                             'font-family': self.header_font,
                                             'font-weight': '300',
                                             'font-size': 20
                                             }
                                      )
                        ])

                    ])

                ])

            ])

        return output

    def ou_tearsheet(self, data, app_display='default'):
        """
        Creates a web application that visualizes the results of the OU model analysis of the provided pair of
        assets. The mentioned pair is subjected to Engle-Granger test and optimal portfolio creation.

        OU model analysis is provided for both combinations of assets:
        asset_1 = b_1 * asset_2 and asset_2 = b_2 * asset_1.

        :param data: (pd.Dataframe) A dataframe of two asset prices with asset names as the names of the columns.
        :param app_display: (str) Parameter that signifies whether to open a web app in a separate tab or inside
            the jupyter notebook ['default' or 'jupyter'].
        :return: (Dash) The Dash app object, which can be run using run_server.
        """

        # Setting the data class attribute
        self.data = data

        # Setting the parameter that refers to the first asset combination
        data_1 = self.data

        # Getting the basic data on both provided assets
        name_1, norm_price_asset_1, _, _, name_2, norm_price_asset_2, _, _ = self._get_basic_assets_data()

        # Setting the parameter that refers to the second asset combination
        data_2 = self.data[[name_2, name_1]]

        # Getting the Engle-Granger analysis results for the first combination of assets
        eg_adf_dataframe_1, eg_adf_test_stat_1, _, _, _, _, _, _, _, _, _ = self._get_engle_granger_data(data_1)

        # Getting the Engle-Granger analysis results for the second combination of assets
        eg_adf_dataframe_2, eg_adf_test_stat_2, _, _, _, _, _, _, _, _, _ = self._get_engle_granger_data(data_2)

        # Initializing the OU models for both asset combinations
        model_1 = OrnsteinUhlenbeck()
        model_2 = OrnsteinUhlenbeck()

        # Fitting the OU models for both asset combinations
        model_1.delta_t = 1 / 252
        model_2.delta_t = 1 / 252
        model_1.fit_to_assets(data_1)
        model_2.fit_to_assets(data_2)

        # Calculating the data connected to the OU models for both asset combinations
        spread_dataframe_1, spread_price_1, ou_modelled_process_1 = self._spread_analysis(model_1)
        spread_dataframe_2, spread_price_2, ou_modelled_process_2 = self._spread_analysis(model_2)

        app = self._get_app(app_display)

        app.layout = html.Div(style={'backgroundColor': self.light_grey,
                                     'padding-bottom': 30
                                     }, children=[
            # Add the ArbitrageLab logo
            html.Img(src='https://hudsonthames.org/wp-content/uploads/2021/03/Asset-7.png',
                     style={'width': '18%',
                            'height': '18%',
                            'padding-top': 30,
                            'display': 'block',
                            'margin-left': 'auto',
                            'margin-right': 'auto'
                            }
                     ),

            html.H1(children='ORNSTEIN-UHLENBECK MODEL ANALYSIS OF {}/{} PAIR'.format(name_1, name_2),
                    style={'textAlign': 'center',
                           'color': self.black,
                           'font-family': self.header_font,
                           'font-weight': '300',
                           'font-size': 50,
                           'padding-top': 20,
                           'padding-left': 50,
                           'margin-top': 30,
                           'margin-left': 'auto',
                           'margin-right': 'auto'
                           }
                    ),

            # Establishing the buttons for both asset combinations
            html.Div(style={'margin-left': '5%',
                            'margin-right': '5%',
                            'margin-top': '3%',
                            'margin-bottom': '5%',
                            'padding-left': 0,
                            'padding-right': 0,
                            'padding-top': 0,
                            'padding-bottom': 0,
                            'backgroundColor': self.white,
                            'horizontal-align': 'center'
                            }, children=[

                html.Div(style={'display': 'block',
                                'padding-left': 50,
                                'padding-right': 50,
                                'padding-top': 60,
                                'padding-bottom': 0,
                                'margin-left': 50,
                                'margin-right': 50,
                                'margin-top': 0,
                                'margin-bottom': 0,
                                'vertical-align': 'center',
                                'horizontal-align': 'right'
                                }, children=[

                    html.Button('{}/{}'.format(name_1, name_2), id='button-1', n_clicks=0,
                                style={'BackgroundColor': self.white,
                                       'padding-left': 40,
                                       'padding-right': 40,
                                       'padding-top': 20,
                                       'padding-bottom': 20,
                                       'margin-left': 0,
                                       'margin-right': 25,
                                       'margin-top': 0,
                                       'margin-bottom': 0,
                                       'font-family': self.header_font,
                                       'font-weight': '300',
                                       'font-size': 20
                                       }
                                ),

                    html.Button('{}/{}'.format(name_2, name_1), id='button-2', n_clicks=0,
                                style={'BackgroundColor': self.white,
                                       'padding-left': 40,
                                       'padding-right': 40,
                                       'padding-top': 20,
                                       'padding-bottom': 20,
                                       'margin-left': 25,
                                       'margin-right': 40,
                                       'margin-top': 0,
                                       'margin-bottom': 0,
                                       'font-family': self.header_font,
                                       'font-weight': '300',
                                       'font-size': 20
                                       }
                                )
                ]),

                # Div visualizing the information about the OU model and optimal portfolio depending on chosen
                # asset combination
                html.Div(id='pair-container-button'),

            ]),

            # Optimal levels analysis
            html.Div(style={'margin-left': '5%',
                            'margin-right': '5%',
                            'margin-top': '5%',
                            'margin-bottom': '5%',
                            'padding-left': 0,
                            'padding-right': 0,
                            'padding-top': 0,
                            'padding-bottom': 0,
                            'backgroundColor': self.white,
                            'horizontal-align': 'center'
                            }, children=[

                html.H2(children='OPTIMAL POSITION ENTRY/EXIT ANALYSIS',
                        style={'textAlign': 'left',
                               'color': self.black,
                               'font-family': self.header_font,
                               'font-weight': '300',
                               'font-size': 40,
                               'padding-left': 50,
                               'padding-right': 50,
                               'padding-top': 60,
                               'padding-bottom': 0,
                               'margin-left': 50,
                               'margin-right': 50,
                               'margin-top': 0,
                               'margin-bottom': 0
                               }
                        ),

                html.Div(style={'horizontal-align': 'center'}, children=[

                    # Model constraints input fields
                    html.Div(style={'backgroundColor': self.white,
                                    'display': 'inline-block',
                                    'padding-left': 50,
                                    'padding-right': 60,
                                    'padding-top': 60,
                                    'padding-bottom': 20,
                                    'margin-left': 50,
                                    'margin-right': 0,
                                    'margin-top': 0,
                                    'margin-bottom': 50,
                                    'vertical-align': 'center',
                                    'horizontal-align': 'center',
                                    'width': '24%'
                                    }, children=[

                        html.H3(children='Model constraints:',
                                style={'textAlign': 'left',
                                       'color': self.black,
                                       'font-family': self.header_font,
                                       'font-weight': '500',
                                       'font-size': 24,
                                       'padding-bottom': '2%',
                                       'display': 'block'
                                       }
                                ),

                        html.Div(style={'display': 'inline-block'}, children=[

                            html.P(children='Discount rate: ',
                                   style={'textAlign': 'left',
                                          'color': self.black,
                                          'font-family': self.header_font,
                                          'font-weight': '300',
                                          'font-size': 20,
                                          'padding-top': '10%',
                                          'padding-bottom': '10%',
                                          'padding-right': '10%',
                                          'display': 'block'
                                          }
                                   ),

                            html.P(children='Transaction cost: ',
                                   style={'textAlign': 'left',
                                          'color': self.black,
                                          'font-family': self.header_font,
                                          'font-weight': '300',
                                          'font-size': 20,
                                          'padding-top': '10%',
                                          'padding-bottom': '10%',
                                          'padding-right': '10%',
                                          'display': 'block'
                                          }
                                   ),

                            html.P(children='Stop-loss level: ',
                                   style={'textAlign': 'left',
                                          'color': self.black,
                                          'font-family': self.header_font,
                                          'font-weight': '300',
                                          'font-size': 20,
                                          'padding-top': '10%',
                                          'padding-bottom': '10%',
                                          'padding-right': '10%',
                                          'display': 'block'})

                        ]),

                        html.Div(style={'display': 'inline-block', 'vertical-align': 'top'}, children=[

                            dcc.Input(id='discount-rate', type='number', value=0,
                                      style={'color': self.black,
                                             'font-family': self.header_font,
                                             'font-weight': '300',
                                             'font-size': 20,
                                             'height': 50,
                                             'width': 150,
                                             'margin-top': '10%',
                                             'margin-bottom': '10%',
                                             'vertical-align': 'top',
                                             'horizontal-align': 'center',
                                             'display': 'block',
                                             'textAlign': 'center'
                                             }),

                            dcc.Input(id='transaction-cost', type='number', value=0,
                                      style={'color': self.black,
                                             'font-family': self.header_font,
                                             'font-weight': '300',
                                             'font-size': 20,
                                             'height': 50,
                                             'width': 150,
                                             'margin-top': '15%',
                                             'margin-bottom': '15%',
                                             'vertical-align': 'top',
                                             'horizontal-align': 'center',
                                             'display': 'block',
                                             'textAlign': 'center'}),

                            dcc.Input(id='stop-loss', type='number', placeholder="None",
                                      style={'color': self.black,
                                             'font-family': self.header_font,
                                             'font-weight': '300',
                                             'font-size': 20,
                                             'height': 50,
                                             'width': 150,
                                             'margin-top': '18%',
                                             'margin-bottom': '10%',
                                             'vertical-align': 'top',
                                             'horizontal-align': 'center',
                                             'display': 'block',
                                             'textAlign': 'center'}),

                        ]),

                    ]),

                    # Optimal levels output depending on constraint input fields
                    html.Div(style={'backgroundColor': self.white,
                                    'display': 'inline-block',
                                    'padding-left': 0,
                                    'padding-right': 0,
                                    'padding-top': 50,
                                    'padding-bottom': 50,
                                    'margin-left': 0,
                                    'margin-right': 0,
                                    'margin-top': '2%',
                                    'margin-bottom': 20,
                                    'vertical-align': 'top',
                                    'horizontal-align': 'center',
                                    'width': '60%',
                                    'height': '100%'
                                    },
                             children=[html.Div(id='optimal-levels-container')
                                       ])

                ])

            ])
        ])

        @app.callback(Output('pair-container-button', 'children'), Input('button-1', 'n_clicks'),
                      Input('button-2', 'n_clicks'))
        def displayclick(btn1, btn2):  # pragma: no cover
            """
            Returns the OU model analysis div depending on what asset combination was chosen by pressing
            a respective button.

            :param btn1: (int) Prop parameter that represents the first button.
            :param btn2: (int) Prop parameter that represents the second button.
            :return: (html.Div) OU model analysis div.
            """

            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'button-1' in changed_id:
                asset_1 = name_1
                asset_2 = name_2
                coint_test = eg_adf_dataframe_1
                test_statistic = eg_adf_test_stat_1
                spread_dataframe = spread_dataframe_1
                spread_price = spread_price_1
                ou_modelled_process = ou_modelled_process_1
                model = model_1

            elif 'button-2' in changed_id:
                asset_1 = name_2
                asset_2 = name_1
                coint_test = eg_adf_dataframe_2
                test_statistic = eg_adf_test_stat_2
                spread_dataframe = spread_dataframe_2
                spread_price = spread_price_2
                ou_modelled_process = ou_modelled_process_2
                model = model_2

            else:
                asset_1 = name_1
                asset_2 = name_2
                coint_test = eg_adf_dataframe_1
                test_statistic = eg_adf_test_stat_1
                spread_dataframe = spread_dataframe_1
                spread_price = spread_price_1
                ou_modelled_process = ou_modelled_process_1
                model = model_1

            return self._ou_div(spread_price, ou_modelled_process, spread_dataframe, model, coint_test, test_statistic,
                                norm_price_asset_1, norm_price_asset_2, asset_1, asset_2)

        @app.callback(Output(component_id='optimal-levels-container', component_property='children'),
                      Input(component_id='button-1', component_property='n_clicks'),
                      Input(component_id='button-2', component_property='n_clicks'),
                      Input(component_id='discount-rate', component_property='value'),
                      Input(component_id='transaction-cost', component_property='value'),
                      Input(component_id='stop-loss', component_property='value'))
        def display(btn1, btn2, discount_rate, transaction_cost, stop_loss):  # pragma: no cover
            """
            Returns the OU model analysis div depending on what asset combination was chosen by pressing
            a respective button.

            :param btn1: (int) Prop parameter that represents the first button.
            :param btn2: (int) Prop parameter that represents the second button.
            :param discount_rate: (int) Prop parameter that represents the discount rate field input.
            :param transaction_cost: (int) Prop parameter that represents the transaction cost field input.
            :param stop_loss: (int) Prop parameter that represents the stop loss field input.
            :return: (html.Div) OU model analysis div.
            """

            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'button-1' in changed_id:
                spread_price = spread_price_1
                model = model_1
                data = data_1

            elif 'button-2' in changed_id:
                spread_price = spread_price_2
                model = model_2
                data = data_2

            else:
                spread_price = spread_price_1
                model = model_1
                data = data_1

            # Setting the model constraints parameters
            discount_rate = float(discount_rate)

            transaction_cost = float(transaction_cost)

            if not isinstance(stop_loss, type(None)):
                stop_loss = float(stop_loss)

            with warnings.catch_warnings():  # Silencing IntegrationWarnings
                warnings.filterwarnings('ignore')
                model.fit(data, data_frequency="D", discount_rate=discount_rate, transaction_cost=transaction_cost,
                          stop_loss=stop_loss, )

            b = None
            d = None
            b_sl = None
            d_sl = None

            # Checking if the optimal solution exists and returning an appropriate output
            try:
                if stop_loss is not None:
                    b_sl = model.optimal_liquidation_level_stop_loss()
                    d_sl = model.optimal_entry_interval_stop_loss()
                else:
                    b = model.optimal_liquidation_level()
                    d = model.optimal_entry_level()
            except Exception:
                output = self._optimal_levels_error_div()

            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')  # Silencing IntegrationWarnings
                    output = self._optimal_levels_div(data, spread_price, b, d, b_sl, d_sl)

            return output

        return app
