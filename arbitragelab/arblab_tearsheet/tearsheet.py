# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

import arbitragelab as al
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kurtosis, skew, shapiro
from arbitragelab.cointegration_approach import get_half_life_of_mean_reversion
from statsmodels.tsa.stattools import pacf, adfuller, acf

import dash
import dash_core_components as dcc
import dash_html_components as html
from flask import request
import dash_table
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from arbitragelab.util import devadarsh


class TearSheet:
    """

    """

    def __init__(self):

        self.data = None
        self.tear_sheet_type = None
        self.confidence_levels = None
        self.blue = '#0C9AAC'
        self.orange = '#DE612F'
        self.grey = '#949494'
        self.light_grey = '#F2F3F4'
        self.black = '#0B0D13'
        self.white = '#ffffff'
        self.oxford_blue = '#072040'

        devadarsh.track('Tearsheet')

    def create_tearsheet(self, data, tearsheet_type='cointegration'):
        """

        :param data: (pd.DataFrame)

        """

        self.data = data

        self.confidence_levels = ['Confidence level', ['99%', '95%', '90%']]

        if tearsheet_type == 'cointegration':
            self.get_cointegration_approach_tearsheet()
        # elif tearsheet_type == 'ou_model':
        #     self.get_ou_model_tearsheet()
        # elif tearsheet_type == 'pair_selection':
        #     self.get_pair_selection_tearsheet()

    def get_basic_asset_data(self):
        """

        """

        data = self.data

        asset_name_1 = data.columns[0]

        norm_asset_price_1 = (data[asset_name_1] - data[asset_name_1].min()) / (
                data[asset_name_1].max() - data[asset_name_1].min())

        asset_price_1 = pd.DataFrame(data=data[asset_name_1])

        adf_asset_1 = pd.DataFrame(data={self.confidence_levels[0]: self.confidence_levels[1],
                                         'Values': list(adfuller(asset_price_1, autolag='AIC')[4].values())})

        test_stat_1 = adfuller(asset_price_1, autolag='AIC')[0]

        asset_name_2 = data.columns[1]

        norm_asset_price_2 = (data[asset_name_2] - data[asset_name_2].min()) / (
                data[asset_name_2].max() - data[asset_name_2].min())

        asset_price_2 = pd.DataFrame(data=data[asset_name_2])

        adf_asset_2 = pd.DataFrame(data={self.confidence_levels[0]: self.confidence_levels[1],
                                         'Values': list(adfuller(asset_price_2, autolag='AIC')[4].values())})

        test_stat_2 = adfuller(asset_price_2, autolag='AIC')[0]

        output = [asset_name_1, norm_asset_price_1, adf_asset_1, test_stat_1, asset_name_2, norm_asset_price_2,
                  adf_asset_2, test_stat_2]

        return output

    @staticmethod
    def residual_analysis(residuals):
        """

        """
        standard_deviation = residuals.std()
        half_life = get_half_life_of_mean_reversion(residuals)
        skewness = skew(residuals)
        kurtosis_ = kurtosis(residuals)

        statistic, p = shapiro(residuals)

        if p > 0.05:
            shapiro_wilk = 'Passed'
        else:
            shapiro_wilk = 'Failed'

        residuals_dataframe = pd.DataFrame(data={
            'Characteristic': ['Standard Deviation', 'Half-life', 'Skewness', 'Kurtosis',
                               'Shapiro-Wilk normality test'],
            'Value': [round(standard_deviation, 5), round(half_life, 5), round(skewness, 5), round(kurtosis_, 5),
                      shapiro_wilk]})

        qq_y = stats.probplot(residuals, dist='norm', sparams=(1))
        x = np.array([qq_y[0][0][0], qq_y[0][0][-1]])

        pacf_result = pacf(residuals, nlags=20)

        acf_result = acf(residuals, nlags=20, fft=True)

        return residuals, residuals_dataframe, qq_y, x, pacf_result, acf_result

    def get_engle_granger_data(self, data):
        """

        """

        data_returns = (data / data.shift(1) - 1)[1:]

        weights = data.iloc[0] / abs(data.iloc[0]).sum()

        portfolio = al.cointegration_approach.EngleGrangerPortfolio()

        portfolio.fit(data, add_constant=True)

        adf = portfolio.adf_statistics

        adf_test_stat = adf.loc['statistic_value'][0]

        adf_dataframe = pd.DataFrame(
            data={self.confidence_levels[0]: self.confidence_levels[1], 'Values': list(adf[:-1][0].round(5))})

        cointegration_vector = portfolio.cointegration_vectors

        scaled_vector = (cointegration_vector.loc[0] / abs(cointegration_vector.loc[0]).sum())

        portfolio_returns = (data_returns * scaled_vector * weights).sum(axis=1)

        portfolio_price = (portfolio_returns + 1).cumprod()

        residuals, residuals_dataframe, qq_y, x, pacf_result, acf_result = self.residual_analysis(portfolio.residuals)

        return (adf_dataframe, adf_test_stat, cointegration_vector, portfolio_returns, portfolio_price, residuals,
                residuals_dataframe, qq_y, x, pacf_result, acf_result)

    def get_johansen_data(self):
        """

        """
        data = self.data

        asset_name_1 = data.columns[0]
        asset_name_2 = data.columns[1]

        data_returns = (data / data.shift(1) - 1)[1:]

        weights = data.iloc[0] / abs(data.iloc[0]).sum()

        portfolio = al.cointegration_approach.JohansenPortfolio()

        portfolio.fit(data, det_order=0)

        test_eigen = portfolio.johansen_eigen_statistic

        test_trace = portfolio.johansen_trace_statistic

        test_eigen_dataframe = pd.DataFrame(data={self.confidence_levels[0]: self.confidence_levels[1],
                                                  'Values for {}'.format(asset_name_1): list(
                                                      test_eigen.iloc[2::-1][asset_name_1].round(5)),
                                                  'Values for {}'.format(asset_name_2): list(
                                                      test_eigen.iloc[2::-1][asset_name_2].round(5))})
        test_trace_dataframe = pd.DataFrame(data={self.confidence_levels[0]: self.confidence_levels[1],
                                                  'Values for {}'.format(asset_name_1): list(
                                                      test_trace.iloc[2::-1][asset_name_1].round(5)),
                                                  'Values for {}'.format(asset_name_2): list(
                                                      test_trace.iloc[2::-1][asset_name_2].round(5))})

        eigen_test_statistic_1 = test_eigen[asset_name_1][-1].round(5)

        eigen_test_statistic_2 = test_eigen[asset_name_2][-1].round(5)

        trace_test_statistic_1 = test_trace[asset_name_1][-1].round(5)

        trace_test_statistic_2 = test_trace[asset_name_2][-1].round(5)

        cointegration_vector_1 = portfolio.cointegration_vectors.loc[[0]].round(5)

        cointegration_vector_2 = portfolio.cointegration_vectors.loc[[1]].round(5)

        scaled_vector_1 = (portfolio.cointegration_vectors.loc[0] / abs(portfolio.cointegration_vectors.loc[0]).sum())

        scaled_vector_2 = (portfolio.cointegration_vectors.loc[1] / abs(portfolio.cointegration_vectors.loc[1]).sum())

        portfolio_returns_1 = (data_returns * scaled_vector_1 * weights).sum(axis=1)

        portfolio_price_1 = (portfolio_returns_1 + 1).cumprod()

        portfolio_returns_2 = (data_returns * scaled_vector_2 * weights).sum(axis=1)

        portfolio_price_2 = (portfolio_returns_2 + 1).cumprod()



        return (test_eigen_dataframe, test_trace_dataframe, eigen_test_statistic_1, eigen_test_statistic_2,
                trace_test_statistic_1, trace_test_statistic_2, cointegration_vector_1, cointegration_vector_2,
                portfolio_returns_1, portfolio_price_1, portfolio_returns_2, portfolio_price_2)

    def adf_test_result(self, dataframe, test_statistic):
        """

        """

        font = 'Roboto'

        color = self.black

        if test_statistic < dataframe.iloc[0][1]:

            message = 'The hypothesis is not rejected at 99% confidence level'

        elif test_statistic < dataframe.iloc[1][1]:

            message = 'The hypothesis is not rejected at 95% confidence level'

        elif test_statistic < dataframe.iloc[2][1]:

            message = 'The hypothesis is not rejected at 90% confidence level'

        else:

            message = 'HYPOTHESIS REJECTED'
            color = self.orange
            font = 'Josefin Sans'

        output = [message, color, font]

        return output

    def johansen_test_result(self, dataframe, test_statistic_1, test_statistic_2):
        """

        """

        font = 'Roboto'

        color = self.black

        if (test_statistic_1 > dataframe.iloc[0][1]) and test_statistic_2 > dataframe.iloc[0][2]:

            message = 'The hypothesis is not rejected at 99% confidence level'

        elif test_statistic_1 > dataframe.iloc[1][1] and test_statistic_2 > dataframe.iloc[1][2]:

            message = 'The hypothesis is not rejected at 95% confidence level'

        elif test_statistic_1 > dataframe.iloc[2][1] and test_statistic_2 > dataframe.iloc[2][2]:

            message = 'The hypothesis is not rejected at 90% confidence level'

        else:

            message = 'HYPOTHESIS REJECTED'
            color = self.orange
            font = 'Josefin Sans'

        output = [message, color, font]

        return output

    def asset_prices_plot(self, data, norm_1, norm_2, name_1, name_2):
        """

        """

        asset_prices = go.Figure()

        asset_prices.add_trace(
            go.Scatter(x=data.index, y=norm_1, mode='lines', line=dict(color=self.blue), name=name_1))
        asset_prices.add_trace(
            go.Scatter(x=data.index, y=norm_2, mode='lines', line=dict(color=self.orange), name=name_2))

        asset_prices.update_layout(legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.925), title="Asset prices",
                                   xaxis_title="Date", yaxis_title="Price", font_family='Josefin Sans', font_size=18,
                                   height=500, hovermode='x unified')

        asset_prices.update_xaxes(rangeslider_visible=True)

        return asset_prices

    def portfolio_plot(self, portfolio_price, portfolio_return):
        """

        """
        portfolio = go.Figure()

        portfolio.add_trace(
            go.Scatter(x=portfolio_price.index, y=portfolio_price, mode='lines', line=dict(color=self.blue),
                       name='Portfolio price'))
        portfolio.add_trace(
            go.Scatter(x=portfolio_return.index, y=portfolio_return, mode='lines', line=dict(color=self.orange),
                       name='Portfolio return', visible='legendonly'))

        portfolio.update_layout(legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.77),
                                title="Normalized portfolio", xaxis_title="Date", yaxis_title="Price",
                                font_family='Josefin Sans', font_size=18, height=500,
                                margin=dict(l=30, r=30, t=50, b=30))

        portfolio.update_xaxes(rangeslider_visible=True)

        return portfolio

    def pacf_plot(self, pacf):
        """

        """

        trace = {"name": "PACF", "type": "bar", "marker_color": self.blue, "y": pacf}

        pacf_plot = go.Figure(data=trace)

        pacf_plot.update_layout(title="PACF", xaxis_title="Lag", font_family='Josefin Sans', font_size=14, height=250,
                                margin=dict(l=30, r=30, t=50, b=30))

        return pacf_plot

    def acf_plot(self, acf):
        """

        """

        trace = {"name": "ACF", "type": "bar", "marker_color": self.blue, "y": acf}

        acf_plot = go.Figure(data=trace)

        acf_plot.update_layout(title="ACF", xaxis_title="Lag", font_family='Josefin Sans', font_size=14, height=250,
                               margin=dict(l=30, r=30, t=50, b=30))

        return acf_plot

    def residuals_plot(self, residuals):
        """

        """

        resid_plot = go.Figure()
        resid_plot.add_trace(go.Scatter(x=residuals.index, y=residuals, mode='lines', line=dict(color=self.orange)))
        resid_plot.add_shape(type='line', x0=residuals.index[0], y0=residuals.mean(), x1=residuals.index[-1],
                             y1=residuals.mean(), line=dict(color=self.grey, dash='dash'))

        resid_plot.update_layout(title="Residuals plot", font_family='Josefin Sans', font_size=14, height=250,
                                 margin=dict(l=30, r=30, t=50, b=30))

        return resid_plot

    def qq_plot(self, qq_data, x_data):
        """

        """

        qq_plot = go.Figure()

        qq_plot.add_scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', line=dict(color=self.blue))
        qq_plot.add_scatter(x=x_data, y=qq_data[1][1] + qq_data[1][0] * x_data, mode='lines',
                            line=dict(color=self.grey))

        qq_plot.update_layout(title="Q-Q Plot", font_family='Josefin Sans', font_size=14, height=550,
                              margin=dict(l=30, r=30, t=50, b=30), showlegend=False)
        return qq_plot

    def jh_coint_test_div(self, asset_1, asset_2, coint_test_eigen, coint_test_trace, eigen_test_statistic_1,
                          eigen_test_statistic_2, trace_test_statistic_1, trace_test_statistic_2, ):
        """

        """

        output = html.Div(
            style={'padding-left': 50, 'padding-right': 0, 'padding-top': 20, 'padding-bottom': 50, 'margin-left': 50,
                   'margin-right': 0, 'margin-top': 0, 'margin-bottom': 0, 'backgroundColor': self.white,
                   'horizontal-align': 'center', }, children=[

                html.H3(children='Cointegration tests results:',
                        style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                               'font-weight': '500', 'font-size': 24, 'padding-bottom': '1%', 'padding-top': '1%',
                               'display': 'block'}),

                html.Div(style={'backgroundColor': self.white, 'display': 'inline-block', 'padding-left': 0,
                                'padding-right': 50, 'padding-top': 0, 'padding-bottom': 0, 'margin-left': 10,
                                'margin-right': 50, 'margin-top': 0, 'margin-bottom': 0, 'vertical-align': 'center',
                                'horizontal-align': 'center', 'width': '34%'}, children=[

                    html.H3(children='Eigenvalue test:',
                            style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                   'font-weight': '500', 'font-size': 22, 'padding-bottom': '2%', 'display': 'block'}),

                    html.P(children=
                    self.johansen_test_result(coint_test_eigen, eigen_test_statistic_1, eigen_test_statistic_2)[0],
                        style={'textAlign': 'left', 'color':
                            self.johansen_test_result(coint_test_eigen, eigen_test_statistic_1, eigen_test_statistic_2)[
                                1], 'font-family': self.johansen_test_result(coint_test_eigen, eigen_test_statistic_1,
                                                                             eigen_test_statistic_2)[2],
                               'font-weight': '350', 'font-size': 20, 'padding-bottom': '2%', 'padding-top': 0,
                               'display': 'block'}),

                    dash_table.DataTable(data=coint_test_eigen.to_dict('records'),
                                         columns=[{'id': c, 'name': c} for c in coint_test_eigen.columns],
                                         style_as_list_view=True,

                                         style_cell={'padding': '10px', 'backgroundColor': 'white', 'fontSize': 14,
                                                     'font-family': 'Roboto'},

                                         style_header={'backgroundColor': 'white', 'fontWeight': 'bold', 'fontSize': 14,
                                                       'font-family': 'Josefin Sans'

                                                       }),

                    html.Div(style={'display': 'block'}, children=[

                        html.P(children='Test statistic value for {}: '.format(asset_1),
                               style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                      'font-weight': '300', 'font-size': 18, 'padding-bottom': 0,
                                      'display': 'inline-block'}),

                        html.P(children='⠀{}⠀'.format(round(eigen_test_statistic_1, 5)),
                               style={'textAlign': 'center', 'color': self.blue, 'font-family': 'Josefin Sans',
                                      'font-weight': '400', 'font-size': 18, 'padding-bottom': 0,
                                      'display': 'inline-block'}), html.Div(children=[

                            html.P(children='Test statistic value for {}: '.format(asset_2),
                                   style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                          'font-weight': '300', 'font-size': 18, 'padding-bottom': '5%',
                                          'display': 'inline-block'}),

                            html.P(children='⠀{}⠀'.format(round(eigen_test_statistic_2, 5)),
                                   style={'textAlign': 'center', 'color': self.blue, 'font-family': 'Josefin Sans',
                                          'font-weight': '400', 'font-size': 18, 'padding-bottom': '5%',
                                          'display': 'inline-block'}), ])

                    ]),

                ]),

                html.Div(style={'backgroundColor': self.white, 'display': 'inline-block', 'padding-left': 50,
                                'padding-right': 0, 'padding-top': 0, 'padding-bottom': 0, 'margin-left': 0,
                                'margin-right': 0, 'margin-top': 0, 'margin-bottom': 0, 'vertical-align': 'center',
                                'horizontal-align': 'center', 'width': '34%'}, children=[

                    html.H3(children='Trace test:',
                            style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                   'font-weight': '500', 'font-size': 22, 'padding-bottom': '2%', 'display': 'block'}),

                    html.P(children=
                           self.johansen_test_result(coint_test_trace, trace_test_statistic_1, trace_test_statistic_2)[
                               0], style={'textAlign': 'left', 'color':
                        self.johansen_test_result(coint_test_trace, trace_test_statistic_1, trace_test_statistic_2)[1],
                                          'font-family':
                                              self.johansen_test_result(coint_test_trace, trace_test_statistic_1,
                                                                        trace_test_statistic_2)[2], 'font-size': 20,
                                          'padding-bottom': '2%', 'padding-top': 0, 'display': 'block'}),

                    dash_table.DataTable(data=coint_test_trace.to_dict('records'),
                                         columns=[{'id': c, 'name': c} for c in coint_test_trace.columns],
                                         style_as_list_view=True,
                                         style_cell={'padding': '10px', 'backgroundColor': 'white', 'fontSize': 14,
                                                     'font-family': 'Roboto'},

                                         style_header={'backgroundColor': 'white', 'fontWeight': 'bold', 'fontSize': 14,
                                                       'font-family': 'Josefin Sans'}),

                    html.Div(style={'display': 'block'}, children=[

                        html.P(children='Test statistic value for {}: '.format(asset_1),

                               style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                      'font-weight': '300', 'font-size': 18, 'padding-bottom': 0,
                                      'display': 'inline-block'}),

                        html.P(children='⠀{}⠀'.format(round(trace_test_statistic_1, 5)),

                               style={'textAlign': 'center', 'color': self.blue, 'font-family': 'Josefin Sans',
                                      'font-weight': '400', 'font-size': 18, 'padding-bottom': 0,
                                      'display': 'inline-block'}), html.Div(children=[

                            html.P(children='Test statistic value for {}: '.format(asset_2),
                                   style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                          'font-weight': '300', 'font-size': 18, 'padding-bottom': '5%',
                                          'display': 'inline-block'}),

                            html.P(children='⠀{}⠀'.format(round(trace_test_statistic_2, 5)),
                                   style={'textAlign': 'center', 'color': self.blue, 'font-family': 'Josefin Sans',
                                          'font-weight': '400', 'font-size': 18, 'padding-bottom': '5%',
                                          'display': 'inline-block'}), ])

                    ]),

                ])

            ])

        return output

    def jh_div(self, asset_1, asset_2, coint_vector, portfolio_price, portfolio_return):

        div = html.Div(
            style={'padding-left': 50, 'padding-right': 50, 'padding-top': 20, 'padding-bottom': 50, 'margin-left': 50,
                   'margin-right': 50, 'margin-top': 0, 'margin-bottom': 50, 'backgroundColor': self.white,
                   'horizontal-align': 'center', }, children=[

                html.Div(style={'backgroundColor': self.white, 'padding-bottom': 20, 'textAlign': 'center',
                                'horizontal-align': 'center'}, children=[

                    html.H2(children='S⠀',
                            style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                   'font-weight': '500', 'font-size': 30, 'padding-bottom': 30,
                                   'display': 'inline-block'}),

                    html.H2(children='=',
                            style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                   'font-weight': '300', 'font-size': 30, 'padding-bottom': 30,
                                   'display': 'inline-block'}),

                    html.H2(children='⠀{}⠀'.format(round(coint_vector.iloc[0][0], 4)),
                            style={'textAlign': 'left', 'color': self.orange, 'font-family': 'Josefin Sans',
                                   'font-weight': '400', 'font-size': 30, 'padding-bottom': 30,
                                   'display': 'inline-block'}),

                    html.H2(children='* {} + '.format(asset_1),
                            style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                   'font-weight': '300', 'font-size': 30, 'padding-bottom': 30,
                                   'display': 'inline-block'}),

                    html.H2(children='⠀{}⠀'.format(round(coint_vector.iloc[0][1], 4)),
                            style={'textAlign': 'left', 'color': self.blue, 'font-family': 'Josefin Sans',
                                   'font-weight': '400', 'font-size': 30, 'padding-bottom': 30,
                                   'display': 'inline-block'}),

                    html.H2(children='* {}'.format(asset_2),
                            style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                   'font-weight': '300', 'font-size': 30, 'padding-bottom': 30,
                                   'display': 'inline-block'}),

                ]),

                # Coint tests
                html.Div(children=[

                    dcc.Graph(id='portfolio-plot', figure=self.portfolio_plot(portfolio_price, portfolio_return),
                              style={'width': '98%', 'height': '100%', 'padding-left': 0, 'padding-right': 0,
                                     'padding-top': 0, 'padding-bottom': 50, 'margin-right': 0, 'margin-top': 0,
                                     'margin-bottom': 50, 'vertical-align': 'top', 'horizontal-align': 'center',
                                     'display': 'inline-block', 'font-family': 'Josefin Sans', 'font-weight': '300',
                                     'font-size': 20, 'textAlign': 'center'})

                ]),

            ])

        return div

    def eg_div(self, asset_1, asset_2, coint_test, test_statistic, beta, portfolio_price, portfolio_return, pacf, acf,
               residuals, qq_data, x_data, res_data):

        div = html.Div(

            style={'padding-left': 50, 'padding-right': 0, 'padding-top': 20, 'padding-bottom': 50, 'margin-left': 50,
                   'margin-right': 0, 'margin-top': 0, 'margin-bottom': 0, 'backgroundColor': self.white,
                   'horizontal-align': 'center', }, children=[

                html.Div(style={'backgroundColor': self.white, 'padding-top': 30, 'textAlign': 'left',
                                'horizontal-align': 'center'}, children=[

                    html.H2(children='S⠀',
                            style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                   'font-weight': '500', 'font-size': 30, 'padding-bottom': 30,
                                   'display': 'inline-block'}),

                    html.H2(children='=',
                            style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                   'font-weight': '300', 'font-size': 30, 'padding-bottom': 30,
                                   'display': 'inline-block'}),

                    html.H2(children='⠀{} +'.format(asset_1),
                            style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                   'font-weight': '300', 'font-size': 30, 'padding-bottom': 30,
                                   'display': 'inline-block'}),

                    html.H2(children='⠀{}⠀'.format(round(beta, 4)),
                            style={'textAlign': 'left', 'color': self.blue, 'font-family': 'Josefin Sans',
                                   'font-weight': '400', 'font-size': 30, 'padding-bottom': 30,
                                   'display': 'inline-block'}),

                    html.H2(children='* {}'.format(asset_2),
                            style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                   'font-weight': '300', 'font-size': 30, 'padding-bottom': 30,
                                   'display': 'inline-block'}),

                ]),

                # Coint tests
                html.Div(children=[

                    html.Div(style={'backgroundColor': self.white, 'display': 'inline-block', 'padding-left': 0,
                                    'padding-right': 50, 'padding-top': 0, 'padding-bottom': 0, 'margin-left': 0,
                                    'margin-right': 10, 'margin-top': 0, 'margin-bottom': 0, 'vertical-align': 'center',
                                    'horizontal-align': 'center', 'width': '22%'}, children=[

                        html.H3(children='Cointegration test results:',
                                style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                       'font-weight': '500', 'font-size': 24, 'padding-bottom': '2%',
                                       'display': 'block'}),
                        html.P(children=self.adf_test_result(coint_test, test_statistic)[0],
                               style={'textAlign': 'left', 'color': self.adf_test_result(coint_test, test_statistic)[1],
                                      'font-family': self.adf_test_result(coint_test, test_statistic)[2],
                                      'font-weight': '350', 'font-size': 20, 'padding-bottom': '15%', 'padding-top': 0,
                                      'display': 'block'}),

                        dash_table.DataTable(data=coint_test.to_dict('records'),
                                             columns=[{'id': c, 'name': c} for c in coint_test.columns],
                                             style_as_list_view=True,

                                             style_cell={'padding': '10px', 'backgroundColor': 'white', 'fontSize': 14,
                                                         'font-family': 'Roboto'},

                                             style_header={'backgroundColor': 'white', 'fontWeight': 'bold',
                                                           'fontSize': 14, 'font-family': 'Josefin Sans'

                                                           }, ),

                        html.Div(style={'display': 'block'}, children=[

                            html.P(children='Test statistic value: ',
                                   style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                          'font-weight': '300', 'font-size': 18, 'padding-bottom': '10%',
                                          'display': 'inline-block'}),

                            html.P(children='⠀{}⠀'.format(round(test_statistic, 5)),
                                   style={'textAlign': 'center', 'color': self.blue, 'font-family': 'Josefin Sans',
                                          'font-weight': '400', 'font-size': 18, 'padding-bottom': '10%',
                                          'display': 'inline-block'}),

                        ]),

                    ]),

                    dcc.Graph(id='portfolio-plot', figure=self.portfolio_plot(portfolio_price, portfolio_return),
                              style={'width': '68%', 'height': '100%', 'padding-left': 50, 'padding-right': 0,
                                     'padding-top': 0, 'padding-bottom': 0, 'margin-right': 0, 'margin-top': 0,
                                     'margin-bottom': 0, 'vertical-align': 'top', 'horizontal-align': 'right',
                                     'display': 'inline-block', 'font-family': 'Josefin Sans', 'font-weight': '300',
                                     'font-size': 20, })

                ]),

                # Coint tests
                html.Div(id='resid', children=[

                    html.Div(style={'backgroundColor': self.white, 'display': 'inline-block', 'padding-left': 0,
                                    'padding-right': 0, 'padding-top': 0, 'padding-bottom': 0, 'margin-left': 0,
                                    'margin-right': 0, 'margin-top': 0, 'margin-bottom': 0, 'vertical-align': 'center',
                                    'horizontal-align': 'center', 'width': '100%'}, children=[

                        html.H3(children='Residuals analysis results:',
                                style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                       'font-weight': '500', 'font-size': 22, 'padding-bottom': '2%',
                                       'display': 'block'}), html.Div(
                            style={'backgroundColor': self.white, 'display': 'inline-block', 'padding-left': 0,
                                   'padding-right': 20, 'padding-top': 0, 'padding-bottom': 0, 'margin-left': 0,
                                   'margin-right': 0, 'margin-top': 0, 'margin-bottom': 0, 'vertical-align': 'top',
                                   'horizontal-align': 'center', 'width': '30%'}, children=[

                                dash_table.DataTable(data=res_data.to_dict('records'),
                                                     columns=[{'id': c, 'name': c} for c in res_data.columns],
                                                     style_as_list_view=True,

                                                     style_cell={'padding': '10px', 'backgroundColor': 'white',
                                                                 'fontSize': 14, 'font-family': 'Roboto',
                                                                 'textAlign': 'left'},

                                                     style_header={'padding': '15px',
                                                                   'backgroundColor': self.light_grey, 'fontSize': 18,
                                                                   'font-family': 'Josefin Sans', 'textAlign': 'left'

                                                                   }),

                                dcc.Graph(id='residuals-plot', figure=self.residuals_plot(residuals),
                                          style={'width': '100%', 'height': '100%', 'padding-left': 0,
                                                 'padding-right': 0, 'padding-top': 50, 'padding-bottom': 0,
                                                 'margin-right': 0, 'margin-top': 0, 'margin-bottom': 0,
                                                 'vertical-align': 'top', 'horizontal-align': 'right',
                                                 'display': 'block', 'font-family': 'Josefin Sans',
                                                 'font-weight': '300', 'font-size': 20, }),

                            ]),

                        html.Div(style={'backgroundColor': self.white, 'display': 'inline-block', 'width': '30%',
                                        'vertical-align': 'center', 'horizontal-align': 'center', }, children=[

                            dcc.Graph(id='pacf-plot', figure=self.pacf_plot(pacf),
                                      style={'width': '100%', 'height': '100%', 'padding-left': 0, 'padding-right': 0,
                                             'padding-top': 0, 'padding-bottom': 50, 'margin-right': 0, 'margin-top': 0,
                                             'margin-bottom': 0, 'vertical-align': 'top', 'horizontal-align': 'right',
                                             'display': 'block', 'font-family': 'Josefin Sans', 'font-weight': '300',
                                             'font-size': 20, }),

                            dcc.Graph(id='acf-plot', figure=self.acf_plot(acf),
                                      style={'width': '100%', 'height': '100%', 'padding-left': 0, 'padding-right': 0,
                                             'padding-top': 0, 'padding-bottom': 50, 'margin-right': 0, 'margin-top': 0,
                                             'margin-bottom': 0, 'vertical-align': 'top', 'horizontal-align': 'right',
                                             'display': 'block', 'font-family': 'Josefin Sans', 'font-weight': '300',
                                             'font-size': 20, }), ]),

                        html.Div(style={'backgroundColor': self.white, 'display': 'inline-block', 'padding-left': 0,
                                        'padding-right': 0, 'padding-top': 0, 'padding-bottom': 0, 'margin-left': 0,
                                        'margin-right': 0, 'margin-top': 0, 'margin-bottom': 0,
                                        'vertical-align': 'center', 'horizontal-align': 'center', 'width': '30%'},
                                 children=[

                                     dcc.Graph(id='qq-plot', figure=self.qq_plot(qq_data, x_data),
                                               style={'width': '100%', 'height': '100%', 'padding-left': 10,
                                                      'padding-right': 0, 'padding-top': 0, 'padding-bottom': 50,
                                                      'margin-right': 0, 'margin-top': 0, 'margin-bottom': 0,
                                                      'vertical-align': 'top', 'horizontal-align': 'right',
                                                      'display': 'block', 'font-family': 'Josefin Sans',
                                                      'font-weight': '300', 'font-size': 20, }),

                                 ]), ]), ]), ])

        return div

    def get_cointegration_approach_tearsheet(self):
        """

        """

        data_1 = self.data

        name_1, norm_price_asset_1, adf_asset_1, test_stat_1, name_2, norm_price_asset_2, adf_asset_2, test_stat_2 = self.get_basic_asset_data()

        data_2 = self.data[[name_2, name_1]]

        eg_adf_dataframe_1, eg_adf_test_stat_1, eg_cointegration_vector_1, eg_portfolio_returns_1, \
        eg_portfolio_price_1, eg_residuals_1, eg_residuals_dataframe_1, \
        eg_qq_y_1, eg_x_1, eg_pacf_result_1, eg_acf_result_1 = self.get_engle_granger_data(data_1)

        eg_adf_dataframe_2, eg_adf_test_stat_2, eg_cointegration_vector_2, eg_portfolio_returns_2,\
        eg_portfolio_price_2, eg_residuals_2, eg_residuals_dataframe_2,\
        eg_qq_y_2, eg_x_2, eg_pacf_result_2, eg_acf_result_2 = self.get_engle_granger_data(data_2)

        j_test_eigen_dataframe, j_test_trace_dataframe, j_eigen_test_statistic_1, \
        j_eigen_test_statistic_2, j_trace_test_statistic_1, j_trace_test_statistic_2,\
        j_cointegration_vector_1, j_cointegration_vector_2, j_portfolio_returns_1, \
        j_portfolio_price_1, j_portfolio_returns_2, j_portfolio_price_2 = self.get_johansen_data()

        app = dash.Dash()

        app.layout = html.Div(style={'backgroundColor': self.light_grey, 'padding-bottom': 30}, children=[

            html.Img(src='/assets/ArbitrageLab-logo.png',
                     style={'width': '20%', 'height': '20%', 'padding-top': 50, 'display': 'block',
                            'margin-left': 'auto', 'margin-right': 'auto'}),

            html.H1(children='COINTEGRATION ANALYSIS OF {}/{}'.format(name_1, name_2),
                    style={'textAlign': 'center', 'color': self.black, 'font-family': 'Josefin Sans',
                           'font-weight': '300', 'font-size': 50, 'padding-top': 30, 'padding-left': 50,
                           'margin-top': 30, 'margin-left': 'auto', 'margin-right': 'auto'}),

            # INITIAL DIV
            html.Div(style={'margin-left': '5%', 'margin-right': '5%', 'margin-top': '5%', 'margin-bottom': '5%',
                            'backgroundColor': self.white, 'horizontal-align': 'center'}, children=[

                html.Div(style={'backgroundColor': self.white, 'display': 'inline-block', 'padding-left': 30,
                                'padding-right': 50, 'padding-top': 60, 'padding-bottom': 50, 'margin-left': 50,
                                'margin-right': 50, 'margin-top': 0, 'margin-bottom': 50, 'vertical-align': 'center',
                                'horizontal-align': 'center', 'height': 315, 'width': '20%'}, children=[

                    html.H2(children='The results of the ADF tests:',
                            style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                                   'font-weight': '500', 'font-size': 24, 'padding-bottom': '20%', }),

                    html.P(children='{}'.format(name_1),
                           style={'textAlign': 'left', 'color': self.blue, 'font-family': 'Josefin Sans',
                                  'font-weight': '500', 'font-size': 20

                                  }),

                    html.P(children=self.adf_test_result(adf_asset_1, test_stat_1)[0],
                           style={'textAlign': 'left', 'color': self.adf_test_result(adf_asset_1, test_stat_1)[1],
                                  'font-family': self.adf_test_result(adf_asset_1, test_stat_1)[2],
                                  'font-weight': '300', 'font-size': 18, 'padding-bottom': 10

                                  }),

                    html.P(children='GDXJ',
                           style={'textAlign': 'left', 'color': self.blue, 'font-family': 'Josefin Sans',
                                  'font-weight': '500', 'font-size': 20

                                  }),

                    html.P(children=self.adf_test_result(adf_asset_2, test_stat_2)[0],
                           style={'textAlign': 'left', 'color': self.adf_test_result(adf_asset_2, test_stat_2)[1],
                                  'font-family': self.adf_test_result(adf_asset_2, test_stat_2)[2],
                                  'font-weight': '300', 'font-size': 18

                                  }),

                ]),

                dcc.Graph(id='asset-prices',
                          figure=self.asset_prices_plot(data_1, norm_price_asset_1, norm_price_asset_2, name_1, name_2),
                          style={'width': '60%', 'height': '100%', 'padding-left': 50,  # 'padding-right': 50,
                                 'padding-top': 50, 'padding-bottom': 30,  # 'margin-right': 50,
                                 'margin-top': 0, 'margin-bottom': 0, 'vertical-align': 'top',
                                 'horizontal-align': 'right', 'display': 'inline-block', 'font-family': 'Josefin Sans',
                                 'font-weight': '300', 'font-size': 24,

                                 })

            ]),

            html.Div(style={'margin-left': '5%', 'margin-right': '5%', 'margin-top': '5%', 'margin-bottom': '5%',
                            'padding-left': 0, 'padding-right': 0, 'padding-top': 0, 'padding-bottom': 0,
                            'backgroundColor': self.white, 'horizontal-align': 'center'}, children=[

                html.H2(children='ENGLE-GRANGER APPROACH',
                        style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                               'font-weight': '300', 'font-size': 40, 'padding-left': 50, 'padding-right': 50,
                               'padding-top': 60, 'padding-bottom': 0, 'margin-left': 50, 'margin-right': 50,
                               'margin-top': 0, 'margin-bottom': 0,

                               }),

                html.Div(style={'display': 'block', 'padding-left': 50, 'padding-right': 50, 'padding-top': 60,
                                'padding-bottom': 0, 'margin-left': 50, 'margin-right': 50, 'margin-top': 0,
                                'margin-bottom': 0, 'vertical-align': 'center', 'horizontal-align': 'right'}, children=[

                    html.Button('{}/{}'.format(name_1, name_2), id='button-1', n_clicks=0,
                                style={'BackgroundColor': self.white, 'padding-left': 40, 'padding-right': 40,
                                       'padding-top': 20, 'padding-bottom': 20, 'margin-left': 0, 'margin-right': 25,
                                       'margin-top': 0, 'margin-bottom': 0, 'font-family': 'Josefin Sans',
                                       'font-weight': '300', 'font-size': 20, }),

                    html.Button('{}/{}'.format(name_2, name_1), id='button-2', n_clicks=0,
                                style={'BackgroundColor': self.white, 'padding-left': 40, 'padding-right': 40,
                                       'padding-top': 20, 'padding-bottom': 20, 'margin-left': 25, 'margin-right': 40,
                                       'margin-top': 0, 'margin-bottom': 0, 'font-family': 'Josefin Sans',
                                       'font-weight': '300', 'font-size': 20, }), ]),

                html.Div(id='container-button'),

            ]),

            html.Div(style={'margin-left': '5%', 'margin-right': '5%', 'margin-top': '5%', 'margin-bottom': '5%',
                            'padding-left': 0, 'padding-right': 0, 'padding-top': 0, 'padding-bottom': 0,
                            'backgroundColor': self.white, 'horizontal-align': 'center'}, children=[

                html.H2(children='JOHANSEN APPROACH',
                        style={'textAlign': 'left', 'color': self.black, 'font-family': 'Josefin Sans',
                               'font-weight': '300', 'font-size': 40, 'padding-left': 50, 'padding-right': 50,
                               'padding-top': 60, 'padding-bottom': 0, 'margin-left': 50, 'margin-right': 50,
                               'margin-top': 0, 'margin-bottom': 0,

                               }),

                html.Div(style={'horizontal-align': 'center'}, children=[

                    html.Div(style={'display': 'inline-block', 'horizontal-align': 'center'}, children=[
                        self.jh_coint_test_div(name_1, name_2, j_test_eigen_dataframe, j_test_trace_dataframe,
                                               j_eigen_test_statistic_1, j_eigen_test_statistic_2,
                                               j_trace_test_statistic_1, j_trace_test_statistic_2)]),

                    html.Div(

                        style={'display': 'inline-block', 'padding-left': '5%', 'padding-right': 0, 'padding-top': 0,
                               'padding-bottom': '5%', 'margin-left': 10, 'margin-right': 0, 'margin-top': 0,
                               'margin-bottom': 0, 'vertical-align': 'bottom', 'width': '25%', 'height': '100%',
                               'horizontal-align': 'center'}, children=[

                            html.Div(style={'display': 'block'}, children=[

                                html.Button('Cointegration vector 1', id='button-3', n_clicks=0,
                                            style={'BackgroundColor': self.white, 'padding-left': 40,
                                                   'padding-right': 40, 'padding-top': 20, 'padding-bottom': 20,
                                                   'margin-left': 50, 'margin-right': 50, 'margin-top': 50,
                                                   'margin-bottom': 50, 'font-family': 'Josefin Sans',
                                                   'font-weight': '300', 'font-size': 20, 'vertical-align': 'center',
                                                   'horizontal-align': 'center'}), ]),

                            html.Div(style={'display': 'block', 'padding-bottom': "5%", }, children=[

                                html.Button('Cointegration vector 2', id='button-4', n_clicks=0,
                                            style={'BackgroundColor': self.white, 'padding-left': 40,
                                                   'padding-right': 40, 'padding-top': 20, 'padding-bottom': 20,
                                                   'margin-left': 50, 'margin-right': 50, 'margin-top': 50,
                                                   'margin-bottom': 50, 'font-family': 'Josefin Sans',
                                                   'font-weight': '300', 'font-size': 20, }), ]),

                        ]), ]),

                html.Div(id='johansen_container'),

            ])])

        @app.callback(Output('container-button', 'children'), Input('button-1', 'n_clicks'),
                      Input('button-2', 'n_clicks'))
        def displayClick(btn1, btn2):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'button-1' in changed_id:
                asset_1 = name_1
                asset_2 = name_2
                coint_test = eg_adf_dataframe_1
                test_statistic = eg_adf_test_stat_1
                beta = eg_cointegration_vector_1.loc[0][1]
                portfolio_price = eg_portfolio_price_1
                portfolio_return = eg_portfolio_returns_1
                pacf = eg_pacf_result_1
                acf = eg_acf_result_1
                residuals = eg_residuals_1
                qq_data = eg_qq_y_1
                x_data = eg_x_1
                res_data = eg_residuals_dataframe_1

            elif 'button-2' in changed_id:
                asset_1 = name_2
                asset_2 = name_1
                coint_test = eg_adf_dataframe_2
                test_statistic = eg_adf_test_stat_2
                beta = eg_cointegration_vector_2.loc[0][1]
                portfolio_price = eg_portfolio_price_2
                portfolio_return = eg_portfolio_returns_2
                pacf = eg_pacf_result_2
                acf = eg_acf_result_2
                residuals = eg_residuals_2
                qq_data = eg_qq_y_2
                x_data = eg_x_2
                res_data = eg_residuals_dataframe_2

            else:
                asset_1 = name_1
                asset_2 = name_2
                coint_test = eg_adf_dataframe_1
                test_statistic = eg_adf_test_stat_1
                beta = eg_cointegration_vector_1.loc[0][1]
                portfolio_price = eg_portfolio_price_1
                portfolio_return = eg_portfolio_returns_1
                pacf = eg_pacf_result_1
                acf = eg_acf_result_1
                residuals = eg_residuals_1
                qq_data = eg_qq_y_1
                x_data = eg_x_1
                res_data = eg_residuals_dataframe_1

            return self.eg_div(asset_1, asset_2, coint_test, test_statistic, beta, portfolio_price, portfolio_return,
                               pacf, acf, residuals, qq_data, x_data, res_data)

        @app.callback(Output('johansen_container', 'children'), Input('button-3', 'n_clicks'),
                      Input('button-4', 'n_clicks'))
        def displayClick(btn1, btn2):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'button-3' in changed_id:
                asset_1 = name_1
                asset_2 = name_2
                coint_vector = j_cointegration_vector_1
                portfolio_price = j_portfolio_price_1
                portfolio_return = j_portfolio_returns_1


            elif 'button-4' in changed_id:
                asset_1 = name_1
                asset_2 = name_2
                coint_vector = j_cointegration_vector_2
                portfolio_price = j_portfolio_price_2
                portfolio_return = j_portfolio_returns_2


            else:
                asset_1 = name_1
                asset_2 = name_2
                coint_vector = j_cointegration_vector_1
                portfolio_price = j_portfolio_price_1
                portfolio_return = j_portfolio_returns_1


            return self.jh_div(asset_1, asset_2, coint_vector, portfolio_price, portfolio_return)

        if __name__ == '__main__':
            app.run_server()
