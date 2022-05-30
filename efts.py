# --------------------------------------------------  PACKAGES
from main import app
import dash
from dash import dcc
import dash_bootstrap_components as dbc

from dash import Input, Output, dcc, html

import pandas_ta as ta

import ta

import numpy as np

import warnings

warnings.filterwarnings("ignore")

import yfinance as yf

import requests

from bs4 import BeautifulSoup


import pandas as pd

from plotly.subplots import make_subplots

from statsmodels.tsa.stattools import adfuller

import plotly.express as px

import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error

import math


from itertools import cycle

# -----------------------------------------------------------

etfs = {'First Trust Developed Markets ex-US Small Cap AlphaDEX Fund': '/quote/FDTS?p=FDTS',
 'ProShares Ultra QQQ': '/quote/QLD?p=QLD',
 'Consumer Discretionary Select Sector SPDR\xa0Fund': '/quote/XLY?p=XLY',
 'Fidelity MSCI Consumer Discretionary Index ETF': '/quote/FDIS?p=FDIS',
 'Vanguard Consumer Discretionary Fund': '/quote/VCR?p=VCR',
 'Global X MSCI China Consumer Discretionary ETF': '/quote/CHIQ?p=CHIQ',
 'KraneShares MSCI China Clean Technology ETF': '/quote/KGRN?p=KGRN',
 'VanEck Retail ETF': '/quote/RTH?p=RTH',
 'iShares US Consumer Discretionary ETF': '/quote/IYC?p=IYC',
 'Vanguard U.S. Quality Factor ETF': '/quote/VFQY?p=VFQY',
 'iShares Evolved U.S. Discretionary Spending ETF': '/quote/IEDI?p=IEDI',
 'Invesco Dynamic Semiconductors ETF': '/quote/PSI?p=PSI',
 'Invesco DWA Technology Momentum ETF': '/quote/PTF?p=PTF',
 'Invesco S&P 500 High Beta ETF': '/quote/SPHB?p=SPHB',
 'SPDR\xa0S&P\xa0Semiconductor ETF': '/quote/XSD?p=XSD',
 'iShares U.S. Home Construction ETF': '/quote/ITB?p=ITB',
 'SPDR S&P Kensho Smart Mobility ETF': '/quote/HAIL?p=HAIL',
 'iShares Semiconductor ETF': '/quote/SOXX?p=SOXX',
 'Barclays ETN+ Shiller Capet ETN': '/quote/CAPD?p=CAPD',
 'iShares MSCI Kokusai ETF': '/quote/TOK?p=TOK',
 'Direxion Russell 1000 Growth Over Value ETF': '/quote/RWGV?p=RWGV',
 'First Trust NASDAQ-100-Technology Sector Index Fund': '/quote/QTEC?p=QTEC',
 'SPDR\xa0S&P\xa0Transportation ETF': '/quote/XTN?p=XTN',
 'SPDR\xa0S&P\xa0Metals and Mining ETF': '/quote/XME?p=XME',
 'VanEck Semiconductor ETF': '/quote/SMH?p=SMH'}

models = {'Linear Regression': LinearRegression(),
           'Decision Tree': DecisionTreeRegressor(random_state=93, max_depth=6),
           'Suport Vector Machine': SVR(kernel='linear', tol=1e-5),
           'Multi-Layer Perceptron': MLPRegressor(random_state=93)}

# -----------------------------------------------------------DROPDOWNS
efts_drop = dcc.Dropdown(id='dropdown_efts',
                         options=[{'label': x, 'value': x} for x in etfs.keys()],
                         value='ProShares Ultra QQQ',
                         style={'color': '#000000'})
#
# exchange_rate_drop = dcc.Dropdown(id='dropdown_exchange_rate',
#                                   # options=[{'label': x, 'value': x} for x in crypto_dict.keys()],
#                                   placeholder="Select the currency",
#                                   value='EUR USD',
#                                   style={'color': '#000000'})

# --------------------------------------------------TOOLTIP-------------

def create_tooltip(id, tooltip_text):
    return html.Div(
        children=[
            html.Span(
                "?",
                id=id,
                style={"textAlign": "center", "color": "white",},
                className="dot",
            ),
            dbc.Tooltip(tooltip_text, target=id, autohide=False),
        ],
        style={"display": "inline"},
    )

# ------------------------------------------------

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# --------------------------------------------------

efts_main =dbc.Container([
    html.Br(),
    html.P('Here are the real time values for the price, change, percentage change, volume and net assets:'),
    html.Br(),

html.Div([
        dbc.Col([
            html.Div([
                    html.H4('Price', style={'font-weight': 'normal'}),
                    html.H5(id='price_efts',className='number-color')
                ], className='cool_box', style={'width': '33.33%'}),
            html.Div([
                    html.H4('Change', style={'font-weight': 'normal'}),
                    html.H5(id='change_efts',className='number-color')
                ], className='cool_box', style = {'width': '33.33%'}),
             html.Div([
                html.H4('%Change', style={'font-weight': 'normal'}),
                html.H5(id='perc_change_efts',className='number-color')
            ], className='cool_box',style = {'width': '33.33%'}),
	], className='center',style = {'display': 'flex'})]),
        html.Br(),
    html.Div([
        dbc.Col([
            html.Div([
                html.H4('Volume', style={'font-weight': 'normal'}),
                html.H5(id='volume_efts', className='number-color')
            ], className='cool_box', style={'width': '50%'}),
            html.Div([
                html.H4('Net Assets', style={'font-weight': 'normal'}),
                html.H5(id='net_assets_efts', className='number-color')
            ], className='cool_box', style={'width': '50%'}),
        ], className='center', style={'display': 'flex'})
    ])])

efts_expl = dbc.Container([
    html.Br(),
    dbc.Row([
        html.Br(),

html.Div([
    html.Div([
            html.Img(src='/assets/calander.png',height='25px'),
        ],style={'width':'6%'}),
        html.Div([
            html.P('Choose a date range:',
                        className='text-left mb-4'),
                ],style={'width':'90%'})
], id='3th row', style={'display': 'flex','width':'50%'}),


        dbc.RadioItems(id='time_picker_efts',
                       options=[{"label": "5d", "value": "5d"},
                                {"label": "1mo", "value": "1mo"},
                                {"label": "3mo", "value": "3mo"},
                                {"label": "6mo", "value": "6mo"},
                                {"label": "YTD", "value": "YTD"},
                                {"label": "1y", "value": "1y"},
                                {"label": "5y", "value": "5y"},
                                {"label": "max", "value": "max"}],
                                value='6mo',
                                input_checked_style={"backgroundColor": "orange"},
                                inline=True,
                                labelClassName='mr-3'),
        html.Br(),
        dcc.Store(id='d_efts'),
        html.Br(),
        dbc.Row([
            dbc.Row([
                dbc.Col([
                    html.H4('Stationarity', style={'text-align': 'right'})
                ], width=7),
                dbc.Col([
                    create_tooltip(id="tooltip",
                                   tooltip_text="Stochastic process where the mean, variance and autocorrelation structure doesn’t change over time.")
                ],width=5),
            ]),
            dcc.Graph(id='stationarity_efts', figure={}),
        ]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),

        dbc.Row([
            dbc.Col([
                html.H4('Augmented Dickey-Fuller (ADF) Test', style={'text-align': 'right'})
            ], width=8),
            dbc.Col([
                create_tooltip(id="tooltip2",
                               tooltip_text="Statistical test used to test whether a given Time series is stationary or not. It is one of the most commonly used statistical test when it comes to analyzing the stationary of a series")
            ], width=3),
        ]),
         html.Br(),
        html.Br(),
        html.Br(),


  html.Div([
        dbc.Col([
            html.Div([
                    html.H4('P-Value', style={'font-weight': 'normal'}),
                    html.H5(id='p_value_efts',className='number-color')
                ], className='cool_box', style={'width': '25%'}),
            html.Div([
                    html.H4('Critical value 1%:', style={'font-weight': 'normal'}),
                    html.H5(id='critical1_efts',className='number-color')
                ], className='cool_box', style = {'width': '25%'}),
             html.Div([
                html.H4('Critical value 5%:', style={'font-weight': 'normal'}),
                html.H5(id='critical5_efts',className='number-color')
            ], className='cool_box',style = {'width': '25%'}),
	        html.Div([
                html.H4('Critical value 10%:', style={'font-weight': 'normal'}),
                html.H5(id='critical10_efts',className='number-color')
            ], className='cool_box',style = {'width': '25%'}),
	], className='center',id = '3th row',style = {'display': 'flex'}),
        html.Br(),

      dbc.Row([
          dbc.Col([
              html.H5(id='comment_efts')
          ], className='center'),
      ])
  ]),

        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Row([
                dbc.Col([
                    html.H4('Tendency with Moving Average', style={'text-align': 'right'})
                ], width=8),
                dbc.Col([
                    create_tooltip(id="tooltip3",
                                   tooltip_text="Used to determine the trend direction of securities when looking at the close price.")
                ], width=3),
            ]),
            dcc.Graph(id='candle_efts', figure={}),
        ]),
    ])
])
efts_tech = dbc.Container([
    html.Br(),
    dcc.Store(id='d2_efts'),
    dcc.Store(id='df_indicators_efts'),
    dbc.Row([
        html.Br(),


html.Div([
    html.Div([
            html.Img(src='/assets/calander.png',height='25px'),
        ],style={'width':'6%'}),
        html.Div([
            html.P('Choose a date range:',
                        className='text-left mb-4'),
                ],style={'width':'90%'})
], id='3th row', style={'display': 'flex','width':'50%'}),


        dbc.RadioItems(id='time_picker2_efts',
                       options=[{"label": "3mo", "value": "3mo"},
                                {"label": "6mo", "value": "6mo"},
                                {"label": "YTD", "value": "YTD"},
                                {"label": "1y", "value": "1y"},
                                {"label": "5y", "value": "5y"},
                                {"label": "max", "value": "max"}],
                       value='6mo',
                       input_checked_style={"backgroundColor": "orange"},
                       inline=True,
                       labelClassName='mr-3'),
        html.Br(),
        html.Br()
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([html.P('Choose Volatility Indicators:')], width=8),
                dbc.Col([create_tooltip(id="tooltip3",
                                   tooltip_text="Technical analysis tools that look at changes in market prices over a specified period of time.")
                         ], width=4)]),
            dcc.Dropdown(id='dropdown_volatility_efts',
                         options=[{"label": "Ulcer Index", "value": "Ulcer Index"},
                                  {"label": "ATR", "value": "ATR"},
                                  {"label": "Bollinger Bands", "value": "Bollinger Bands"},
                                  {"label": "Donchian Channel", "value": "Donchian Channel"},
                                  {"label": "Keltner Channel", "value": "Keltner Channel"}],
                         value=["Bollinger Bands"],
                         multi=True,
                         style={'color': '#000000', 'background-color': '#2B303A'}),
        ], width=4),
        dbc.Col([
            dbc.Row([
                dbc.Col([html.P('Choose Momentum Indicators:')], width=8),
                dbc.Col([create_tooltip(id="tooltip4",
                                        tooltip_text="Show the movement of price over time and how strong those movements are/will be, regardless of the direction the price moves, up, or down.")
                         ], width=4)]),
            dcc.Dropdown(id='dropdown_momentum_efts',
                         options=[{"label": "RSI", "value": "RSI"},
                                  {"label": "Stochastic Oscillator", "value": "Stochastic Oscillator"},
                                  {"label": "Stochastic RSI", "value": "Stochastic RSI"},
                                  {"label": "Awesome Oscillator", "value": "Awesome Oscillator"},
                                  {"label": "ROC", "value": "ROC"}],
                         value=["RSI"],
                         multi=True,
                         style={'color': '#000000', 'background-color': '#2B303A'}),
        ], width=4),
        dbc.Col([
            dbc.Row([
                dbc.Col([html.P('Choose Trend Indicators:')], width=7),
                dbc.Col([create_tooltip(id="tooltip5",
                                   tooltip_text="Measure the direction of the trend by differentiating ups/downs trending markets from ranging markets.")
                         ], width=5)
            ]),
            dcc.Dropdown(id='dropdown_trend_efts',
                         options=[{"label": "SMA", "value": "SMA"},
                                  {"label": "TRIX", "value": "TRIX"},
                                  {"label": "ADX", "value": "ADX"},
                                  {"label": "Ichimoku", "value": "Ichimoku"},
                                  {"label": "MACD", "value": "MACD"}],
                         value=["MACD"],
                         multi=True,
                         style={'color': '#000000', 'background-color': '#2B303A'}),
        ], width=4),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Br(),
                dbc.Col([html.P('Choose Volume Indicators:')], width=7),
                dbc.Col([create_tooltip(id="tooltip6",
                                   tooltip_text="Demonstrates changing of trading volume over time. Displays how strong the current trend is.")
                         ], width=5)
            ]),
            dcc.Dropdown(id='dropdown_volume_efts',
                         options=[{"label": "VWAP", "value": "VWAP"},
                                  {"label": "ADI", "value": "ADI"},
                                  {"label": "OBV", "value": "OBV"},
                                  {"label": "CMF", "value": "CMF"},
                                  {"label": "FI", "value": "FI"}],
                         value=["VWAP"],
                         multi=True,
                         style={'color': '#000000', 'background-color': '#2B303A'}),
        ], width=4)
    ]),
    dcc.Graph(id='graph_volatility_efts', figure={}),
    dcc.Graph(id='graph_momentum_efts', figure={}),
    dcc.Graph(id='graph_trend_efts', figure={}),
    dcc.Graph(id='graph_volume_efts', figure={})
])

#----------------------------DROPDOWN  MODELS-----------------

models_drop=dcc.Dropdown(id='dropdown_models_efts',
                 options=[{'label': x, 'value': x} for x in models.keys()],
                 value="Linear Regression",
                 style={'color': '#000000'})

#----------------------------BUTTONS MODELS-------------------------

button_models=dbc.RadioItems(id='hour_day_efts',
                       options=[{"label": "Per hour", "value": "Per hour"},
                                {"label": "Per day", "value": "Per day"}],
                       value='Per day',
                       input_checked_style={"backgroundColor": "orange"},
                       inline=True,
                       labelClassName='mr-3')


button_price_models=dbc.RadioItems(id='price_efts',
                       options=[{"label": "Open", "value": "Open"},
                                {"label": "Low", "value": "Low"},
                                {"label": "High", "value": "High"},
                                {"label": "Close", "value": "Close"}],
                       value='Close',
                       input_checked_style={"backgroundColor": "orange"},
                       inline=True,
                       labelClassName='mr-3')


#----------------------------------MODELS LAYOUT-----------------------------------------

efts_model = dbc.Container([
    html.Br(),

html.Div([
        dbc.Col([
            html.Div([
                    html.P('Choose type of prediction:', style={'font-weight': 'normal'}),
                    button_models
                ], style={'width': '33.33%'}),
            html.Div([
                    html.P('Choose a price:', style={'font-weight': 'normal'}),
                     button_price_models
                ], style = {'width': '33.33%'}),
             html.Div([
                html.P('Choose a model:', style={'font-weight': 'normal'}),
                models_drop
            ],style = {'width': '33.33%'}),
            dcc.Store(id='df_models_efts'),
	],style = {'display': 'flex'})]),






html.Br(),
html.Br(),
html.Br(),
html.Div([
        dbc.Col([
            html.Div([
                    html.H4('Scaled RMSE', style={'font-weight': 'normal'}),
                    html.H5(id='scal_rmse_efts',className='number-color')
                ], className='cool_box', style={'width': '33.33%'}),
            html.Div([
                    html.H4('Original RMSE', style={'font-weight': 'normal'}),
                    html.H5(id='orig_rmse_efts',className='number-color')
                ], className='cool_box', style = {'width': '33.33%'}),
             html.Div([
                html.H4(id='pred_string_efts', style={'font-weight': 'normal'}),
                html.H5(id='today_pred_efts',className='number-color')
            ], className='cool_box',style = {'width': '33.33%'}),
	], className='center',style = {'display': 'flex'})]),
html.Br(),
html.Br(),
html.Br(),
    dcc.Graph(id='model_graph_efts', figure={})

])

# -------------------------------------------------- CREATING DIFFERENT SEPARATORS
efts_tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Main", tab_id="main_efts_sep", active_label_style={"color": "orange"}),
                dbc.Tab(label="Exploratory Analysis", tab_id="expo_efts_sep", active_label_style={"color": "orange"}),
                dbc.Tab(label="Technical Analysis", tab_id="tech_efts_sep", active_label_style={"color": "orange"}),
                dbc.Tab(label="Models", tab_id="models_efts_sep", active_label_style={"color": "orange"}),

                dbc.Row(dbc.Col(width=12)),

            ],
            id="efts_seps",
            active_tab="main_efts_sep",
        ),
    ], className="mt-3"
)
# -------------------------------------------------- APP LAYOUT

efts_layout = dbc.Container([
html.Div([
    html.Div([
        html.Img(src='/assets/efts_header.png', height='90px'),
    ], style={'width': '18%'}),
    html.Div([
        html.H1('ETFS',
                className='text-left mb-4'),
    ], style={'width': '60%'})
], id='3th row', style={'display': 'flex', 'width': '50%'}),

html.Hr(),

html.Br(),
html.Br(),
html.Div([
        html.Div([
            html.Div([
                html.Img(src='/assets/lupa.png', height='22px'),
            ], style={'width': '8%'}),
            html.Div([
                efts_drop], style={'width': '90%'})
        ], id='3th row', style={'display': 'flex', 'width': '62%'}),

        html.Div([
            html.Div([
                html.H6('An exchange-traded fund (ETF) is a type of pooled investment security that operates much like a mutual fund. An ETF can be structured to track anything from the price of an individual commodity to a large and diverse collection of securities.')
        ], style={'width': '90%'})
    ], id='3th row', style={'display': 'flex', 'width': '90%'})], style={'display': 'flex'}),

html.Br(),
html.Br(),
html.Br(),
    dbc.Row(
        dbc.Col(
            efts_tabs, width=12), className="fr-1"),
    html.Div(id='efts_data', children=[])

])
# -------------------------------------------CALLBACKS
@app.callback(
    Output('price_efts', 'children'),
    Output('change_efts', 'children'),
    Output('perc_change_efts', 'children'),
    Output('volume_efts', 'children'),
    Output('net_assets_efts', 'children'),
    Input('dropdown_efts', 'value'))

def choose_directory_etf(asset):
    site = etfs[asset]
    url = f'https://finance.yahoo.com{site}'
    page = requests.get(url)
    html = BeautifulSoup(page.text, 'lxml')
    price = html.find('fin-streamer',
                  class_ ='Fw(b) Fz(36px) Mb(-4px) D(ib)').text
    change = html.find_all('fin-streamer',
                  {'class':'Fw(500) Pstart(8px) Fz(24px)'})[0].text
    perc_change = html.find_all('fin-streamer',
                  {'class':'Fw(500) Pstart(8px) Fz(24px)'})[1].text
    Volume = html.find_all('td',
                {'class':'Ta(end) Fw(600) Lh(14px)'})[6].text
    Net_assets = html.find_all('td',
                {'class':'Ta(end) Fw(600) Lh(14px)'})[8].text

    return price, change, perc_change, Volume, Net_assets

# -------------------------------------------------------------------

efts_sigla = {'First Trust Developed Markets ex-US Small Cap AlphaDEX Fund': 'FDTS',
 'ProShares Ultra QQQ': 'QLD',
 'Consumer Discretionary Select Sector SPDR\xa0Fund': 'XLY',
 'Fidelity MSCI Consumer Discretionary Index ETF': 'FDIS',
 'Vanguard Consumer Discretionary Fund': 'VCR',
 'Global X MSCI China Consumer Discretionary ETF': 'CHIQ',
 'KraneShares MSCI China Clean Technology ETF': 'KGRN',
 'VanEck Retail ETF': 'RTH',
 'iShares US Consumer Discretionary ETF': 'IYC',
 'Vanguard U.S. Quality Factor ETF': 'VFQY',
 'iShares Evolved U.S. Discretionary Spending ETF': 'IEDI',
 'Invesco Dynamic Semiconductors ETF': 'PSI',
 'Invesco DWA Technology Momentum ETF': 'PTF',
 'Invesco S&P 500 High Beta ETF': 'SPHB',
 'SPDR\xa0S&P\xa0Semiconductor ETF': 'XSD',
 'iShares U.S. Home Construction ETF': 'ITB',
 'SPDR S&P Kensho Smart Mobility ETF': 'HAIL',
 'iShares Semiconductor ETF': 'SOXX',
 'Barclays ETN+ Shiller Capet ETN': 'CAPD',
 'iShares MSCI Kokusai ETF': 'TOK',
 'Direxion Russell 1000 Growth Over Value ETF': 'RWGV',
 'First Trust NASDAQ-100-Technology Sector Index Fund': 'QTEC',
 'SPDR\xa0S&P\xa0Transportation ETF': 'XTN',
 'SPDR\xa0S&P\xa0Metals and Mining ETF': 'XME',
 'VanEck Semiconductor ETF': 'SMH'}

# -----------------------------------------------------------------

# -------------------------------------------------------------------

@app.callback(
    Output('d_efts', 'dataframe'),
    Input('dropdown_efts', 'value'),
    Input('time_picker_efts', 'value'))

# --------------------------------------------------

def dataframes_assets1(asset, time):
    asset = efts_sigla[asset]
    msft = yf.Ticker(asset)
    df = msft.history(period=time)

    def log_return(series, periods=1):
        return np.log(series).diff(periods=periods)

    df['MA_%Change'] = ta.volatility.bollinger_mavg(log_return(df.Close)[1:])
    # Bollinger Channel Middle Band
    df["bb_middle_band"] = ta.volatility.bollinger_mavg(
        df["Close"])

    return df.reset_index().to_dict('records')
# --------------------------------------------------

@app.callback(
    Output('stationarity_efts', 'figure'),
    Output('candle_efts', 'figure'),
    Input('d_efts', 'dataframe'),)

# --------------------------------------------------

def update_plots(data):
    fig1 = px.line(data, x='Date', y='Close')
    fig1.update_traces(line=dict(color="orange", width=3))
    fig1.update_xaxes(showline=True, linewidth=2, linecolor='#ffffff', color='#ffffff')
    fig1.update_yaxes(showline=True, linewidth=2, linecolor='#ffffff', mirror=True, color='#ffffff')
    fig1.update_xaxes(rangeslider_visible=True)
    fig1.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')


    data = pd.DataFrame(data)
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_width=[0.2, 0.7])

    # Plot OHLC on 1st row
    fig2.add_trace(go.Candlestick(x=data["Date"], open=data["Open"], high=data["High"],
                                 low=data["Low"], close=data["Close"], name="Candlestick"),
                  row=1, col=1
                  )

    # MA
    fig2.add_trace(go.Scatter(x=data["Date"], y=data["bb_middle_band"], name='MA', line=dict(color='orange', width=3.3)),row=1, col=1)

    # Bar trace for volumes on 2nd row without legend
    colors = np.where(data['MA_%Change'] < 0,  'red', 'greenyellow')
    fig2.add_trace(go.Bar(x=data['Date'], y=data['MA_%Change'], name='MA histogram', marker_color=colors), row=2, col=1)

    fig2.update_layout(yaxis_title="Price")
    fig2.update_xaxes(showline=True, linewidth=2, linecolor='#ffffff', color='#ffffff')
    fig2.update_yaxes(showline=True, linewidth=2, linecolor='#ffffff', mirror=True, color='#ffffff')
    fig2.update_layout(legend=dict(font=dict(color="#ffffff")))

    fig2.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

    return fig1, fig2

# --------------------------------------------------
@app.callback(
    Output('p_value_efts', 'children'),
    Output('critical1_efts', 'children'),
    Output('critical5_efts', 'children'),
    Output('critical10_efts', 'children'),
    Input('d_efts', 'dataframe'))

# --------------------------------------------------

def stationarity(data):
    data = pd.DataFrame(data)
    ts = data['Close']
    adf = adfuller(ts, autolag='AIC')
    padf = [(str(round(adf[1],6)))]
    for perc, value in adf[4].items():
        (padf.append(str(round(value,4))))

    p_value, critical1, critical5, critical10 = padf

    return p_value, critical1, critical5, critical10

# --------------------------------------------------

@app.callback(
    Output('d2_efts', 'dataframe'),
    Input('dropdown_efts', 'value'),
    Input('time_picker2_efts', 'value'))

# --------------------------------------------------

def dataframes_assets2(asset, time2):
    asset = efts_sigla[asset]
    msft = yf.Ticker(asset)
    df = msft.history(period=time2)

    def log_return(series, periods=1):
        return np.log(series).diff(periods=periods)

    df['MA_%Change'] = ta.volatility.bollinger_mavg(log_return(df.Close)[1:])
    # Bollinger Channel Middle Band
    df["bb_middle_band"] = ta.volatility.bollinger_mavg(
        df["Close"])

    return df.reset_index().to_dict('records')

# --------------------------------------------------

@app.callback(
    Output('df_indicators_efts', 'dataframe'),
    Input('d2_efts', 'dataframe'))

# -----------------------------------------------------

def indicators_mt_c(data2):
    data = pd.DataFrame(data2)
    #############################MOMENTUM

    # 1) AWESOME OSCILLATOR
    data['Awsome_Oscillator'] = ta.momentum.awesome_oscillator(data["High"], data["Low"])

    # 2)RSI
    data["rsi"] = ta.momentum.rsi(data["Close"], fillna=False)

    # 3) Stocchastic RSI
    data['stochastic_rsi'] = ta.momentum.stochrsi(data['Close'])
    data['stochastic_%d'] = ta.momentum.stochrsi_d(data['Close'])
    data['stochastic-%k'] = ta.momentum.stochrsi_k(data['Close'])

    # 4) Stochastic Oscillator
    data['stoch_oscillator'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
    data['stoch_signal'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])

    # ROC (Rate of Change)
    data['roc'] = ta.momentum.roc(data['Close'])

    #######################VOLATILITY

    # 1) Bollinger bands
    data["bb_high_band"] = ta.volatility.bollinger_hband(data["Close"])  # a ma ja esta criada para este indicador
    data["bb_low_band"] = ta.volatility.bollinger_lband(data["Close"])

    # 2) Average True Range (ATR)
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])

    # 3) Donchian Channel
    data['donchian_high'] = ta.volatility.donchian_channel_hband(data['High'], data['Low'], data['Close'])
    data['donchian_middle'] = ta.volatility.donchian_channel_mband(data['High'], data['Low'], data['Close'])
    data['donchian_low'] = ta.volatility.donchian_channel_lband(data['High'], data['Low'], data['Close'])

    # 4) Ulcer Index
    data['Ulcer Index'] = ta.volatility.ulcer_index(data['Close'])

    # 5) Keltner Channels
    data['keltner_high'] = ta.volatility.keltner_channel_hband(data['High'], data['Low'], data['Close'])
    data['keltner_low'] = ta.volatility.keltner_channel_lband(data['High'], data['Low'], data['Close'])
    data['keltner_middle'] = ta.volatility.keltner_channel_mband(data['High'], data['Low'], data['Close'])

    #####################TREND

    # 1) Moving Average Convergence Divergence (MACD)
    data['macd_line'] = ta.trend.macd(data['Close'])
    data['macd_hist'] = ta.trend.macd_diff(data['Close'])
    data['macd_signal'] = ta.trend.macd_signal(data['Close'])

    # 2) Average Directional Movement Index (ADX)
    data['adx'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
    data['adx_neg'] = ta.trend.adx_neg(data['High'], data['Low'], data['Close'])
    data['adx_pos'] = ta.trend.adx_pos(data['High'], data['Low'], data['Close'])

    # 3) Ichimoku
    data['ichimoku_a'] = ta.trend.ichimoku_a(data['High'], data['Low'])
    data['ichimoku_b'] = ta.trend.ichimoku_b(data['High'], data['Low'])
    data['ichimoku_base'] = ta.trend.ichimoku_base_line(data['High'], data['Low'])
    data['ichimoku_conv'] = ta.trend.ichimoku_conversion_line(data['High'], data['Low'])

    # 4) SMA - Simple Moving Average
    data['sma'] = ta.trend.sma_indicator(data['Close'])

    # 5) Trix
    data['trix'] = ta.trend.trix(data['Close'])

    #####################VOLUME

    # 1) Accumulation/Distribution Index (ADI)
    data['adi'] = ta.volume.acc_dist_index(data['High'], data['Low'], data['Close'], data['Volume'])

    # 2) On-Balance-Volume (OBV)
    data["obv"] = ta.volume.on_balance_volume(data["Close"], data['Volume'])

    # 3) Chaikin Money Flow (CMF)
    data['cmf'] = ta.volume.chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'])

    # 4) Volume Weighted Average Price (VWAP)
    data['vwap'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume'])

    # 5) Force Index (FI)
    data['fi'] = ta.volume.force_index(data['Close'], data['Volume'])

    return data.to_dict('records')

# -----------------------------------------------------

@app.callback(
    Output('graph_volatility_efts', 'figure'),
    Input('dropdown_volatility_efts', 'value'),
    Input('df_indicators_efts', 'dataframe'))

# -----------------------------------------------------

def volatility(lista, data3):
    data3=pd.DataFrame(data3)

    n = 0
    height = 0
    fig = make_subplots(rows=len(lista), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=lista)
    for i in lista:

        n += 1
        height += 500

        # um indicador
        if i in ['Ulcer Index'
                 ]:
            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3[i], mode="lines", name=i,
                line=dict(color='orange', width=2), showlegend=True
            ), row=n, col=1)

        elif i in ['ATR'
                   ]:
            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3[i], mode="lines", name=i,
                line=dict(color='aqua', width=2), showlegend=True
            ), row=n, col=1)

        # BB
        elif i in ['Bollinger Bands']:
            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3['Close'], mode="markers+lines", name='Close',
                line=dict(color='yellow', width=2)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3['bb_high_band'], mode="lines", name='bb_high_band',
                line=dict(color='salmon', width=1.5)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3['bb_low_band'], mode="lines", name='bb_low_band',
                line=dict(color='salmon', width=1.5)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3['bb_middle_band'], mode="lines", name='bb_middle_band',
                line=dict(color='salmon', width=1.5, dash='dot')
            ), row=n, col=1)




        ## Donchain Channel
        elif i in ['Donchian Channel']:
            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3['Close'], mode="markers+lines", name="Close",
                line=dict(color='yellow', width=2)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3['donchian_high'], mode="lines", name='donchian_high',
                line=dict(color='hotpink', width=1.5)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3['donchian_middle'], mode="lines", name='donchian_middle',
                line=dict(color='hotpink', width=1.5, dash='dot')
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3['donchian_low'], mode="lines", name='donchian_low',
                line=dict(color='hotpink', width=1.5)
            ), row=n, col=1)


        # 'Keltner Channel'
        elif i in ['Keltner Channel']:
            fig.append_trace(go.Scatter(
                    x=data3["Date"], y=data3['Close'], mode="markers+lines", name="Close",
                    line=dict(color='yellow', width=2)
                ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3['keltner_high'], mode="lines", name='keltner_high',
                line=dict(color='firebrick', width=1.5)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3['keltner_low'], mode="lines", name='keltner_low',
                line=dict(color='firebrick', width=1.5)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data3["Date"], y=data3['keltner_middle'], mode="lines", name='keltner_middle',
                line=dict(color='firebrick', width=1.5, dash='dot')
            ), row=n, col=1)

        else:
            continue

    fig.update_layout(height=height, width=1100,
                      title_text="Trading Strategy - Volatility Indicators", xaxis_rangeslider_visible=False)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='#ffffff', color='#ffffff')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#ffffff', mirror=True, color='#ffffff')
    fig.update_annotations(font=dict(color="#ffffff"))

    fig.update_layout(legend=dict(font=dict(color="#ffffff")), title_font=dict(color='#ffffff'), plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    fig.update_xaxes(rangeslider_visible=False)
    return fig

# -----------------------------------------------------

@app.callback(
    Output('graph_momentum_efts', 'figure'),
    Input('dropdown_momentum_efts', 'value'),
    Input('df_indicators_efts', 'dataframe'))

# -----------------------------------------------------

def momentum(lista, data):
    data = pd.DataFrame(data)

    n = 0
    height = 0
    fig = make_subplots(rows=len(lista), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=lista)
    for i in lista:
        n += 1
        height += 500
        # 1 com candle

        # um indicador
        if i in ['RSI',
                 ]:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['rsi'], mode="markers+lines", name='RSI',
                line=dict(color='lightcyan', width=2), showlegend=True
            ), row=n, col=1)

        elif i in ['ROC'
                   ]:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['roc'], mode="markers+lines", name='ROC',
                line=dict(color='mediumturquoise', width=2), showlegend=True
            ), row=n, col=1)


        #
        elif i in ['Awesome Oscillator']:
            colors = np.where(data['Awsome_Oscillator'] < 0, 'red', 'lime')
            fig.append_trace(go.Bar(
                x=data["Date"], y=data['Awsome_Oscillator'],
                name=i, yaxis="y2",
                marker_color=colors, showlegend=True
            ), row=n, col=1)

            # Stochastic RSI

        elif i in ['Stochastic RSI']:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['stochastic_rsi'], line=dict(color='dodgerblue', width=2),
                name='stochastic_rsi', yaxis="y2",
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['stochastic_%d'],
                name='stochastic_%d', yaxis="y2", line=dict(color='yellow', width=2),
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['stochastic-%k'], line=dict(color='tomato', width=2),
                name='stochastic-%k', yaxis="y2",
            ), row=n, col=1)



        ## Stochastic Oscillator
        elif i in ['Stochastic Oscillator']:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['stoch_oscillator'], mode="lines", name='stoch_oscillator',
                line=dict(color='fuchsia', width=2)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['stoch_signal'], mode="lines", name='stoch_signal',
                line=dict(color='wheat', width=2)
            ), row=n, col=1)


        else:
            continue

    fig.update_layout(height=height, width=1100,
                      title_text="Trading Strategy - Momentum Indicators", xaxis_rangeslider_visible=False
                      )
    fig.update_xaxes(rangeslider_visible=False)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='#ffffff', color='#ffffff')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#ffffff', mirror=True, color='#ffffff')
    fig.update_annotations(font=dict(color="#ffffff"))

    fig.update_layout(legend=dict(font=dict(color="#ffffff")), title_font=dict(color='#ffffff'), plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

    return fig

# -----------------------------------------------------

@app.callback(
    Output('graph_trend_efts', 'figure'),
    Input('dropdown_trend_efts', 'value'),
    Input('df_indicators_efts', 'dataframe'))

# -----------------------------------------------------

def trend(lista, data):
    data = pd.DataFrame(data)

    n = 0
    height = 0
    fig = make_subplots(rows=len(lista), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=lista)
    for i in lista:

        n += 1
        height += 500
        # 1 com candle
        if i in ['SMA']:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['Close'], mode="markers+lines", name='Close',
                line=dict(color='yellow', width=2)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['sma'], mode="lines", name=i,
                line=dict(color='tomato', width=1.5)
            ), row=n, col=1)


        # um indicador
        elif i in ['TRIX'
                   ]:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['trix'], mode="lines", name='TRIX',
                line=dict(color='palegreen', width=2), showlegend=True
            ), row=n, col=1)




        ## ADX
        elif i in ['ADX']:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['adx'], mode="lines", name='adx',
                line=dict(color='cornflowerblue', width=2, dash='dot')
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['adx_neg'], mode="lines", name='adx_neg',
                line=dict(color='palegoldenrod', width=2)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['adx_pos'], mode="lines", name='adx_pos', line=dict(color='lawngreen', width=2)
            ), row=n, col=1)


        ##  'Ichimoku'
        elif i in ['Ichimoku']:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['Close'], mode="markers+lines", name='Close',
                line=dict(color='yellow', width=2)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['ichimoku_a'], mode="lines", name='ichimoku_a',
                line=dict(color='magenta', width=1.5)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['ichimoku_b'], mode="lines", name='ichimoku_b',
                line=dict(color='magenta', width=1.5), fill='tonexty'
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['ichimoku_base'], mode="lines", name='ichimoku_base',
                line=dict(color='seashell', width=1.5, dash='dot')
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['ichimoku_conv'], mode="lines", name='ichimoku_conv',
                line=dict(color='limegreen', width=1.5)
            ), row=n, col=1)




        #  'MACD'
        elif i in ['MACD']:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['macd_line'], mode="lines", name='macd_line',
                line=dict(color='goldenrod', width=2)
            ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['macd_signal'], mode="lines", name='macd_singal',
                line=dict(color='crimson', width=2)
            ), row=n, col=1)

            colors = np.where(data['macd_hist'] < 0, 'red', 'lime')
            fig.append_trace(go.Bar(
                x=data["Date"], y=data['macd_hist'], name='macd_hist', marker_color=colors
            ), row=n, col=1)

        else:
            continue

    fig.update_layout(height=height, width=1100,
                      title_text="Trading Strategy - Trend Indicators", xaxis_rangeslider_visible=True,
                      yaxis_title="Price",
                      )
    fig.update_xaxes(rangeslider_visible=False)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='#ffffff', color='#ffffff')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#ffffff', mirror=True, color='#ffffff')
    fig.update_annotations(font=dict(color="#ffffff"))

    fig.update_layout(legend=dict(font=dict(color="#ffffff")), title_font=dict(color='#ffffff'), plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    return fig

# --------------------------------------------------

@app.callback(
    Output('graph_volume_efts', 'figure'),
    Input('dropdown_volume_efts', 'value'),
    Input('df_indicators_efts', 'dataframe'))

# --------------------------------------------------
def volume(lista, data):
    data=pd.DataFrame(data)

    n = 0
    height = 0
    fig = make_subplots(rows=len(lista), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=lista)
    for i in lista:

        n += 1
        height += 500
        # 1 com candle
        if i in ['VWAP']:
            fig.add_trace(go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="candlestick", ), row=n, col=1)

            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['vwap'], mode="lines", name='VWAP',
                line=dict(color='coral', width=2)
            ), row=n, col=1)


        # um indicador
        elif i in ['ADI'
                   ]:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['adi'], mode="lines", name='ADI',
                line=dict(color='salmon', width=2), showlegend=True
            ), row=n, col=1)


        elif i in ['OBV'
                   ]:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['obv'], mode="lines", name='OBV',
                line=dict(color='lightblue', width=2), showlegend=True
            ), row=n, col=1)

        elif i in ['CMF'
                   ]:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['cmf'], mode="lines", name='CMF',
                line=dict(color='lightgreen', width=2), showlegend=True
            ), row=n, col=1)
        elif i in ['FI'
                   ]:
            fig.append_trace(go.Scatter(
                x=data["Date"], y=data['fi'], mode="lines", name='FI',
                line=dict(color='firebrick', width=2), showlegend=True
            ), row=n, col=1)

        else:
            continue

    fig.update_layout(height=height, width=1100,
                      title_text="Trading Strategy - Volume Indicators", xaxis_rangeslider_visible=True,

                      )
    fig.update_xaxes(rangeslider_visible=False)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='#ffffff', color='#ffffff')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#ffffff', mirror=True, color='#ffffff')
    fig.update_annotations(font=dict(color="#ffffff"))

    fig.update_layout(legend=dict(font=dict(color="#ffffff")), title_font=dict(color='#ffffff'),
                      plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    return fig

# --------------------------------------------------

@app.callback(
    Output('df_models_efts', 'dataframe'),
    Output('pred_string_efts', 'children'),
    Input('dropdown_efts', 'value'),
    Input('hour_day_efts', 'value'))

# -----------------------------------------------------

def dataframes_assets_hour_day(asset, choice):
    asset = efts_sigla[asset]
    if choice == 'Per day':
        msft = yf.Ticker(asset).history(period='1y', interval='1d')
        pred_string = """Today's Prediction"""
    elif choice == 'Per hour':
        msft = yf.Ticker(asset).history(period='12d', interval='1h')
        pred_string = """Next Hour Prediction"""

    return msft.reset_index().rename(columns={'index': 'Date'}).to_dict('records'), pred_string

# ---------------------------------------------------------

@app.callback(
    Output('scal_rmse_efts', 'children'),
    Output('orig_rmse_efts', 'children'),
    Output('model_graph_efts', 'figure'),
    Output('today_pred_efts', 'children'),
    Input('df_models_efts', 'dataframe'),
    Input('dropdown_models_efts', 'value'),
    Input('price_efts', 'value'))

# -----------------------------------------------------

def build_model(data, model, price):
    data = pd.DataFrame(data)

    # creating new dataframe of the coin data containing only date and closing price
    close = data[['Date', price]]

    close.index = np.arange(1, len(close) + 1)
    close_stock = close.copy()

    # normalizing close price value
    del close['Date']
    scaler = MinMaxScaler(feature_range=(0, 1))
    close = scaler.fit_transform(np.array(close).reshape(-1, 1))

    # separate data for train and test
    training_size = int(len(close) * 0.70)
    test_size = len(close) - training_size
    train_data, test_data = close[0:training_size, :], close[training_size:len(close), :1]

    # we will be predicting the close price using the close prices of the 18 days before
    time_step = 18
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    model = models[model]

    model.fit(X_train, y_train)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # calculating rmse with scaled data to compare with the other assets

    scaled_rmse = round(math.sqrt(mean_squared_error(y_test, test_predict)), 4)

    train_predict = train_predict.reshape(-1, 1)
    test_predict = test_predict.reshape(-1, 1)

    # transform back to original form

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    # calculating rmse with original data

    original_rmse = round(math.sqrt(mean_squared_error(original_ytest, test_predict)), 4)

    # shift train predictions for plotting

    look_back = time_step
    train_predict_plot = np.empty_like(close)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    # shift test predictions for plotting
    test_predict_plot = np.empty_like(close)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(close) - 1, :] = test_predict

    names = cycle([f'Original {price} price', f'Train predicted {price} price', f'Test predicted {price} price'])

    plot_df = pd.DataFrame({'Date': close_stock['Date'],
                            'original_close': close_stock[price],
                            'train_predicted_close': train_predict_plot.reshape(1, -1)[0].tolist(),
                            'test_predicted_close': test_predict_plot.reshape(1, -1)[0].tolist()})

    # plotting the real close prices and the predicted ones
    fig = px.line(plot_df, x=plot_df['Date'],
                  y=[plot_df['original_close'], plot_df['train_predicted_close'], plot_df['test_predicted_close']],
                  labels={'value': 'Close price', 'Date': 'Date'},
                  color_discrete_sequence=["aqua", "orange", "firebrick"])

    fig.update_layout(title_text=f'Comparision between original {price} price vs predicted {price} price',
                      font_size=15, font_color='#ffffff', legend_title_text=f'{price} Price')

    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False, showline=True, linewidth=2, linecolor='#ffffff', color='#ffffff')
    fig.update_yaxes(showgrid=False, showline=True, linewidth=2, linecolor='#ffffff', mirror=True, color='#ffffff')
    fig.update_layout(legend=dict(font=dict(color="#ffffff")), title_font=dict(color='#ffffff'),
                      plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

    # prediction next day

    x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = time_step
    i = 0

    if len(temp_input) > time_step:

        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)

        yhat = model.predict(x_input)
        temp_input.extend(yhat.tolist())
        temp_input = temp_input[1:]

        lst_output.extend(yhat.tolist())

    else:

        yhat = model.predict(x_input)

        temp_input.extend(yhat.tolist())
        lst_output.extend(yhat.tolist())

    close = close.tolist()
    close.extend((np.array(lst_output).reshape(-1, 1)).tolist())
    close = scaler.inverse_transform(close).reshape(1, -1).tolist()[0]

    today_pred = round(close[-1], 4)

    return scaled_rmse, original_rmse, fig, today_pred


# -----------------------------------------------------

@app.callback(
    Output('comment_efts', 'children'),
    Input('p_value_efts', 'children'))

def result(p_value):
    if float(p_value) < 0.05 and float(p_value) < 0.1 and float(p_value) < 0.01:
        return 'The Time series is stationary with a significance level of 1%, 5% and 10%.'

    elif float(p_value) < 0.05 and float(p_value) > 0.1 and float(p_value) < 0.01:
        return 'The Time series is stationary with a significance level of 1% and 5%'

    elif float(p_value) > 0.05 and float(p_value) > 0.1 and float(p_value) < 0.01:
        return 'The Time series is stationary with a significance level of 1%.'

    elif float(p_value) > 0.05 and float(p_value) < 0.1 and float(p_value) < 0.01:
        return 'The Time series is stationary with a significance level of 1% and 10%.'

    elif float(p_value) < 0.05 and float(p_value) < 0.1 and float(p_value) > 0.01:
        return 'The Time series is stationary with a significance level of 5% and 10%.'

    elif float(p_value) < 0.05 and float(p_value) > 0.1 and float(p_value) > 0.01:
        return 'The Time series is stationary with a significance level of 5%'

    elif float(p_value) > 0.05 and float(p_value) < 0.1 and float(p_value) > 0.01:
        return 'The Time series is stationary with a significance level of 10%'

    elif float(p_value) > 0.05 and float(p_value) > 0.1 and float(p_value) > 0.01:
        return 'The Time series is not stationary with a significance level of 1%, 5% and 10%'
    # -------------------------------------------------- CALLBACK FOR SEPARATORS
@app.callback(
    Output("efts_data", "children"),
    [Input("efts_seps", "active_tab")]
)

def seperator_switch(choice):
    if choice == "main_efts_sep":
        return efts_main
    elif choice == "expo_efts_sep":
        return efts_expl
    elif choice == "tech_efts_sep":
        return efts_tech
    elif choice == "models_efts_sep":
        return efts_model
