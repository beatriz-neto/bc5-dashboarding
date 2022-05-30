# --------------------------------------------------  PACKAGES
from main import app
import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import html
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from datetime import datetime
from dash import Input, Output, dcc, html

# --------------------------------------------------  Connect to separators
from Crypto import crypto_layout
from Stocks import stocks_layout
from Mutual_Funds import funds_layout
from Futures import futures_layout
from Home import home_layout
from Currencies import currencies_layout
from efts import efts_layout

from main import app
from main import server
# -------------------------------------------------- CREATING DIFFERENT SEPARATORS
# app = dash.Dash(__name__,  external_stylesheets=[dbc.themes.SUPERHERO])
# server = app.server

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#2B303A",
    "color":"#FFFFF"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "color":"#FFFFF",

}

sidebar = html.Div(
    [
        dbc.Col([
         html.H2(dbc.Row([dbc.Col(html.Img(src='assets/4 Assets.png', height="210px"))]))],className='center'),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(dbc.Row([dbc.Col(html.Img(src='assets/home.png.png'),width=2,),dbc.Col("Home"),],justify="start",align="center",),id="home-nav",href="/",active="exact",),
                dbc.NavLink(dbc.Row([dbc.Col(html.Img(src='assets/crypto.png',height='20px'),width=2,),dbc.Col("Cryptocurrency"),],justify="start",align="center",),id="crypto-nav",href="/page-1",active="exact",),
                dbc.NavLink(dbc.Row([dbc.Col(html.Img(src='assets/stocks.png',height='22px'),width=2,),dbc.Col("Stocks"),],justify="start",align="center",),id="stocks-nav",href="/page-2",active="exact",),
                dbc.NavLink(dbc.Row([dbc.Col(html.Img(src="assets/funds.png",height='25px'),width=2,),dbc.Col("Mutual Funds"),],justify="start",align="center",),id="funds-nav",href="/page-3",active="exact",),
                dbc.NavLink(dbc.Row([dbc.Col(html.Img(src="assets/efts.png",height='33px'),width=2,),dbc.Col("ETFS"),],justify="start",align="center",),id="efts-nav",href="/page-4",active="exact",),
                dbc.NavLink(dbc.Row([dbc.Col(html.Img(src="assets/coin.png",height='22px'),width=2,),dbc.Col("Currencies"),],justify="start",align="center",),id="currencies-nav",href="/page-5",active="exact",),
                dbc.NavLink(dbc.Row([dbc.Col(html.Img(src="assets/futures.png",height='25px'),width=2,),dbc.Col("Futures"),],justify="start",align="center",),id="futures-nav",href="/page-6",active="exact",),
            ],
            vertical=True,
            pills=True,
        ),

    ],
    style=SIDEBAR_STYLE,

)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"),sidebar,content])

#
#
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])

#

def render_page_content(pathname):
    if pathname == "/":
        return home_layout
    elif pathname == "/page-1":
        return crypto_layout
    elif pathname == "/page-2":
        return stocks_layout
    elif pathname == "/page-3":
        return funds_layout
    elif pathname == "/page-4":
        return efts_layout
    elif pathname == "/page-5":
        return currencies_layout
    elif pathname == "/page-6":
        return futures_layout
#     # If the user tries to reach a different page, return a 404 message
#     return dbc.Jumbotron(
#         [
#             html.H1("404: Not found", className="text-danger"),
#             html.Hr(),
#             html.P(f"The pathname {pathname} was not recognised..."),
#         ]
#     )
#dbc.Tab(label="Socials", tab_id="soc_sep", labelClassName="text-success font-weight-bold",
 #       activeLabelClassName="text-danger"),

if __name__ == '__main__':
    app.run_server(debug=True)
